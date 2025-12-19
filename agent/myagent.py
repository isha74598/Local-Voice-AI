import logging
import os
import asyncio
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.agents import ChatContext, ChatMessage
from livekit.agents.stt import STT, STTCapabilities, SpeechEvent, SpeechEventType, SpeechData
from livekit import rtc

from livekit.plugins import silero
from livekit.plugins import openai
from kokoro_tts_adapter import KokoroLiveKitTTS
from camera_vision import CameraVision


from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

load_dotenv()

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger("local-agent")
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------
# Load embedding + docs (RAG)
# -------------------------------------------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
docs_dir = os.path.join(os.path.dirname(__file__), "docs")
if os.path.exists(docs_dir):
    for fn in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, fn), encoding="utf-8") as f:
            docs.append(f.read())

if docs:
    embs = embed_model.encode(docs, show_progress_bar=False)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
else:
    dim = embed_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    logger.warning("No documents found in docs directory. RAG disabled.")

async def rag_lookup(query: str) -> str:
    loop = asyncio.get_running_loop()
    q_emb = await loop.run_in_executor(None, lambda: embed_model.encode([query]))
    D, I = await loop.run_in_executor(None, lambda: index.search(q_emb, min(3, index.ntotal)))
    if index.ntotal == 0:
        return ""
    return "\n\n---\n\n".join(docs[i] for i in I[0])

def is_vision_query(text: str) -> bool:
    """
    Check if the user's query is asking about what the camera can see.
    """
    vision_keywords = [
        "what can you see",
        "what do you see",
        "what's in the camera",
        "what can you see from the camera",
        "what do you see from the camera",
        "describe what you see",
        "what is visible",
        "what's in front",
        "what's in the frame",
        "what can the camera see",
        "describe the camera",
        "what's on camera",
        "show me what you see",
        "tell me what you see",
        "camera",
        "see",
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in vision_keywords)

# -------------------------------------------------------------------
# LOCAL WHISPER STT (correct LiveKit-compatible adapter)
# -------------------------------------------------------------------
class LocalWhisperSTT(STT):
    def __init__(self):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
        )

    async def _recognize_impl(
        self,
        buffer,
        *,
        language=None,
        conn_options=None,
    ) -> SpeechEvent:
        # Combine audio frames and convert to numpy array
        combined_frame = rtc.combine_audio_frames(buffer)
        
        # Convert to numpy array (faster-whisper expects float32 in range [-1, 1])
        audio_data = np.frombuffer(combined_frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Run transcription in executor to avoid blocking
        # Use English language explicitly
        lang = language if language else "en"
        loop = asyncio.get_running_loop()
        def transcribe():
            segments, info = self._whisper_model.transcribe(audio_data, language=lang)
            return list(segments), info
        
        segments, info = await loop.run_in_executor(None, transcribe)
        
        # Combine all segments into final text
        text = " ".join(seg.text.strip() for seg in segments)
        
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    language="en",
                    text=text,
                )
            ],
        )

# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------
class LocalAgent(Agent):
    def __init__(self, use_vision_model: bool = False) -> None:
        stt = LocalWhisperSTT()

        # Use vision model if available, otherwise use regular model
        # Try llava (vision model) first, fallback to llama3
        model_name = "llava" if use_vision_model else "llama3"
        
        llm = openai.LLM(
            base_url="http://localhost:11434/v1",
            model=model_name,
            timeout=30,
        )

        tts = KokoroLiveKitTTS(
            voice="af_sky",
            sample_rate=24000,
        )

        vad = silero.VAD.load()
        
        # Initialize camera vision
        try:
            self.camera = CameraVision(device_path="/dev/video0")
            if self.camera.cap is not None and self.camera.cap.isOpened():
                logger.info("Camera vision initialized successfully at /dev/video0")
            else:
                logger.warning("Camera opened but not ready, will retry on first use")
                self.camera = None
        except Exception as e:
            logger.warning(f"Could not initialize camera: {e}")
            logger.info("Camera will not be available. Make sure /dev/video0 exists and is not in use by another app.")
            self.camera = None

        super().__init__(
            instructions="""
You are a helpful voice assistant with camera vision capabilities.
You can see what the camera captures and describe it when asked.
Never use emojis.
Keep responses short and natural.
Speak clearly.
When asked about what you can see, describe the camera view in detail.
""",
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        user_text = new_message.text_content.lower() if new_message.text_content else ""
        
        # Check if this is a vision query
        if is_vision_query(user_text):
            # Try to initialize camera if not already done
            if self.camera is None:
                try:
                    logger.info("Initializing camera for vision query...")
                    self.camera = CameraVision(device_path="/dev/video0")
                    if self.camera.cap is None or not self.camera.cap.isOpened():
                        self.camera = None
                        raise RuntimeError("Camera not accessible")
                except Exception as e:
                    logger.error(f"Failed to initialize camera: {e}")
                    self.camera = None
            
            if self.camera is not None:
                try:
                    # Capture frame from camera
                    logger.info("Capturing frame from camera for vision query")
                    frame_base64 = await asyncio.get_running_loop().run_in_executor(
                        None, 
                        self.camera.get_latest_frame_base64
                    )
                    
                    if frame_base64:
                        # Use vision model to actually see and describe the image
                        logger.info("Sending image to vision model for description...")
                        vision_description = await describe_image_with_vision_model(
                            image_base64=frame_base64,
                            question=new_message.text_content,
                            model="qwen2.5vl:7b",  # Use available vision model
                            base_url="http://localhost:11434"
                        )
                        
                        if vision_description:
                            # Add the actual vision description to the conversation
                            vision_prompt = (
                                f"User asked: '{new_message.text_content}'\n\n"
                                f"I have captured a live frame from the camera and analyzed it with a vision model. "
                                f"Here's what I can see:\n\n{vision_description}\n\n"
                                f"Please respond to the user's question based on this description."
                            )
                            
                            turn_ctx.add_message(
                                role="user",
                                content=vision_prompt,
                            )
                            logger.info(f"Added vision description from model: {len(vision_description)} chars")
                        else:
                            # Fallback if vision model fails
                            logger.warning("Vision model failed, using fallback description")
                            turn_ctx.add_message(
                                role="user",
                                content=(
                                    f"User asked: '{new_message.text_content}'\n\n"
                                    "I captured a frame from the camera but couldn't process it with the vision model. "
                                    "The camera is active at /dev/video0. Please describe what you might typically see in a camera view."
                                ),
                            )
                    else:
                        logger.warning("Failed to capture camera frame")
                        turn_ctx.add_message(
                            role="user",
                            content=f"I tried to capture from the camera but the frame capture failed. The camera might be in use by another application. Please close any apps using the camera (like VLC) and try again. Original question: {new_message.text_content}",
                        )
                except Exception as e:
                    logger.error(f"Error processing vision query: {e}")
                    turn_ctx.add_message(
                        role="user",
                        content=f"Error accessing camera: {str(e)}. Make sure /dev/video0 exists and is not in use. Original question: {new_message.text_content}",
                    )
            else:
                # Camera not available
                logger.warning("Camera not available for vision query")
                turn_ctx.add_message(
                    role="user",
                    content=f"I cannot access the camera at /dev/video0 right now. Please make sure: 1) The camera device exists, 2) No other application is using it (close VLC if open), 3) You have permission to access it. Original question: {new_message.text_content}",
                )
        
        # RAG lookup for document context
        rag = await rag_lookup(new_message.text_content)
        if rag:
            turn_ctx.add_message(
                role="user",
                content=f"Additional context:\n{rag}",
            )

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession()
    await session.start(
        agent=LocalAgent(),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            job_memory_warn_mb=1500,
        )
    )
