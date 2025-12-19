import os
import numpy as np
import kokoro_tts
import asyncio

from kokoro_tts import Kokoro
from livekit.agents.tts import TTS, TTSCapabilities, ChunkedStream, AudioEmitter
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.rtc import AudioFrame


class KokoroChunkedStream(ChunkedStream):
    def __init__(self, *, tts, input_text: str, conn_options: APIConnectOptions, kokoro, voice: str):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._kokoro = kokoro
        self._voice = voice

    async def _run(self, output_emitter: AudioEmitter) -> None:
        # kokoro.create returns (audio_array, sample_rate) tuple
        # Use English language explicitly
        # Run in executor to avoid blocking
        loop = asyncio.get_running_loop()
        def create_audio():
            return self._kokoro.create(
                text=self._input_text,
                voice=self._voice,
                lang="en-us",  # Explicitly set to English (US)
            )
        
        audio_array, _ = await loop.run_in_executor(None, create_audio)

        # Convert float32 audio array to int16 PCM
        # Audio array is normalized to [-1, 1], convert to int16 range
        pcm = (audio_array * 32767).astype(np.int16)
        
        # Convert to bytes (int16 is 2 bytes per sample)
        pcm_bytes = pcm.tobytes()

        # Initialize the AudioEmitter
        output_emitter.initialize(
            request_id="",
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",  # Raw PCM audio
        )

        # Push the audio data as bytes
        output_emitter.push(pcm_bytes)
        
        # Flush to ensure all data is sent
        output_emitter.flush()


class KokoroLiveKitTTS(TTS):
    def __init__(
        self,
        voice: str = "af_sky",
        sample_rate: int = 24000,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=1,
            capabilities=TTSCapabilities(streaming=False),
        )

        base_dir = os.path.dirname(kokoro_tts.__file__)

        # kokoro_onnx expects file paths, not directories
        model_path = os.path.join(base_dir, "models", "kokoro-v1.0.onnx")
        voices_path = os.path.join(base_dir, "voices")  # This is already a file

        self.kokoro = Kokoro(
            model_path=model_path,
            voices_path=voices_path,
        )

        self.voice = voice

    def synthesize(
        self, 
        text: str, 
        *, 
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return KokoroChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            kokoro=self.kokoro,
            voice=self.voice,
        )
