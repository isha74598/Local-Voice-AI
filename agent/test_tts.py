import asyncio
from kokoro_tts_adapter import KokoroLiveKitTTS

async def test():
    tts = KokoroLiveKitTTS()
    print("Testing TTS synthesize...")
    async for frame in tts.synthesize('Hello, this is a test.'):
        print(f'Generated audio frame: {len(frame.data)} bytes, sample_rate={frame.sample_rate}, num_channels={frame.num_channels}')
    print('✅ TTS synthesize works!')

if __name__ == "__main__":
    asyncio.run(test())
