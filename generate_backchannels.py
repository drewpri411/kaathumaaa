"""Generate backchannel audio files using OpenAI TTS."""
import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import AsyncOpenAI
import numpy as np
import wave
from scipy import signal


async def generate_backchannels():
    """Generate all backchannel audio files."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("Please add your OpenAI API key to .env file first")
        return
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Create backchannels directory
    backchannel_dir = Path("backchannels")
    backchannel_dir.mkdir(exist_ok=True)
    
    # Backchannel definitions
    backchannels = {
        "mmhmm": "Mm-hmm",
        "okay": "Okay",
        "yeah": "Yeah",
        "i_see": "I see",
        "right": "Right"
    }
    
    print("🎙️  Generating backchannel audio files...")
    print("=" * 60)
    
    for name, text in backchannels.items():
        output_path = backchannel_dir / f"{name}.wav"
        
        if output_path.exists():
            print(f"⏭️  Skipping {name} (already exists)")
            continue
        
        try:
            print(f"🔊 Generating: {name} ('{text}')...")
            
            # Generate audio using TTS
            response = await client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="pcm"  # Get raw PCM data for resampling
            )
            
            # Get audio bytes
            audio_bytes = await response.aread()
            
            # Convert bytes to numpy array (24kHz, 16-bit PCM)
            audio_24k = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Resample from 24kHz to 16kHz
            num_samples_16k = int(len(audio_24k) * 16000 / 24000)
            audio_16k = signal.resample(audio_24k, num_samples_16k).astype(np.int16)
            
            # Save as WAV file at 16kHz
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_16k.tobytes())
            
            # Get file size
            size_kb = output_path.stat().st_size / 1024
            
            print(f"   ✅ Saved to {output_path} ({size_kb:.1f} KB, 16kHz)")
        
        except Exception as e:
            print(f"   ❌ Error generating {name}: {e}")
    
    print("=" * 60)
    print("✅ Backchannel generation complete!")
    print(f"📁 Files saved to: {backchannel_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(generate_backchannels())
