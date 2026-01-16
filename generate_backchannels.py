"""Generate backchannel audio files using OpenAI TTS."""
import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import AsyncOpenAI


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
                response_format="wav"
            )
            
            # Save to file
            audio_bytes = await response.aread()
            output_path.write_bytes(audio_bytes)
            
            # Get file size
            size_kb = len(audio_bytes) / 1024
            
            print(f"   ✅ Saved to {output_path} ({size_kb:.1f} KB)")
        
        except Exception as e:
            print(f"   ❌ Error generating {name}: {e}")
    
    print("=" * 60)
    print("✅ Backchannel generation complete!")
    print(f"📁 Files saved to: {backchannel_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(generate_backchannels())
