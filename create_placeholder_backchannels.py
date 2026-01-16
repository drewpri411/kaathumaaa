"""Create placeholder backchannel audio files (no API key required)."""
import numpy as np
import wave
from pathlib import Path


def create_placeholder_backchannels():
    """Create simple tone-based placeholder backchannels."""
    
    # Create backchannels directory
    backchannel_dir = Path("backchannels")
    backchannel_dir.mkdir(exist_ok=True)
    
    # Backchannel names
    backchannels = ["mmhmm", "okay", "yeah", "i_see", "right"]
    
    print("🎵 Creating placeholder backchannel audio files...")
    print("=" * 60)
    print("NOTE: These are simple tone placeholders.")
    print("For better quality, run generate_backchannels.py with OpenAI API key.")
    print("=" * 60)
    
    sample_rate = 16000
    duration = 0.3  # 300ms
    
    for i, name in enumerate(backchannels):
        output_path = backchannel_dir / f"{name}.wav"
        
        if output_path.exists():
            print(f"⏭️  Skipping {name} (already exists)")
            continue
        
        # Create a simple tone (different frequency for each)
        frequency = 440 + (i * 50)  # A4, then higher
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate tone with envelope to avoid clicks
        tone = np.sin(2 * np.pi * frequency * t)
        
        # Apply fade in/out envelope
        envelope = np.ones_like(tone)
        fade_samples = int(sample_rate * 0.05)  # 50ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        tone = tone * envelope
        
        # Convert to int16
        audio_int16 = (tone * 16000).astype(np.int16)
        
        # Write WAV file
        with wave.open(str(output_path), 'w') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        print(f"✅ Created: {name}.wav ({frequency}Hz tone)")
    
    print("=" * 60)
    print("✅ Placeholder backchannels created!")
    print(f"📁 Files saved to: {backchannel_dir.absolute()}")
    print("\n💡 TIP: Replace these with real audio using:")
    print("   python generate_backchannels.py")


if __name__ == "__main__":
    create_placeholder_backchannels()
