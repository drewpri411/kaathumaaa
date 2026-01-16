"""Backchannel audio library management."""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import wave

from .config import config


class BackchannelLibrary:
    """
    Manages backchannel audio files.
    
    Features:
    - Load WAV files from directory
    - Validate format (16kHz mono)
    - Normalize volume
    - In-memory storage for low latency
    """
    
    def __init__(self):
        """Initialize backchannel library."""
        self.sample_rate = config.sample_rate
        self.volume = config.backchannel_volume
        self.backchannel_dir = config.backchannel_dir
        
        # Storage: name -> audio data
        self.backchannels: Dict[str, np.ndarray] = {}
        
        # Load backchannels
        self._load_backchannels()
    
    def _load_backchannels(self) -> None:
        """Load all backchannel WAV files."""
        if not self.backchannel_dir.exists():
            print(f"WARNING: Backchannel directory not found: {self.backchannel_dir}")
            print("  Run generate_backchannels.py to create backchannel audio files")
            return
        
        # Find all WAV files
        wav_files = list(self.backchannel_dir.glob("*.wav"))
        
        if not wav_files:
            print(f"WARNING: No backchannel WAV files found in {self.backchannel_dir}")
            return
        
        # Load each file
        for wav_path in wav_files:
            try:
                audio = self._load_wav(wav_path)
                name = wav_path.stem  # filename without extension
                self.backchannels[name] = audio
                print(f"✓ Loaded backchannel: {name} ({len(audio)/self.sample_rate:.2f}s)")
            except Exception as e:
                print(f"✗ Failed to load {wav_path.name}: {e}")
    
    def _load_wav(self, path: Path) -> np.ndarray:
        """
        Load WAV file and validate format.
        
        Args:
            path: Path to WAV file
        
        Returns:
            Audio data as numpy array
        """
        with wave.open(str(path), 'rb') as wav:
            # Validate format
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            
            if channels != 1:
                raise ValueError(f"Expected mono audio, got {channels} channels")
            
            if framerate != self.sample_rate:
                raise ValueError(f"Expected {self.sample_rate}Hz, got {framerate}Hz")
            
            # Read audio data
            frames = wav.readframes(wav.getnframes())
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio = np.frombuffer(frames, dtype=np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to float32 [-1, 1]
            audio = audio.astype(np.float32) / 32768.0
            
            # Normalize volume
            audio = audio * self.volume
            
            return audio
    
    def get_backchannel(self, name: str) -> Optional[np.ndarray]:
        """
        Get backchannel audio by name.
        
        Args:
            name: Backchannel name (e.g., "mmhmm")
        
        Returns:
            Audio data or None if not found
        """
        return self.backchannels.get(name)
    
    def get_all_names(self) -> List[str]:
        """Get list of all backchannel names."""
        return list(self.backchannels.keys())
    
    def get_duration(self, name: str) -> Optional[float]:
        """
        Get duration of backchannel in seconds.
        
        Args:
            name: Backchannel name
        
        Returns:
            Duration in seconds or None if not found
        """
        audio = self.backchannels.get(name)
        if audio is not None:
            return len(audio) / self.sample_rate
        return None
    
    def validate_files(self) -> bool:
        """
        Validate that all expected backchannels are present.
        
        Returns:
            True if all expected files found
        """
        expected = ["mmhmm", "okay", "yeah", "i_see", "right"]
        
        for name in expected:
            if name not in self.backchannels:
                print(f"WARNING: Missing backchannel: {name}")
                return False
        
        return True
