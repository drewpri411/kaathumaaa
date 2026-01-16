"""Audio mixer for multi-channel audio."""
import asyncio
import numpy as np
from collections import deque
from typing import Optional

from .config import config


class AudioMixer:
    """
    Multi-channel audio mixer.
    
    Channels:
    - Primary: Agent main speech (100% volume)
    - Secondary: Backchannels (50% volume)
    
    Features:
    - Real-time mixing
    - Volume control per channel
    - Output buffering
    """
    
    def __init__(self):
        """Initialize audio mixer."""
        self.sample_rate = config.sample_rate
        
        # Channel buffers
        self.primary_buffer = deque()
        self.secondary_buffer = deque()
        
        # Output buffer
        self.output_buffer = deque()
        
        # Volume levels
        self.primary_volume = 1.0
        self.secondary_volume = config.backchannel_volume
        
        self._lock = asyncio.Lock()
        
        # Mixing task
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start mixer."""
        self.running = True
        self.task = asyncio.create_task(self._mixing_loop())
        print("✓ Audio mixer started")
    
    async def stop(self) -> None:
        """Stop mixer."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        print("✓ Audio mixer stopped")
    
    async def add_primary_audio(self, audio: np.ndarray) -> None:
        """
        Add audio to primary channel.
        
        Args:
            audio: Audio samples
        """
        async with self._lock:
            self.primary_buffer.extend(audio * self.primary_volume)
    
    async def add_secondary_audio(self, audio: np.ndarray) -> None:
        """
        Add audio to secondary channel (backchannels).
        
        Args:
            audio: Audio samples
        """
        async with self._lock:
            self.secondary_buffer.extend(audio * self.secondary_volume)
    
    async def _mixing_loop(self) -> None:
        """Main mixing loop."""
        while self.running:
            try:
                await self._mix_channels()
                await asyncio.sleep(0.01)  # 10ms mixing interval
            except Exception as e:
                print(f"Mixer error: {e}")
                await asyncio.sleep(0.1)
    
    async def _mix_channels(self) -> None:
        """Mix channels into output buffer."""
        async with self._lock:
            # Determine how many samples to mix
            primary_len = len(self.primary_buffer)
            secondary_len = len(self.secondary_buffer)
            
            if primary_len == 0 and secondary_len == 0:
                return
            
            # Mix up to the minimum length (or all if one is empty)
            mix_len = max(primary_len, secondary_len)
            
            if mix_len == 0:
                return
            
            # Get samples from each channel
            primary_samples = []
            secondary_samples = []
            
            for _ in range(min(mix_len, primary_len)):
                primary_samples.append(self.primary_buffer.popleft())
            
            for _ in range(min(mix_len, secondary_len)):
                secondary_samples.append(self.secondary_buffer.popleft())
            
            # Pad shorter channel with zeros
            if len(primary_samples) < mix_len:
                primary_samples.extend([0.0] * (mix_len - len(primary_samples)))
            
            if len(secondary_samples) < mix_len:
                secondary_samples.extend([0.0] * (mix_len - len(secondary_samples)))
            
            # Mix (simple addition with clipping)
            primary_arr = np.array(primary_samples, dtype=np.float32)
            secondary_arr = np.array(secondary_samples, dtype=np.float32)
            
            mixed = primary_arr + secondary_arr
            
            # Clip to [-1, 1]
            mixed = np.clip(mixed, -1.0, 1.0)
            
            # Add to output buffer
            self.output_buffer.extend(mixed)
    
    def get_output_audio(self, num_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get mixed audio from output buffer.
        
        Args:
            num_samples: Number of samples to retrieve
        
        Returns:
            Mixed audio or None if buffer empty
        """
        if not self.output_buffer:
            return None
        
        if num_samples is None:
            num_samples = len(self.output_buffer)
        
        num_samples = min(num_samples, len(self.output_buffer))
        
        samples = []
        for _ in range(num_samples):
            samples.append(self.output_buffer.popleft())
        
        return np.array(samples, dtype=np.float32)
    
    def clear_buffers(self) -> None:
        """Clear all buffers."""
        self.primary_buffer.clear()
        self.secondary_buffer.clear()
        self.output_buffer.clear()
