"""Audio pipeline for buffering and processing audio streams."""
import numpy as np
from collections import deque
from typing import Optional, Generator
import asyncio
from scipy import signal as scipy_signal

from .config import config
from .event_bus import event_bus, EventType


class AudioPipeline:
    """
    Manages audio buffering and processing.
    
    Features:
    - Circular buffer for recent audio (30 seconds)
    - VAD chunk accumulator (30ms chunks)
    - Whisper chunk accumulator (1.5s with 0.5s overlap)
    - Audio format conversion
    - Bidirectional audio flow
    """
    
    def __init__(self):
        """Initialize audio pipeline."""
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        
        # Circular buffer for 30 seconds of audio
        self.buffer_duration_s = 30
        self.buffer_size = self.sample_rate * self.buffer_duration_s
        self.circular_buffer = deque(maxlen=self.buffer_size)
        
        # VAD chunk accumulator (30ms chunks)
        self.vad_chunk_size = config.get_chunk_size_samples()
        self.vad_accumulator = []
        
        # Whisper chunk accumulator (1.5s chunks with 0.5s overlap)
        self.whisper_chunk_size = config.get_whisper_chunk_size_samples()
        self.whisper_overlap_size = config.get_whisper_overlap_samples()
        self.whisper_accumulator = []
        self.whisper_last_chunk = []
        
        # Output buffer for playback
        self.output_buffer = deque()
        
        self._lock = asyncio.Lock()
    
    async def receive_audio(self, audio_data: np.ndarray) -> None:
        """
        Receive raw audio from WebRTC.
        
        Args:
            audio_data: Audio samples as numpy array
        """
        async with self._lock:
            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = self.convert_to_mono(audio_data)
            
            # Debug: Log audio reception
            audio_level = np.abs(audio_data).max()
            print(f"📥 Audio received: {len(audio_data)} samples, level: {audio_level:.4f}")
            
            # Add to circular buffer
            self.circular_buffer.extend(audio_data)
            
            # Add to accumulators
            self.vad_accumulator.extend(audio_data)
            self.whisper_accumulator.extend(audio_data)
            
            # Emit event
            await event_bus.emit(EventType.AUDIO_CHUNK_RECEIVED, {
                "samples": len(audio_data),
                "duration_ms": len(audio_data) / self.sample_rate * 1000
            })
    
    def resample_audio(self, audio: np.ndarray, original_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio samples
            original_rate: Original sample rate
        
        Returns:
            Resampled audio at target rate
        """
        if original_rate == self.sample_rate:
            return audio
        
        # Calculate resampling ratio
        num_samples = int(len(audio) * self.sample_rate / original_rate)
        resampled = scipy_signal.resample(audio, num_samples)
        return resampled.astype(np.float32)
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo/multi-channel audio to mono.
        
        Args:
            audio: Audio samples (can be multi-channel)
        
        Returns:
            Mono audio
        """
        if len(audio.shape) == 1:
            return audio
        
        # Average across channels
        return np.mean(audio, axis=1).astype(np.float32)
    
    def get_vad_chunks(self) -> Generator[np.ndarray, None, None]:
        """
        Get 30ms chunks for VAD processing.
        
        Yields:
            Audio chunks of 30ms duration
        """
        while len(self.vad_accumulator) >= self.vad_chunk_size:
            chunk = np.array(self.vad_accumulator[:self.vad_chunk_size], dtype=np.float32)
            self.vad_accumulator = self.vad_accumulator[self.vad_chunk_size:]
            yield chunk
    
    def get_whisper_chunks(self) -> Generator[np.ndarray, None, None]:
        """
        Get 1.5s chunks with 0.5s overlap for Whisper.
        
        Yields:
            Audio chunks of 1.5s duration with overlap
        """
        while len(self.whisper_accumulator) >= self.whisper_chunk_size:
            # Get chunk
            chunk = np.array(self.whisper_accumulator[:self.whisper_chunk_size], dtype=np.float32)
            
            # Move forward by (chunk_size - overlap_size)
            step_size = self.whisper_chunk_size - self.whisper_overlap_size
            self.whisper_accumulator = self.whisper_accumulator[step_size:]
            
            yield chunk
    
    async def add_output_audio(self, audio: np.ndarray) -> None:
        """
        Add audio to output buffer for playback.
        
        Args:
            audio: Audio samples to play
        """
        async with self._lock:
            self.output_buffer.extend(audio)
    
    def get_output_audio(self, num_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get audio from output buffer.
        
        Args:
            num_samples: Number of samples to retrieve (None for all)
        
        Returns:
            Audio samples or None if buffer empty
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
        self.circular_buffer.clear()
        self.vad_accumulator.clear()
        self.whisper_accumulator.clear()
        self.whisper_last_chunk.clear()
        self.output_buffer.clear()
    
    def get_buffer_fill_level(self) -> dict:
        """
        Get fill levels of all buffers.
        
        Returns:
            Dictionary with buffer fill percentages
        """
        return {
            "circular_buffer": len(self.circular_buffer) / self.buffer_size * 100,
            "vad_accumulator": len(self.vad_accumulator) / self.vad_chunk_size * 100,
            "whisper_accumulator": len(self.whisper_accumulator) / self.whisper_chunk_size * 100,
            "output_buffer": len(self.output_buffer)
        }
    
    def get_recent_audio(self, duration_s: float) -> np.ndarray:
        """
        Get recent audio from circular buffer.
        
        Args:
            duration_s: Duration in seconds
        
        Returns:
            Recent audio samples
        """
        num_samples = int(duration_s * self.sample_rate)
        num_samples = min(num_samples, len(self.circular_buffer))
        
        # Get last N samples
        recent = list(self.circular_buffer)[-num_samples:]
        return np.array(recent, dtype=np.float32)
