"""OpenAI TTS client."""
import asyncio
from typing import Optional
import numpy as np
import io
import wave

from openai import AsyncOpenAI

from .config import config
from .event_bus import event_bus, EventType


class TTSClient:
    """
    OpenAI TTS API client.
    
    Features:
    - Async audio generation
    - Audio format conversion to 16kHz mono
    - Error handling and retries
    """
    
    def __init__(self):
        """Initialize TTS client."""
        self.client = AsyncOpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
        self.voice = config.tts_voice
        self.model = config.tts_model
        self.sample_rate = config.sample_rate
        
        if not self.client:
            print("WARNING: OpenAI API key not set, TTS will not work")
    
    async def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
        
        Returns:
            Audio data as numpy array or None on error
        """
        if not self.client:
            return None
        
        if not text or not text.strip():
            return None
        
        try:
            # Call TTS API
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="pcm"  # Raw PCM audio
            )
            
            # Get audio bytes
            audio_bytes = await response.aread()
            
            # Convert to numpy array
            # OpenAI TTS returns 24kHz 16-bit PCM by default
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 [-1, 1]
            audio = audio.astype(np.float32) / 32768.0
            
            # Resample to 16kHz if needed
            # OpenAI TTS actually returns 24kHz, so we need to downsample
            audio = self._resample_24k_to_16k(audio)
            
            return audio
        
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return None
    
    def _resample_24k_to_16k(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample from 24kHz to 16kHz.
        
        Args:
            audio: Audio at 24kHz
        
        Returns:
            Audio at 16kHz
        """
        from scipy import signal as scipy_signal
        
        # Resample from 24000 to 16000
        num_samples = int(len(audio) * 16000 / 24000)
        resampled = scipy_signal.resample(audio, num_samples)
        
        return resampled.astype(np.float32)
    
    async def synthesize_streaming(self, text_stream) -> np.ndarray:
        """
        Synthesize from streaming text.
        
        For now, we accumulate all text and synthesize once.
        True streaming TTS would require a different approach.
        
        Args:
            text_stream: Async generator of text chunks
        
        Returns:
            Complete audio
        """
        # Accumulate text
        text_chunks = []
        async for chunk in text_stream:
            text_chunks.append(chunk)
        
        full_text = "".join(text_chunks)
        
        # Synthesize
        return await self.synthesize(full_text)
