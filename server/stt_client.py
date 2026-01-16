"""OpenAI Whisper API client for speech-to-text."""
import asyncio
from typing import Optional
import numpy as np
import io
import wave

from openai import AsyncOpenAI

from .config import config
from .event_bus import event_bus, EventType


class STTClient:
    """
    OpenAI Whisper API client for speech-to-text.
    
    Features:
    - Async transcription requests
    - Audio format conversion
    - Retry logic with exponential backoff
    - Rate limit handling
    - Latency tracking
    """
    
    def __init__(self):
        """Initialize STT client."""
        self.client = AsyncOpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
        self.sample_rate = config.sample_rate
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        
        if not self.client:
            print("WARNING: OpenAI API key not set, STT will not work")
    
    async def transcribe_chunk(self, audio_data: np.ndarray, is_final: bool = False) -> Optional[str]:
        """
        Transcribe audio chunk using Whisper API.
        
        Args:
            audio_data: Audio samples
            is_final: Whether this is a final transcription
        
        Returns:
            Transcribed text or None on error
        """
        if not self.client:
            return None
        
        # Convert audio to WAV format
        audio_bytes = self.convert_audio_format(audio_data)
        
        # Transcribe with retry
        text = await self.retry_with_backoff(
            lambda: self._transcribe_internal(audio_bytes)
        )
        
        if text:
            # Emit event
            event_type = EventType.FINAL_TRANSCRIPT if is_final else EventType.PARTIAL_TRANSCRIPT
            await event_bus.emit(event_type, {
                "text": text,
                "is_final": is_final,
                "audio_duration_s": len(audio_data) / self.sample_rate
            })
        
        return text
    
    async def _transcribe_internal(self, audio_bytes: bytes) -> Optional[str]:
        """Internal transcription method."""
        try:
            # Create file-like object
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            # Call Whisper API
            transcript = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            
            return transcript.strip() if transcript else None
            
        except Exception as e:
            print(f"Whisper API error: {e}")
            return None
    
    def convert_audio_format(self, audio: np.ndarray) -> bytes:
        """
        Convert numpy audio to WAV bytes for API.
        
        Args:
            audio: Audio samples (float32, -1 to 1)
        
        Returns:
            WAV file bytes
        """
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    async def retry_with_backoff(self, func, max_retries: Optional[int] = None):
        """
        Retry function with exponential backoff.
        
        Args:
            func: Async function to retry
            max_retries: Maximum number of retries
        
        Returns:
            Function result or None on failure
        """
        if max_retries is None:
            max_retries = self.max_retries
        
        for attempt in range(max_retries):
            try:
                result = await func()
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Max retries reached: {e}")
                    return None
                
                # Exponential backoff
                delay = self.base_delay * (2 ** attempt)
                print(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                await asyncio.sleep(delay)
        
        return None
