"""Voice Activity Detection using Silero VAD."""
import numpy as np
import onnxruntime as ort
from enum import Enum
from datetime import datetime
from typing import Optional
import asyncio

from .config import config
from .event_bus import event_bus, EventType


class VADState(Enum):
    """VAD state machine states."""
    NOT_SPEAKING = "not_speaking"
    SPEAKING = "speaking"
    SILENCE_AFTER_SPEECH = "silence_after_speech"


class VADProcessor:
    """
    Voice Activity Detection using Silero VAD ONNX model.
    
    Features:
    - Silero VAD model inference
    - State machine with hysteresis
    - Speech/silence duration tracking
    - Event emission for state transitions
    """
    
    def __init__(self):
        """Initialize VAD processor."""
        self.sample_rate = config.sample_rate
        self.threshold = config.vad_threshold
        self.min_speech_duration_ms = config.vad_min_speech_duration_ms
        self.min_silence_duration_ms = config.vad_min_silence_duration_ms
        
        # State tracking
        self.current_state = VADState.NOT_SPEAKING
        self.speech_start_time: Optional[datetime] = None
        self.silence_start_time: Optional[datetime] = None
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        
        # Hysteresis thresholds
        self.speech_start_threshold = 3  # chunks
        self.speech_end_threshold = 5    # chunks
        
        # ONNX model
        self.session: Optional[ort.InferenceSession] = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        
        self._lock = asyncio.Lock()
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Silero VAD ONNX model."""
        try:
            model_path = str(config.vad_model_path)
            self.session = ort.InferenceSession(model_path)
            print(f"✓ Loaded VAD model from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load VAD model: {e}")
            print(f"  Model should be at: {config.vad_model_path}")
            self.session = None
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for VAD model.
        
        Args:
            audio: Audio samples
        
        Returns:
            Preprocessed audio
        """
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        
        return audio
    
    async def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Process audio chunk and return speech probability.
        
        Args:
            audio_chunk: Audio samples (30ms chunk)
        
        Returns:
            Speech probability (0-1)
        """
        if self.session is None:
            # No model loaded, return default
            return 0.0
        
        # Preprocess
        audio = self.preprocess_audio(audio_chunk)
        
        # Ensure correct shape for model
        audio = audio.reshape(1, -1)
        
        # Run inference
        try:
            ort_inputs = {
                'input': audio,
                'h': self._h,
                'c': self._c,
                'sr': np.array([self.sample_rate], dtype=np.int64)
            }
            
            ort_outputs = self.session.run(None, ort_inputs)
            probability = ort_outputs[0][0][0]
            
            # Update hidden states
            self._h = ort_outputs[1]
            self._c = ort_outputs[2]
            
        except Exception as e:
            print(f"VAD inference error: {e}")
            probability = 0.0
        
        # Update state based on probability
        await self.update_state(probability)
        
        return float(probability)
    
    async def update_state(self, probability: float) -> None:
        """
        Update VAD state based on speech probability.
        
        Args:
            probability: Speech probability from model
        """
        async with self._lock:
            is_speech = probability > self.threshold
            
            if is_speech:
                self.consecutive_speech_chunks += 1
                self.consecutive_silence_chunks = 0
            else:
                self.consecutive_silence_chunks += 1
                self.consecutive_speech_chunks = 0
            
            # State transitions with hysteresis
            if self.current_state == VADState.NOT_SPEAKING:
                if self.consecutive_speech_chunks >= self.speech_start_threshold:
                    # Transition to SPEAKING
                    self.current_state = VADState.SPEAKING
                    self.speech_start_time = datetime.now()
                    self.silence_start_time = None
                    
                    await event_bus.emit(EventType.SPEECH_STARTED, {
                        "timestamp": self.speech_start_time,
                        "probability": probability
                    })
            
            elif self.current_state == VADState.SPEAKING:
                if is_speech:
                    # Continue speaking
                    await event_bus.emit(EventType.SPEECH_CONTINUING, {
                        "duration_ms": self.get_speech_duration(),
                        "probability": probability
                    })
                else:
                    if self.consecutive_silence_chunks >= self.speech_end_threshold:
                        # Transition to SILENCE_AFTER_SPEECH
                        self.current_state = VADState.SILENCE_AFTER_SPEECH
                        self.silence_start_time = datetime.now()
                        
                        await event_bus.emit(EventType.SILENCE_DETECTED, {
                            "timestamp": self.silence_start_time,
                            "speech_duration_ms": self.get_speech_duration(),
                            "probability": probability
                        })
            
            elif self.current_state == VADState.SILENCE_AFTER_SPEECH:
                if self.consecutive_speech_chunks >= self.speech_start_threshold:
                    # User resumed speaking
                    self.current_state = VADState.SPEAKING
                    self.speech_start_time = datetime.now()
                    self.silence_start_time = None
                    
                    await event_bus.emit(EventType.SPEECH_STARTED, {
                        "timestamp": self.speech_start_time,
                        "probability": probability,
                        "resumed": True
                    })
                else:
                    # Check if silence is long enough to end speech
                    silence_duration = self.get_silence_duration()
                    if silence_duration >= self.min_silence_duration_ms:
                        # Emit ongoing silence (for turn detection)
                        await event_bus.emit(EventType.SILENCE_DETECTED, {
                            "timestamp": self.silence_start_time,
                            "silence_duration_ms": silence_duration,
                            "probability": probability
                        })
    
    def get_speech_duration(self) -> float:
        """Get duration of current speech in milliseconds."""
        if self.speech_start_time:
            return (datetime.now() - self.speech_start_time).total_seconds() * 1000
        return 0.0
    
    def get_silence_duration(self) -> float:
        """Get duration of current silence in milliseconds."""
        if self.silence_start_time:
            return (datetime.now() - self.silence_start_time).total_seconds() * 1000
        return 0.0
    
    def reset(self) -> None:
        """Reset VAD state."""
        self.current_state = VADState.NOT_SPEAKING
        self.speech_start_time = None
        self.silence_start_time = None
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        
        # Reset LSTM states
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
