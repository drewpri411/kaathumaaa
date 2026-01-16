"""Configuration management for the voice agent system."""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Config(BaseSettings):
    """Central configuration for all voice agent components."""
    
    # API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Audio Settings
    sample_rate: int = Field(default=16000, env="SAMPLE_RATE")
    channels: int = Field(default=1, env="CHANNELS")
    chunk_duration_ms: int = Field(default=30, env="CHUNK_DURATION_MS")
    whisper_chunk_duration_s: float = Field(default=1.5, env="WHISPER_CHUNK_DURATION_S")
    whisper_overlap_s: float = Field(default=0.5, env="WHISPER_OVERLAP_S")
    
    # VAD Settings
    vad_threshold: float = Field(default=0.5, env="VAD_THRESHOLD")
    vad_min_speech_duration_ms: int = Field(default=250, env="VAD_MIN_SPEECH_DURATION_MS")
    vad_min_silence_duration_ms: int = Field(default=300, env="VAD_MIN_SILENCE_DURATION_MS")
    vad_speech_pad_ms: int = Field(default=30, env="VAD_SPEECH_PAD_MS")
    
    # Turn Detection Thresholds
    short_pause_ms: int = Field(default=400, env="SHORT_PAUSE_MS")
    medium_pause_ms: int = Field(default=1000, env="MEDIUM_PAUSE_MS")
    long_pause_ms: int = Field(default=1500, env="LONG_PAUSE_MS")
    turn_end_score_threshold: int = Field(default=65, env="TURN_END_SCORE_THRESHOLD")
    
    # Scoring Weights (must sum to 1.0)
    silence_weight: float = Field(default=0.4, env="SILENCE_WEIGHT")
    linguistic_weight: float = Field(default=0.35, env="LINGUISTIC_WEIGHT")
    context_weight: float = Field(default=0.25, env="CONTEXT_WEIGHT")
    
    # Backchannel Settings
    backchannel_base_probability: float = Field(default=0.4, env="BACKCHANNEL_BASE_PROBABILITY")
    backchannel_min_interval_s: float = Field(default=5.0, env="BACKCHANNEL_MIN_INTERVAL_S")
    backchannel_safe_zone_ms: int = Field(default=300, env="BACKCHANNEL_SAFE_ZONE_MS")
    backchannel_volume: float = Field(default=0.5, env="BACKCHANNEL_VOLUME")
    
    # Continuation Words
    continuation_words: List[str] = Field(
        default=["and", "so", "but", "um", "uh", "like", "or", "because", 
                 "then", "well", "actually", "basically", "you know"]
    )
    
    # Emotion Keywords (for backchannel triggering)
    emotion_keywords: List[str] = Field(
        default=["amazing", "terrible", "wonderful", "awful", "excited", 
                 "frustrated", "happy", "sad", "angry", "love", "hate"]
    )
    
    # Explicit Prompt Phrases
    explicit_prompts: List[str] = Field(
        default=["you know?", "right?", "don't you think?", "isn't it?", 
                 "you see?", "understand?", "make sense?"]
    )
    
    # LLM Settings
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_max_tokens: int = Field(default=150, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    # TTS Settings
    tts_voice: str = Field(default="alloy", env="TTS_VOICE")
    tts_model: str = Field(default="tts-1", env="TTS_MODEL")
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    backchannel_dir: Optional[Path] = Field(default=None)
    vad_model_path: Optional[Path] = Field(default=None)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }
    
    def model_post_init(self, __context) -> None:
        """Initialize paths after model validation."""
        # Set default paths relative to base_dir
        if self.backchannel_dir is None:
            self.backchannel_dir = self.base_dir / "backchannels"
        if self.vad_model_path is None:
            self.vad_model_path = self.base_dir / "models" / "silero_vad.onnx"
    
    @field_validator("silence_weight", "linguistic_weight", "context_weight")
    @classmethod
    def validate_weights(cls, v):
        """Ensure weights are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {v}")
        return v
    
    @field_validator("vad_threshold", "backchannel_base_probability", "backchannel_volume")
    @classmethod
    def validate_probability(cls, v):
        """Ensure probability values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {v}")
        return v
    
    def validate_all(self) -> bool:
        """Validate all configuration values."""
        # Check weights sum to approximately 1.0
        weight_sum = self.silence_weight + self.linguistic_weight + self.context_weight
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        # Check API keys are set (warning only)
        if not self.openai_api_key:
            print("WARNING: OPENAI_API_KEY not set")
        
        # Check paths exist
        if not self.vad_model_path.exists():
            print(f"WARNING: VAD model not found at {self.vad_model_path}")
        
        if not self.backchannel_dir.exists():
            print(f"WARNING: Backchannel directory not found at {self.backchannel_dir}")
        
        return True
    
    def get_chunk_size_samples(self) -> int:
        """Get the number of samples per VAD chunk."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)
    
    def get_whisper_chunk_size_samples(self) -> int:
        """Get the number of samples per Whisper chunk."""
        return int(self.sample_rate * self.whisper_chunk_duration_s)
    
    def get_whisper_overlap_samples(self) -> int:
        """Get the number of samples for Whisper overlap."""
        return int(self.sample_rate * self.whisper_overlap_s)


# Global config instance
config = Config()
