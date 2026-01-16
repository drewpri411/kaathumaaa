"""Conversation state management - single source of truth."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
import asyncio


class ConversationState(Enum):
    """Possible conversation states."""
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    EVALUATING_PAUSE = "evaluating_pause"
    AGENT_THINKING = "agent_thinking"
    AGENT_SPEAKING = "agent_speaking"


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    text: str
    timestamp: datetime
    is_final: bool
    speaker: str  # "user" or "agent"
    
    def __repr__(self) -> str:
        final_marker = "[FINAL]" if self.is_final else "[PARTIAL]"
        return f"{self.speaker.upper()} {final_marker}: {self.text}"


@dataclass
class BackchannelEvent:
    """Record of a backchannel event."""
    type: str  # backchannel name (e.g., "mmhmm")
    timestamp: datetime
    was_successful: bool  # Did user continue speaking after?
    
    def __repr__(self) -> str:
        status = "✓" if self.was_successful else "✗"
        return f"{status} {self.type} @ {self.timestamp.isoformat()}"


@dataclass
class ConversationContext:
    """Complete conversation state."""
    # Current state
    state: ConversationState = ConversationState.IDLE
    
    # Timing
    current_user_speech_start: Optional[datetime] = None
    current_silence_start: Optional[datetime] = None
    current_silence_duration: float = 0.0  # seconds
    
    # Transcription
    partial_transcript: str = ""
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    
    # Backchannel tracking
    last_backchannel_time: Optional[datetime] = None
    backchannel_history: List[BackchannelEvent] = field(default_factory=list)
    
    # User behavior learning
    user_avg_pause_duration: float = 0.8  # seconds, learned over time
    user_sentence_count_current_turn: int = 0
    user_word_count_current_turn: int = 0
    
    # Current agent response
    current_agent_response: str = ""
    
    def get_user_speaking_duration(self) -> float:
        """Get duration of current user speech in seconds."""
        if self.current_user_speech_start:
            return (datetime.now() - self.current_user_speech_start).total_seconds()
        return 0.0
    
    def get_silence_duration(self) -> float:
        """Get duration of current silence in seconds."""
        if self.current_silence_start:
            return (datetime.now() - self.current_silence_start).total_seconds()
        return 0.0
    
    def get_time_since_last_backchannel(self) -> float:
        """Get seconds since last backchannel."""
        if self.last_backchannel_time:
            return (datetime.now() - self.last_backchannel_time).total_seconds()
        return float('inf')


class ConversationManager:
    """
    Manages conversation state and history.
    Single source of truth for all conversation data.
    """
    
    def __init__(self, event_bus):
        """
        Initialize conversation manager.
        
        Args:
            event_bus: EventBus instance for emitting state changes
        """
        self.context = ConversationContext()
        self.event_bus = event_bus
        self._lock = asyncio.Lock()
    
    async def update_state(self, new_state: ConversationState) -> None:
        """
        Update conversation state and emit event.
        
        Args:
            new_state: New conversation state
        """
        async with self._lock:
            old_state = self.context.state
            self.context.state = new_state
            
            # Emit state change event
            from .event_bus import EventType
            await self.event_bus.emit(EventType.STATE_CHANGED, {
                "old_state": old_state.value,
                "new_state": new_state.value,
                "timestamp": datetime.now()
            })
    
    def get_state(self) -> ConversationState:
        """Get current conversation state."""
        return self.context.state
    
    async def add_transcript(self, text: str, is_final: bool, speaker: str = "user") -> None:
        """
        Add a transcript segment.
        
        Args:
            text: Transcribed text
            is_final: Whether this is a final transcription
            speaker: "user" or "agent"
        """
        async with self._lock:
            segment = TranscriptSegment(
                text=text,
                timestamp=datetime.now(),
                is_final=is_final,
                speaker=speaker
            )
            self.context.transcript_segments.append(segment)
            
            # Update partial transcript for user
            if speaker == "user":
                if is_final:
                    self.context.partial_transcript = ""
                    # Update word and sentence counts
                    words = text.split()
                    self.context.user_word_count_current_turn += len(words)
                    # Simple sentence count (periods, question marks, exclamation marks)
                    sentences = text.count('.') + text.count('?') + text.count('!')
                    self.context.user_sentence_count_current_turn += max(1, sentences)
                else:
                    self.context.partial_transcript = text
    
    async def record_backchannel(self, backchannel_type: str, was_successful: bool) -> None:
        """
        Record a backchannel event.
        
        Args:
            backchannel_type: Type of backchannel (e.g., "mmhmm")
            was_successful: Whether user continued speaking after
        """
        async with self._lock:
            event = BackchannelEvent(
                type=backchannel_type,
                timestamp=datetime.now(),
                was_successful=was_successful
            )
            self.context.backchannel_history.append(event)
            self.context.last_backchannel_time = event.timestamp
    
    async def reset_turn(self) -> None:
        """Reset turn-specific state."""
        async with self._lock:
            self.context.current_user_speech_start = None
            self.context.current_silence_start = None
            self.context.current_silence_duration = 0.0
            self.context.partial_transcript = ""
            self.context.user_sentence_count_current_turn = 0
            self.context.user_word_count_current_turn = 0
    
    def get_recent_transcript(self, n: int = 5) -> List[TranscriptSegment]:
        """
        Get last N transcript segments.
        
        Args:
            n: Number of segments to retrieve
        
        Returns:
            List of recent transcript segments
        """
        return self.context.transcript_segments[-n:]
    
    def get_full_conversation(self) -> str:
        """
        Get complete conversation history as formatted string.
        
        Returns:
            Formatted conversation string
        """
        lines = []
        for segment in self.context.transcript_segments:
            if segment.is_final:
                prefix = "User:" if segment.speaker == "user" else "Agent:"
                lines.append(f"{prefix} {segment.text}")
        return "\n".join(lines)
    
    def get_user_transcript_current_turn(self) -> str:
        """Get all user text from current turn."""
        # Get all final user segments since last agent response
        user_segments = []
        for segment in reversed(self.context.transcript_segments):
            if segment.speaker == "agent":
                break
            if segment.speaker == "user" and segment.is_final:
                user_segments.insert(0, segment.text)
        
        return " ".join(user_segments)
    
    async def start_user_speech(self) -> None:
        """Mark start of user speech."""
        async with self._lock:
            self.context.current_user_speech_start = datetime.now()
            self.context.current_silence_start = None
    
    async def start_silence(self) -> None:
        """Mark start of silence."""
        async with self._lock:
            self.context.current_silence_start = datetime.now()
    
    async def update_silence_duration(self, duration: float) -> None:
        """Update current silence duration."""
        async with self._lock:
            self.context.current_silence_duration = duration
    
    def get_context(self) -> ConversationContext:
        """Get full conversation context (read-only)."""
        return self.context
