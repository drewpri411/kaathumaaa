"""Backchannel selection logic."""
import random
from collections import deque
from typing import List, Dict

from .event_bus import event_bus, EventType, Event
from .conversation_manager import ConversationManager


class BackchannelSelector:
    """
    Selects appropriate backchannel based on context.
    
    Features:
    - Context-aware selection
    - Anti-repetition logic
    - Weighted randomization
    """
    
    def __init__(self, conversation_manager: ConversationManager):
        """
        Initialize backchannel selector.
        
        Args:
            conversation_manager: ConversationManager instance
        """
        self.conversation_manager = conversation_manager
        
        # Recent backchannels (for anti-repetition)
        self.recent_backchannels = deque(maxlen=3)
        
        # Usage count (for balancing)
        self.usage_count: Dict[str, int] = {
            "mmhmm": 0,
            "okay": 0,
            "yeah": 0,
            "i_see": 0,
            "right": 0
        }
        
        # Subscribe to trigger events
        event_bus.subscribe(EventType.BACKCHANNEL_TRIGGERED, self.on_backchannel_triggered)
    
    async def on_backchannel_triggered(self, event: Event) -> None:
        """
        Handle backchannel trigger event.
        
        Args:
            event: BACKCHANNEL_TRIGGERED event
        """
        # Get context
        transcript = self.conversation_manager.get_user_transcript_current_turn()
        
        # Select backchannel
        selected = self.select_backchannel(transcript)
        
        # Record usage
        self.record_usage(selected)
        
        # Emit selection event (for timing controller)
        await event_bus.emit(EventType.BACKCHANNEL_TRIGGERED, {
            "backchannel_type": selected,
            "trigger_strength": event.data.get("trigger_strength", 0.4),
            "silence_duration_ms": event.data.get("silence_duration_ms", 0)
        })
    
    def select_backchannel(self, transcript: str) -> str:
        """
        Select appropriate backchannel.
        
        Args:
            transcript: Current user transcript
        
        Returns:
            Selected backchannel name
        """
        # Get candidates based on context
        candidates = self.get_candidates(transcript)
        
        # Apply anti-repetition
        candidates = self.apply_anti_repetition(candidates)
        
        # If no candidates left, use all
        if not candidates:
            candidates = list(self.usage_count.keys())
        
        # Select randomly from candidates
        # Weight by inverse usage count (prefer less used)
        weights = [1.0 / (self.usage_count[c] + 1) for c in candidates]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        selected = random.choices(candidates, weights=weights, k=1)[0]
        
        return selected
    
    def get_candidates(self, transcript: str) -> List[str]:
        """
        Get candidate backchannels based on context.
        
        Args:
            transcript: Current user transcript
        
        Returns:
            List of appropriate backchannel names
        """
        transcript_lower = transcript.lower()
        
        # Question detection
        if transcript.rstrip().endswith('?') or any(
            transcript_lower.startswith(q) for q in ['what', 'when', 'where', 'who', 'why', 'how']
        ):
            # Questions -> "right" or "I see"
            return ["right", "i_see"]
        
        # Emotional/excited tone detection
        emotion_words = ["amazing", "terrible", "wonderful", "awful", "excited", "love", "hate"]
        if any(word in transcript_lower for word in emotion_words):
            # Emotional -> "yeah"
            return ["yeah", "right"]
        
        # Neutral continuation -> "mm-hmm" or "okay"
        return ["mmhmm", "okay", "i_see"]
    
    def apply_anti_repetition(self, candidates: List[str]) -> List[str]:
        """
        Remove recently used backchannels.
        
        Args:
            candidates: Candidate backchannels
        
        Returns:
            Filtered candidates
        """
        # Remove last used
        if self.recent_backchannels:
            last_used = self.recent_backchannels[-1]
            candidates = [c for c in candidates if c != last_used]
        
        # If same type used twice in a row, exclude it
        if len(self.recent_backchannels) >= 2:
            if self.recent_backchannels[-1] == self.recent_backchannels[-2]:
                candidates = [c for c in candidates if c != self.recent_backchannels[-1]]
        
        return candidates
    
    def record_usage(self, backchannel_type: str) -> None:
        """
        Record backchannel usage.
        
        Args:
            backchannel_type: Type of backchannel used
        """
        self.recent_backchannels.append(backchannel_type)
        self.usage_count[backchannel_type] = self.usage_count.get(backchannel_type, 0) + 1
