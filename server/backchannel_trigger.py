"""Backchannel trigger detection."""
import random
from typing import Optional

from .config import config
from .event_bus import event_bus, EventType, Event
from .conversation_manager import ConversationManager, ConversationState


class BackchannelTriggerDetector:
    """
    Detects when to trigger backchannels.
    
    Conditions:
    - User is speaking
    - Short pause (300-700ms)
    - At least 5s since last backchannel
    - Complete clause/sentence boundary
    - User has spoken at least 2 sentences
    
    Probability-based triggering with modifiers.
    """
    
    def __init__(self, conversation_manager: ConversationManager):
        """
        Initialize backchannel trigger detector.
        
        Args:
            conversation_manager: ConversationManager instance
        """
        self.conversation_manager = conversation_manager
        
        # Configuration
        self.base_probability = config.backchannel_base_probability
        self.min_interval_s = config.backchannel_min_interval_s
        self.short_pause_ms = config.short_pause_ms
        
        # Keywords
        self.emotion_keywords = set(word.lower() for word in config.emotion_keywords)
        self.explicit_prompts = [phrase.lower() for phrase in config.explicit_prompts]
        
        # Subscribe to events
        event_bus.subscribe(EventType.SILENCE_DETECTED, self.on_silence_detected)
        event_bus.subscribe(EventType.PARTIAL_TRANSCRIPT, self.on_partial_transcript)
    
    async def on_silence_detected(self, event: Event) -> None:
        """
        Handle silence detection.
        
        Args:
            event: SILENCE_DETECTED event
        """
        # Check trigger conditions
        if not await self.check_trigger_conditions(event):
            return
        
        # Calculate probability
        probability = self.calculate_probability()
        
        # Make decision
        if random.random() < probability:
            # Trigger backchannel
            await event_bus.emit(EventType.BACKCHANNEL_TRIGGERED, {
                "trigger_strength": probability,
                "silence_duration_ms": event.data.get("silence_duration_ms", 0)
            })
    
    async def on_partial_transcript(self, event: Event) -> None:
        """
        Handle partial transcript updates.
        
        Args:
            event: PARTIAL_TRANSCRIPT event
        """
        # Could use this for more sophisticated triggering
        # For now, we rely on silence detection
        pass
    
    async def check_trigger_conditions(self, event: Event) -> bool:
        """
        Check if all trigger conditions are met.
        
        Args:
            event: SILENCE_DETECTED event
        
        Returns:
            True if conditions met
        """
        # 1. Current state must be USER_SPEAKING
        state = self.conversation_manager.get_state()
        if state != ConversationState.USER_SPEAKING:
            return False
        
        # 2. Silence duration in short pause range (300-700ms)
        silence_duration_ms = event.data.get("silence_duration_ms", 0)
        if not (300 <= silence_duration_ms <= 700):
            return False
        
        # 3. At least 5 seconds since last backchannel
        context = self.conversation_manager.get_context()
        time_since_last = context.get_time_since_last_backchannel()
        if time_since_last < self.min_interval_s:
            return False
        
        # 4. User has spoken at least 2 sentences
        if context.user_sentence_count_current_turn < 2:
            return False
        
        # 5. Has some transcript
        transcript = self.conversation_manager.get_user_transcript_current_turn()
        if not transcript or len(transcript.split()) < 5:
            return False
        
        return True
    
    def calculate_probability(self) -> float:
        """
        Calculate backchannel probability with modifiers.
        
        Returns:
            Probability (0-1)
        """
        prob = self.base_probability
        
        context = self.conversation_manager.get_context()
        transcript = self.conversation_manager.get_user_transcript_current_turn().lower()
        
        # Modifier: emotion keywords
        if self.detect_emotion_keywords(transcript):
            prob += 0.3
        
        # Modifier: explicit prompts ("you know?", "right?")
        if self.detect_explicit_prompts(transcript):
            prob += 0.5
        
        # Modifier: just played backchannel recently
        time_since_last = context.get_time_since_last_backchannel()
        if time_since_last < 8:  # within 8 seconds
            prob -= 0.2
        
        # Modifier: user just started speaking
        speaking_duration = context.get_user_speaking_duration()
        if speaking_duration < 3:
            prob -= 0.3
        
        # Modifier: pause after complete sentence (ends with punctuation)
        if transcript.rstrip().endswith(('.', '!', '?')):
            prob += 0.2
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, prob))
    
    def detect_emotion_keywords(self, text: str) -> bool:
        """
        Detect emotion keywords in text.
        
        Args:
            text: Text to analyze
        
        Returns:
            True if emotion keywords found
        """
        words = set(text.lower().split())
        return bool(words & self.emotion_keywords)
    
    def detect_explicit_prompts(self, text: str) -> bool:
        """
        Detect explicit prompts in text.
        
        Args:
            text: Text to analyze
        
        Returns:
            True if explicit prompts found
        """
        text_lower = text.lower()
        return any(prompt in text_lower for prompt in self.explicit_prompts)
