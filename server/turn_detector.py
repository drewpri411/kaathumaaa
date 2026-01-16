"""Turn detection using multi-signal analysis."""
import asyncio
from typing import Optional
from dataclasses import dataclass

from .config import config
from .event_bus import event_bus, EventType, Event
from .linguistic_analyzer import LinguisticAnalyzer
from .conversation_manager import ConversationManager, ConversationState


@dataclass
class TurnScores:
    """Turn detection scores."""
    silence_score: int  # 0-100
    linguistic_score: int  # 0-100
    context_score: int  # 0-100
    final_score: float  # weighted combination
    silence_duration_ms: float
    transcript: str


class TurnDetector:
    """
    Multi-signal turn detection.
    
    Combines:
    - Silence duration (40% weight)
    - Linguistic completeness (35% weight)
    - Conversation context (25% weight)
    
    Emits TURN_ENDED when confidence > threshold.
    """
    
    def __init__(self, conversation_manager: ConversationManager):
        """
        Initialize turn detector.
        
        Args:
            conversation_manager: ConversationManager instance
        """
        self.conversation_manager = conversation_manager
        self.linguistic_analyzer = LinguisticAnalyzer()
        
        # Weights
        self.silence_weight = config.silence_weight
        self.linguistic_weight = config.linguistic_weight
        self.context_weight = config.context_weight
        
        # Thresholds
        self.short_pause_ms = config.short_pause_ms
        self.medium_pause_ms = config.medium_pause_ms
        self.long_pause_ms = config.long_pause_ms
        self.turn_end_threshold = config.turn_end_score_threshold
        
        # Subscribe to events
        event_bus.subscribe(EventType.SILENCE_DETECTED, self.on_silence_detected)
    
    async def on_silence_detected(self, event: Event) -> None:
        """
        Handle silence detection event.
        
        Args:
            event: SILENCE_DETECTED event
        """
        # Only evaluate if user is speaking
        state = self.conversation_manager.get_state()
        if state != ConversationState.USER_SPEAKING:
            return
        
        # Get silence duration
        silence_duration_ms = event.data.get("silence_duration_ms", 0)
        
        # Update conversation manager
        await self.conversation_manager.update_silence_duration(silence_duration_ms / 1000)
        
        # Calculate scores
        silence_score = self.calculate_silence_score(silence_duration_ms)
        linguistic_score = await self.calculate_linguistic_score()
        context_score = self.calculate_context_score()
        
        # Fuse scores
        final_score = (
            self.silence_weight * silence_score +
            self.linguistic_weight * linguistic_score +
            self.context_weight * context_score
        )
        
        # Get current transcript
        transcript = self.conversation_manager.get_user_transcript_current_turn()
        
        # Create scores object
        scores = TurnScores(
            silence_score=silence_score,
            linguistic_score=linguistic_score,
            context_score=context_score,
            final_score=final_score,
            silence_duration_ms=silence_duration_ms,
            transcript=transcript
        )
        
        # Emit evaluation event (for debugging)
        await event_bus.emit(EventType.TURN_EVALUATION, {
            "silence_score": silence_score,
            "linguistic_score": linguistic_score,
            "context_score": context_score,
            "final_score": final_score,
            "silence_duration_ms": silence_duration_ms,
            "transcript": transcript
        })
        
        # Make decision
        await self.make_turn_decision(scores)
    
    def calculate_silence_score(self, duration_ms: float) -> int:
        """
        Calculate silence score based on duration.
        
        Args:
            duration_ms: Silence duration in milliseconds
        
        Returns:
            Score from 0-100
        """
        if duration_ms < self.short_pause_ms:
            # Too short (< 400ms)
            return 10
        elif duration_ms < 700:
            # Short pause (400-700ms)
            return 20
        elif duration_ms < self.medium_pause_ms:
            # Medium pause (700-1000ms)
            return 50
        elif duration_ms < self.long_pause_ms:
            # Long pause (1000-1500ms)
            return 80
        else:
            # Very long pause (> 1500ms)
            return 100
    
    async def calculate_linguistic_score(self) -> int:
        """
        Calculate linguistic completeness score.
        
        Returns:
            Score from 0-100
        """
        # Get current transcript
        transcript = self.conversation_manager.get_user_transcript_current_turn()
        
        if not transcript:
            return 0
        
        # Analyze completeness
        analysis = self.linguistic_analyzer.analyze_completeness(transcript)
        
        return analysis.completeness_score
    
    def calculate_context_score(self) -> int:
        """
        Calculate context score based on conversation state.
        
        Returns:
            Score from 0-100
        """
        context = self.conversation_manager.get_context()
        
        score = 50  # baseline
        
        # User speaking duration
        speaking_duration = context.get_user_speaking_duration()
        
        if speaking_duration > 15:
            # User is on a roll, less likely to be done
            score += 20
        elif speaking_duration < 2:
            # Very short, might not be complete
            score -= 10
        
        # Word count
        if context.user_word_count_current_turn < 5:
            # Very short utterance
            score -= 20
        
        # Sentence count
        if context.user_sentence_count_current_turn >= 2:
            # Multiple sentences, more likely complete
            score += 10
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    async def make_turn_decision(self, scores: TurnScores) -> None:
        """
        Make turn ending decision based on scores.
        
        Args:
            scores: TurnScores object
        """
        if scores.final_score > self.turn_end_threshold:
            # Turn has ended
            await self.conversation_manager.update_state(ConversationState.AGENT_THINKING)
            
            await event_bus.emit(EventType.TURN_ENDED, {
                "final_score": scores.final_score,
                "silence_score": scores.silence_score,
                "linguistic_score": scores.linguistic_score,
                "context_score": scores.context_score,
                "transcript": scores.transcript,
                "silence_duration_ms": scores.silence_duration_ms
            })
        
        elif 40 < scores.final_score <= self.turn_end_threshold:
            # Uncertain, update state to evaluating
            await self.conversation_manager.update_state(ConversationState.EVALUATING_PAUSE)
        
        # else: score < 40, assume continuation (stay in USER_SPEAKING)
