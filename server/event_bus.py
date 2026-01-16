"""Event bus system for component communication."""
import asyncio
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import inspect


class EventType(Enum):
    """All event types in the system."""
    # Audio events
    AUDIO_CHUNK_RECEIVED = "audio_chunk_received"
    
    # VAD events
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUING = "speech_continuing"
    SILENCE_DETECTED = "silence_detected"
    SPEECH_ENDED = "speech_ended"
    
    # Transcription events
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    
    # Turn detection events
    TURN_EVALUATION = "turn_evaluation"
    TURN_ENDED = "turn_ended"
    
    # Backchannel events
    BACKCHANNEL_TRIGGERED = "backchannel_triggered"
    BACKCHANNEL_PLAYED = "backchannel_played"
    BACKCHANNEL_ABORTED = "backchannel_aborted"
    
    # Response events
    RESPONSE_GENERATING = "response_generating"
    RESPONSE_STARTED = "response_started"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE_ENDED = "response_ended"
    
    # State events
    STATE_CHANGED = "state_changed"


@dataclass
class Event:
    """Event data structure."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self) -> str:
        return f"Event({self.event_type.value}, {self.timestamp.isoformat()})"


class EventBus:
    """
    Central event bus for pub/sub communication between components.
    
    Features:
    - Async event emission
    - Support for both sync and async callbacks
    - Event history for debugging
    - Thread-safe operations
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize the event bus.
        
        Args:
            history_size: Number of events to keep in history
        """
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._history: deque = deque(maxlen=history_size)
        self._lock = asyncio.Lock()
    
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is emitted (can be sync or async)
        """
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
    
    async def emit(self, event_type: EventType, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event_type: Type of event to emit
            data: Event data dictionary
        """
        if data is None:
            data = {}
        
        event = Event(event_type=event_type, data=data)
        
        # Add to history
        async with self._lock:
            self._history.append(event)
        
        # Call all subscribers
        subscribers = self._subscribers[event_type].copy()
        
        for callback in subscribers:
            try:
                # Check if callback is async
                if inspect.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    # Run sync callback in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, callback, event)
            except Exception as e:
                print(f"Error in event callback for {event_type.value}: {e}")
    
    def emit_sync(self, event_type: EventType, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Synchronous emit (creates task in event loop).
        
        Args:
            event_type: Type of event to emit
            data: Event data dictionary
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.emit(event_type, data))
            else:
                loop.run_until_complete(self.emit(event_type, data))
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self.emit(event_type, data))
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 10) -> List[Event]:
        """
        Get recent events from history.
        
        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events to return
        
        Returns:
            List of recent events
        """
        history = list(self._history)
        
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        
        return history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers[event_type])
    
    def get_all_subscribers(self) -> Dict[EventType, int]:
        """Get subscriber counts for all event types."""
        return {et: len(subs) for et, subs in self._subscribers.items()}


# Global event bus instance
event_bus = EventBus()
