"""Backchannel timing and abort control."""
import asyncio
from typing import Optional

from .config import config
from .event_bus import event_bus, EventType, Event
from .conversation_manager import ConversationState


class BackchannelTimingController:
    """
    Controls backchannel timing and abort logic.
    
    Features:
    - 300ms safe zone timer
    - Monitor VAD for user speech resumption
    - Abort if user resumes speaking
    - Proceed to playback if safe
    """
    
    def __init__(self):
        """Initialize backchannel timing controller."""
        self.safe_zone_ms = config.backchannel_safe_zone_ms
        
        # State
        self.pending_backchannel: Optional[str] = None
        self.safe_zone_task: Optional[asyncio.Task] = None
        self.is_waiting = False
        
        # Subscribe to events
        event_bus.subscribe(EventType.BACKCHANNEL_TRIGGERED, self.on_backchannel_triggered)
        event_bus.subscribe(EventType.SPEECH_STARTED, self.on_speech_started)
    
    async def on_backchannel_triggered(self, event: Event) -> None:
        """
        Handle backchannel trigger event.
        
        Args:
            event: BACKCHANNEL_TRIGGERED event with backchannel_type
        """
        backchannel_type = event.data.get("backchannel_type")
        
        if not backchannel_type:
            return
        
        # Start safe zone timer
        await self.start_safe_zone_timer(backchannel_type)
    
    async def on_speech_started(self, event: Event) -> None:
        """
        Handle speech started event (user resumed speaking).
        
        Args:
            event: SPEECH_STARTED event
        """
        # If we're waiting, abort backchannel
        if self.is_waiting and self.pending_backchannel:
            await self.abort_backchannel()
    
    async def start_safe_zone_timer(self, backchannel_type: str) -> None:
        """
        Start safe zone timer.
        
        Args:
            backchannel_type: Type of backchannel to play
        """
        self.pending_backchannel = backchannel_type
        self.is_waiting = True
        
        # Create timer task
        self.safe_zone_task = asyncio.create_task(
            self._safe_zone_countdown()
        )
    
    async def _safe_zone_countdown(self) -> None:
        """Safe zone countdown (300ms)."""
        try:
            # Wait for safe zone duration
            await asyncio.sleep(self.safe_zone_ms / 1000)
            
            # If we get here, safe zone completed without interruption
            if self.is_waiting and self.pending_backchannel:
                await self.proceed_to_playback()
        
        except asyncio.CancelledError:
            # Timer was cancelled (user resumed speaking)
            pass
    
    async def abort_backchannel(self) -> None:
        """Abort pending backchannel."""
        backchannel_type = self.pending_backchannel
        
        # Cancel timer
        if self.safe_zone_task:
            self.safe_zone_task.cancel()
            try:
                await self.safe_zone_task
            except asyncio.CancelledError:
                pass
        
        # Clear state
        self.pending_backchannel = None
        self.is_waiting = False
        self.safe_zone_task = None
        
        # Emit abort event
        await event_bus.emit(EventType.BACKCHANNEL_ABORTED, {
            "backchannel_type": backchannel_type,
            "reason": "user_resumed_speaking"
        })
    
    async def proceed_to_playback(self) -> None:
        """Proceed to backchannel playback."""
        backchannel_type = self.pending_backchannel
        
        # Clear state
        self.pending_backchannel = None
        self.is_waiting = False
        self.safe_zone_task = None
        
        # Emit playback event (for player)
        # We reuse BACKCHANNEL_TRIGGERED but with a "proceed" flag
        await event_bus.emit(EventType.BACKCHANNEL_TRIGGERED, {
            "backchannel_type": backchannel_type,
            "proceed_to_play": True
        })
