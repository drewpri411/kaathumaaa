"""Backchannel playback coordination."""
from typing import Optional

from .config import config
from .event_bus import event_bus, EventType, Event
from .backchannel_library import BackchannelLibrary


class BackchannelPlayer:
    """
    Coordinates backchannel playback.
    
    Features:
    - Get audio from library
    - Send to mixer on secondary channel
    - Track playback completion
    - Record success in conversation manager
    """
    
    def __init__(self, backchannel_library: BackchannelLibrary, audio_mixer, conversation_manager):
        """
        Initialize backchannel player.
        
        Args:
            backchannel_library: BackchannelLibrary instance
            audio_mixer: AudioMixer instance
            conversation_manager: ConversationManager instance
        """
        self.library = backchannel_library
        self.mixer = audio_mixer
        self.conversation_manager = conversation_manager
        
        # Subscribe to events
        event_bus.subscribe(EventType.BACKCHANNEL_TRIGGERED, self.on_backchannel_triggered)
    
    async def on_backchannel_triggered(self, event: Event) -> None:
        """
        Handle backchannel trigger event.
        
        Args:
            event: BACKCHANNEL_TRIGGERED event
        """
        # Only proceed if this is a playback event
        if not event.data.get("proceed_to_play"):
            return
        
        backchannel_type = event.data.get("backchannel_type")
        
        if backchannel_type:
            await self.play_backchannel(backchannel_type)
    
    async def play_backchannel(self, backchannel_type: str) -> None:
        """
        Play backchannel audio.
        
        Args:
            backchannel_type: Type of backchannel to play
        """
        # Get audio from library
        audio = self.library.get_backchannel(backchannel_type)
        
        if audio is None:
            print(f"WARNING: Backchannel '{backchannel_type}' not found in library")
            return
        
        # Send to mixer on secondary channel
        await self.mixer.add_secondary_audio(audio)
        
        # Record in conversation manager
        # Note: We assume success here. In a real system, you'd track
        # whether the user continued speaking after the backchannel.
        await self.conversation_manager.record_backchannel(
            backchannel_type=backchannel_type,
            was_successful=True  # Simplified for now
        )
        
        # Emit played event
        await event_bus.emit(EventType.BACKCHANNEL_PLAYED, {
            "backchannel_type": backchannel_type,
            "duration_s": len(audio) / config.sample_rate
        })
        
        # Note: We do NOT change conversation state
        # User should still be in USER_SPEAKING state
