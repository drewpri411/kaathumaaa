"""Response coordination for LLM -> TTS flow."""
import asyncio
from typing import Optional

from .config import config
from .event_bus import event_bus, EventType, Event
from .conversation_manager import ConversationManager, ConversationState
from .llm_client import LLMClient
from .tts_client import TTSClient


class ResponseCoordinator:
    """
    Coordinates response generation flow.
    
    Flow:
    1. Turn ends -> AGENT_THINKING
    2. Generate LLM response
    3. Synthesize to speech -> AGENT_SPEAKING
    4. Send audio to mixer
    5. Complete -> IDLE
    """
    
    def __init__(
        self,
        conversation_manager: ConversationManager,
        llm_client: LLMClient,
        tts_client: TTSClient,
        audio_mixer
    ):
        """
        Initialize response coordinator.
        
        Args:
            conversation_manager: ConversationManager instance
            llm_client: LLMClient instance
            tts_client: TTSClient instance
            audio_mixer: AudioMixer instance
        """
        self.conversation_manager = conversation_manager
        self.llm_client = llm_client
        self.tts_client = tts_client
        self.audio_mixer = audio_mixer
        
        # Subscribe to turn end events
        event_bus.subscribe(EventType.TURN_ENDED, self.on_turn_ended)
    
    async def on_turn_ended(self, event: Event) -> None:
        """
        Handle turn ended event.
        
        Args:
            event: TURN_ENDED event
        """
        # Get user utterance
        user_utterance = event.data.get("transcript", "")
        
        if not user_utterance:
            print("WARNING: Turn ended with empty transcript")
            return
        
        # Generate and play response
        await self.generate_and_play_response(user_utterance)
    
    async def generate_and_play_response(self, user_utterance: str) -> None:
        """
        Generate LLM response and synthesize to speech.
        
        Args:
            user_utterance: User's utterance
        """
        # Update state to AGENT_THINKING
        await self.conversation_manager.update_state(ConversationState.AGENT_THINKING)
        
        # Get conversation history
        conversation_history = self.conversation_manager.get_full_conversation()
        
        # Generate LLM response
        await event_bus.emit(EventType.RESPONSE_GENERATING, {
            "user_utterance": user_utterance
        })
        
        # Collect response chunks
        response_chunks = []
        async for chunk in self.llm_client.generate_response(
            conversation_history,
            user_utterance
        ):
            response_chunks.append(chunk)
        
        response_text = "".join(response_chunks)
        
        if not response_text:
            print("WARNING: Empty response from LLM")
            await self.conversation_manager.update_state(ConversationState.IDLE)
            return
        
        # Add to conversation history
        await self.conversation_manager.add_transcript(
            text=response_text,
            is_final=True,
            speaker="agent"
        )
        
        # Synthesize to speech
        audio = await self.tts_client.synthesize(response_text)
        
        if audio is None:
            print("WARNING: TTS synthesis failed")
            await self.conversation_manager.update_state(ConversationState.IDLE)
            return
        
        # Update state to AGENT_SPEAKING
        await self.conversation_manager.update_state(ConversationState.AGENT_SPEAKING)
        
        # Emit response started
        await event_bus.emit(EventType.RESPONSE_STARTED, {
            "text": response_text,
            "audio_duration_s": len(audio) / config.sample_rate
        })
        
        # Send to mixer on primary channel
        await self.audio_mixer.add_primary_audio(audio)
        
        # Wait for playback to complete
        # (In a real system, you'd track playback progress)
        playback_duration = len(audio) / config.sample_rate
        await asyncio.sleep(playback_duration)
        
        # Emit response ended
        await event_bus.emit(EventType.RESPONSE_ENDED, {
            "text": response_text
        })
        
        # Reset turn and return to IDLE
        await self.conversation_manager.reset_turn()
        await self.conversation_manager.update_state(ConversationState.IDLE)
