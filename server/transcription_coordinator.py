"""Transcription coordinator for chunked STT."""
import asyncio
from typing import Dict, List, Optional
import uuid
from datetime import datetime

from .config import config
from .event_bus import event_bus, EventType
from .stt_client import STTClient
from .audio_pipeline import AudioPipeline


class TranscriptionCoordinator:
    """
    Coordinates chunked transcription with deduplication.
    
    Features:
    - Collect 1.5s audio chunks with 0.5s overlap
    - Send to STT client
    - Deduplicate overlapping results
    - Build continuous transcript
    """
    
    def __init__(self, audio_pipeline: AudioPipeline, stt_client: STTClient, conversation_manager):
        """
        Initialize transcription coordinator.
        
        Args:
            audio_pipeline: AudioPipeline instance
            stt_client: STTClient instance
            conversation_manager: ConversationManager instance
        """
        self.audio_pipeline = audio_pipeline
        self.stt_client = stt_client
        self.conversation_manager = conversation_manager
        
        # State
        self.in_flight_requests: Dict[str, asyncio.Task] = {}
        self.recent_transcripts: List[str] = []
        self.full_transcript = ""
        
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start transcription coordination loop."""
        self.running = True
        self.task = asyncio.create_task(self._coordination_loop())
        print("✓ Transcription coordinator started")
    
    async def stop(self) -> None:
        """Stop transcription coordination."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        print("✓ Transcription coordinator stopped")
    
    async def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.running:
            try:
                # Get chunks from audio pipeline
                for chunk in self.audio_pipeline.get_whisper_chunks():
                    # Send for transcription
                    chunk_id = str(uuid.uuid4())
                    task = asyncio.create_task(
                        self._transcribe_and_handle(chunk, chunk_id)
                    )
                    self.in_flight_requests[chunk_id] = task
                
                # Clean up completed tasks
                completed_ids = [
                    cid for cid, task in self.in_flight_requests.items()
                    if task.done()
                ]
                for cid in completed_ids:
                    del self.in_flight_requests[cid]
                
                # Wait a bit before next iteration
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Transcription coordination error: {e}")
                await asyncio.sleep(1.0)
    
    async def _transcribe_and_handle(self, audio_chunk, chunk_id: str) -> None:
        """
        Transcribe chunk and handle result.
        
        Args:
            audio_chunk: Audio data
            chunk_id: Unique chunk identifier
        """
        # Transcribe
        text = await self.stt_client.transcribe_chunk(audio_chunk, is_final=False)
        
        if text:
            # Deduplicate
            new_text = self.deduplicate(text)
            
            if new_text:
                # Update conversation manager
                await self.conversation_manager.add_transcript(
                    text=new_text,
                    is_final=True,
                    speaker="user"
                )
                
                # Update full transcript
                self.full_transcript += " " + new_text
    
    def deduplicate(self, new_text: str) -> str:
        """
        Deduplicate new transcription against recent transcripts.
        
        Args:
            new_text: New transcription text
        
        Returns:
            Deduplicated text (only new words)
        """
        if not self.recent_transcripts:
            self.recent_transcripts.append(new_text)
            return new_text
        
        # Compare with last 2 transcripts
        new_words = new_text.lower().split()
        
        for recent in self.recent_transcripts[-2:]:
            recent_words = recent.lower().split()
            
            # Calculate overlap
            overlap_count = 0
            for i in range(min(len(new_words), len(recent_words))):
                if new_words[i] == recent_words[i]:
                    overlap_count += 1
                else:
                    break
            
            # If >80% overlap, extract only new words
            if overlap_count / len(new_words) > 0.8:
                new_words = new_words[overlap_count:]
                break
        
        # Update recent transcripts
        self.recent_transcripts.append(new_text)
        if len(self.recent_transcripts) > 3:
            self.recent_transcripts.pop(0)
        
        # Return deduplicated text
        deduplicated = " ".join(new_words)
        return deduplicated if deduplicated else ""
    
    def reset(self) -> None:
        """Reset transcription state."""
        self.recent_transcripts.clear()
        self.full_transcript = ""
        
        # Cancel in-flight requests
        for task in self.in_flight_requests.values():
            task.cancel()
        self.in_flight_requests.clear()
