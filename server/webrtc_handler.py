"""WebRTC connection handler."""
import asyncio
from typing import Optional
import numpy as np

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from av import AudioFrame

from .config import config
from .audio_pipeline import AudioPipeline


class AudioStreamTrack(MediaStreamTrack):
    """
    Custom audio track for sending audio to client.
    """
    
    kind = "audio"
    
    def __init__(self, audio_mixer):
        super().__init__()
        self.audio_mixer = audio_mixer
        self.sample_rate = config.sample_rate
        
        # Frame parameters (20ms frames)
        self.samples_per_frame = int(self.sample_rate * 0.02)  # 20ms
    
    async def recv(self):
        """
        Receive next audio frame.
        
        Returns:
            AudioFrame to send to client
        """
        # Get audio from mixer
        audio = self.audio_mixer.get_output_audio(self.samples_per_frame)
        
        if audio is None:
            # No audio available, send silence
            audio = np.zeros(self.samples_per_frame, dtype=np.float32)
        
        # Convert to int16 for transmission
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create audio frame
        frame = AudioFrame.from_ndarray(
            audio_int16.reshape(1, -1),  # shape: (channels, samples)
            format='s16',
            layout='mono'
        )
        frame.sample_rate = self.sample_rate
        
        # Set timestamp
        pts = getattr(self, '_pts', 0)
        frame.pts = pts
        frame.time_base = f"1/{self.sample_rate}"
        
        self._pts = pts + self.samples_per_frame
        
        return frame


class WebRTCHandler:
    """
    Handles WebRTC connections.
    
    Features:
    - Peer connection management
    - Audio track handling (send/receive)
    - Integration with audio pipeline
    """
    
    def __init__(self, audio_pipeline: AudioPipeline, audio_mixer):
        """
        Initialize WebRTC handler.
        
        Args:
            audio_pipeline: AudioPipeline instance
            audio_mixer: AudioMixer instance
        """
        self.audio_pipeline = audio_pipeline
        self.audio_mixer = audio_mixer
        
        # Peer connections
        self.pcs = set()
        
        # Media relay
        self.relay = MediaRelay()
    
    async def create_peer_connection(self) -> RTCPeerConnection:
        """
        Create new peer connection.
        
        Returns:
            RTCPeerConnection instance
        """
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await self.close_peer_connection(pc)
        
        @pc.on("track")
        async def on_track(track):
            print(f"Track received: {track.kind}")
            
            if track.kind == "audio":
                # Receive audio from client
                asyncio.create_task(self._receive_audio(track))
        
        return pc
    
    async def _receive_audio(self, track):
        """
        Receive audio from client track.
        
        Args:
            track: Audio track from client
        """
        try:
            while True:
                frame = await track.recv()
                
                # Convert frame to numpy array
                audio = frame.to_ndarray()
                
                # Convert to float32 [-1, 1]
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                
                # Flatten if multi-channel
                if len(audio.shape) > 1:
                    audio = audio.flatten()
                
                # Send to audio pipeline
                await self.audio_pipeline.receive_audio(audio)
        
        except Exception as e:
            print(f"Error receiving audio: {e}")
    
    async def handle_offer(self, pc: RTCPeerConnection, offer: dict) -> dict:
        """
        Handle WebRTC offer from client.
        
        Args:
            pc: Peer connection
            offer: Offer SDP
        
        Returns:
            Answer SDP
        """
        # Set remote description
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        )
        
        # Add audio track for sending audio to client
        audio_track = AudioStreamTrack(self.audio_mixer)
        pc.addTrack(audio_track)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    
    async def close_peer_connection(self, pc: RTCPeerConnection) -> None:
        """
        Close peer connection.
        
        Args:
            pc: Peer connection to close
        """
        await pc.close()
        self.pcs.discard(pc)
    
    async def close_all(self) -> None:
        """Close all peer connections."""
        coros = [self.close_peer_connection(pc) for pc in list(self.pcs)]
        await asyncio.gather(*coros)
