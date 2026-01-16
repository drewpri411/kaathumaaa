"""FastAPI server main entry point."""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pathlib import Path

from .config import config
from .event_bus import event_bus
from .conversation_manager import ConversationManager
from .audio_pipeline import AudioPipeline
from .vad_processor import VADProcessor
from .stt_client import STTClient
from .transcription_coordinator import TranscriptionCoordinator
from .linguistic_analyzer import LinguisticAnalyzer
from .turn_detector import TurnDetector
from .backchannel_library import BackchannelLibrary
from .backchannel_trigger import BackchannelTriggerDetector
from .backchannel_selector import BackchannelSelector
from .backchannel_timing import BackchannelTimingController
from .backchannel_player import BackchannelPlayer
from .llm_client import LLMClient
from .tts_client import TTSClient
from .response_coordinator import ResponseCoordinator
from .audio_mixer import AudioMixer
from .webrtc_handler import WebRTCHandler


# Create FastAPI app
app = FastAPI(title="Voice Conversation Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
components = {}


@app.on_event("startup")
async def startup():
    """Initialize all components on startup."""
    print("=" * 60)
    print("Voice Conversation Agent - Starting Up")
    print("=" * 60)
    
    # Validate configuration
    try:
        config.validate_all()
    except Exception as e:
        print(f"Configuration error: {e}")
    
    # Initialize components
    print("\n📦 Initializing components...")
    
    # Core infrastructure
    components['conversation_manager'] = ConversationManager(event_bus)
    components['audio_pipeline'] = AudioPipeline()
    components['audio_mixer'] = AudioMixer()
    
    # VAD and STT
    components['vad_processor'] = VADProcessor()
    components['stt_client'] = STTClient()
    components['transcription_coordinator'] = TranscriptionCoordinator(
        components['audio_pipeline'],
        components['stt_client'],
        components['conversation_manager']
    )
    
    # Turn detection
    components['linguistic_analyzer'] = LinguisticAnalyzer()
    components['turn_detector'] = TurnDetector(components['conversation_manager'])
    
    # Backchannel system
    components['backchannel_library'] = BackchannelLibrary()
    components['backchannel_trigger'] = BackchannelTriggerDetector(components['conversation_manager'])
    components['backchannel_selector'] = BackchannelSelector(components['conversation_manager'])
    components['backchannel_timing'] = BackchannelTimingController()
    components['backchannel_player'] = BackchannelPlayer(
        components['backchannel_library'],
        components['audio_mixer'],
        components['conversation_manager']
    )
    
    # Response generation
    components['llm_client'] = LLMClient()
    components['tts_client'] = TTSClient()
    components['response_coordinator'] = ResponseCoordinator(
        components['conversation_manager'],
        components['llm_client'],
        components['tts_client'],
        components['audio_mixer']
    )
    
    # WebRTC
    components['webrtc_handler'] = WebRTCHandler(
        components['audio_pipeline'],
        components['audio_mixer']
    )
    
    # Start async components
    print("\n🚀 Starting async components...")
    await components['transcription_coordinator'].start()
    await components['audio_mixer'].start()
    
    # Start VAD processing loop
    asyncio.create_task(vad_processing_loop())
    
    print("\n✅ Server ready!")
    print(f"📡 Listening on http://{config.host}:{config.port}")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    print("\n🛑 Shutting down...")
    
    # Stop async components
    if 'transcription_coordinator' in components:
        await components['transcription_coordinator'].stop()
    
    if 'audio_mixer' in components:
        await components['audio_mixer'].stop()
    
    if 'webrtc_handler' in components:
        await components['webrtc_handler'].close_all()
    
    print("✅ Shutdown complete")


async def vad_processing_loop():
    """Process VAD chunks continuously."""
    vad = components['vad_processor']
    pipeline = components['audio_pipeline']
    
    while True:
        try:
            # Process VAD chunks
            for chunk in pipeline.get_vad_chunks():
                await vad.process_chunk(chunk)
            
            await asyncio.sleep(0.01)  # 10ms
        except Exception as e:
            print(f"VAD processing error: {e}")
            await asyncio.sleep(0.1)


# Routes

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main page."""
    static_dir = Path(__file__).parent.parent / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return index_file.read_text(encoding='utf-8')
    else:
        return "<h1>Voice Agent</h1><p>Static files not found</p>"


@app.post("/offer")
async def offer(request: Request):
    """Handle WebRTC offer."""
    params = await request.json()
    offer = params.get("offer")
    
    if not offer:
        return JSONResponse({"error": "No offer provided"}, status_code=400)
    
    # Create peer connection
    pc = await components['webrtc_handler'].create_peer_connection()
    
    # Handle offer
    answer = await components['webrtc_handler'].handle_offer(pc, offer)
    
    return JSONResponse({"answer": answer})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "components": {
            name: "initialized" for name in components.keys()
        }
    }


@app.get("/status")
async def status():
    """Get current conversation status."""
    conv_manager = components.get('conversation_manager')
    
    if not conv_manager:
        return {"error": "Not initialized"}
    
    context = conv_manager.get_context()
    
    return {
        "state": context.state.value,
        "transcript_segments": len(context.transcript_segments),
        "backchannel_count": len(context.backchannel_history),
        "current_transcript": conv_manager.get_user_transcript_current_turn()
    }


# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=config.host,
        port=config.port,
        reload=False
    )
