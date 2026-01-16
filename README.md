# Voice Conversation Agent

A server-based voice agent system that improves upon LiveKit's baseline VAD through enhanced turn detection, natural backchannel responses, and better handling of pauses and interruptions.

## Features

### 🎯 Enhanced Turn Detection
- **Multi-signal fusion**: Combines silence duration (40%), linguistic completeness (35%), and conversation context (25%)
- **Reduces false interruptions**: Won't cut off users mid-sentence
- **Adaptive thresholds**: Learns user speaking patterns over time

### 💬 Natural Backchannels
- **5 types**: "mm-hmm", "okay", "yeah", "I see", "right"
- **Context-aware selection**: Chooses appropriate responses based on conversation
- **Safe zone timing**: 300ms delay with abort capability if user resumes speaking
- **Anti-repetition**: Varies backchannel types naturally

### 🏗️ Architecture
- **Event-driven design**: Loosely coupled components via event bus
- **Chunked transcription**: 1.5s chunks with 0.5s overlap for continuous STT
- **Multi-channel audio**: Separate channels for agent speech and backchannels
- **WebRTC transport**: Real-time audio streaming

## Technology Stack

- **Framework**: FastAPI + asyncio
- **Audio Transport**: WebRTC (aiortc) with STUN server
- **VAD**: Silero VAD (ONNX model)
- **STT**: OpenAI Whisper API
- **LLM**: OpenAI GPT-4o-mini
- **TTS**: OpenAI TTS (alloy voice)
- **Audio Format**: 16kHz mono PCM

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- OpenAI API key

### 2. Installation

```bash
# Navigate to project directory
cd voice-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Silero VAD Model

Download the Silero VAD ONNX model:

1. Visit: https://github.com/snakers4/silero-vad/releases
2. Download `silero_vad.onnx` (latest version)
3. Place it in the `models/` directory:
   ```
   voice-agent/models/silero_vad.onnx
   ```

### 4. Configure API Keys

Copy the example environment file and add your API keys:

```bash
# Copy example file
copy .env.example .env

# Edit .env and add your key:
# OPENAI_API_KEY=sk-your-openai-api-key-here
```

**API Key Required:**
- **OPENAI_API_KEY**: Used for Whisper Speech-to-Text, TTS (Text-to-Speech), and LLM (GPT-4o-mini)

### 5. Generate Backchannel Audio Files

You have **two options**:

**Option A: Generate with OpenAI TTS (Recommended)**

Run the backchannel generation script (requires OpenAI API key):

```bash
python generate_backchannels.py
```

This will create 5 natural-sounding WAV files in the `backchannels/` directory using OpenAI's TTS API.

**Option B: Create Placeholder Audio (No API Key Required)**

If you don't have an OpenAI API key yet:

```bash
python create_placeholder_backchannels.py
```

This creates simple tone-based placeholders. You can replace them later with real audio using Option A.

### 6. Run the Server

```bash
# Start the server
python -m server.main

# Or use uvicorn directly
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### 7. Open Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

Click "Start Conversation" and grant microphone permissions.

## Project Structure

```
voice-agent/
├── server/                      # Server-side Python code
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration management
│   ├── event_bus.py            # Event system
│   ├── conversation_manager.py # State management
│   ├── audio_pipeline.py       # Audio buffering
│   ├── vad_processor.py        # Voice activity detection
│   ├── stt_client.py           # Whisper STT client
│   ├── transcription_coordinator.py  # STT coordination
│   ├── linguistic_analyzer.py  # Text analysis
│   ├── turn_detector.py        # Turn detection logic
│   ├── backchannel_*.py        # Backchannel system (5 files)
│   ├── llm_client.py           # Gemini LLM client
│   ├── tts_client.py           # OpenAI TTS client
│   ├── response_coordinator.py # Response generation
│   ├── audio_mixer.py          # Multi-channel mixing
│   └── webrtc_handler.py       # WebRTC connections
├── models/
│   └── silero_vad.onnx         # VAD model (download separately)
├── backchannels/               # Generated backchannel audio
│   ├── mmhmm.wav
│   ├── okay.wav
│   ├── yeah.wav
│   ├── i_see.wav
│   └── right.wav
├── static/                     # Web interface
│   ├── index.html
│   └── app.js
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .env                       # Your API keys (create this)
├── generate_backchannels.py   # Backchannel generator
└── README.md                  # This file
```

## Configuration

All tunable parameters are in `.env` file. Key settings:

### Turn Detection
- `TURN_END_SCORE_THRESHOLD=65` - Threshold for turn ending (0-100)
- `SHORT_PAUSE_MS=400` - Short pause duration
- `LONG_PAUSE_MS=1500` - Long pause duration

### Backchannel
- `BACKCHANNEL_BASE_PROBABILITY=0.4` - Base 40% chance
- `BACKCHANNEL_MIN_INTERVAL_S=5` - Minimum 5s between backchannels
- `BACKCHANNEL_SAFE_ZONE_MS=300` - 300ms wait before playing

### Scoring Weights
- `SILENCE_WEIGHT=0.4` - 40% weight for silence
- `LINGUISTIC_WEIGHT=0.35` - 35% weight for linguistics
- `CONTEXT_WEIGHT=0.25` - 25% weight for context

## API Endpoints

- `GET /` - Web interface
- `POST /offer` - WebRTC offer/answer exchange
- `GET /health` - Health check
- `GET /status` - Current conversation status

## Troubleshooting

### "VAD model not found"
- Download `silero_vad.onnx` and place in `models/` directory

### "Backchannel directory not found"
- Run `python generate_backchannels.py` to create backchannel audio files

### "OpenAI API key not set"
- Add `OPENAI_API_KEY` to `.env` file

### "Gemini API key not set"
- Add `GEMINI_API_KEY` to `.env` file

### WebRTC connection fails
- Check firewall settings
- Ensure STUN server is accessible
- Try using HTTPS (required for some browsers)

## Development

To modify the system:

1. **Adjust turn detection**: Edit `turn_detector.py` scoring logic
2. **Change backchannel behavior**: Modify `backchannel_trigger.py` probability calculation
3. **Add new backchannels**: Add to `generate_backchannels.py` and regenerate
4. **Tune parameters**: Edit `.env` file values

## License

MIT License - See LICENSE file for details

## Credits

- Silero VAD: https://github.com/snakers4/silero-vad
- OpenAI Whisper & TTS: https://openai.com/
- Google Gemini: https://deepmind.google/technologies/gemini/
- aiortc: https://github.com/aiortc/aiortc
