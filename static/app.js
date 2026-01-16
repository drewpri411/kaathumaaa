// Voice Conversation Agent - Client-side JavaScript

let pc = null;
let localStream = null;
let isConnected = false;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const transcriptDiv = document.getElementById('transcript');

// Event listeners
startBtn.addEventListener('click', startConversation);
stopBtn.addEventListener('click', stopConversation);

async function startConversation() {
    try {
        updateStatus('connecting', 'Connecting...');
        startBtn.disabled = true;

        // Get microphone access
        localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        });

        // Create peer connection
        pc = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' }
            ]
        });

        // Add local audio track
        localStream.getTracks().forEach(track => {
            pc.addTrack(track, localStream);
        });

        // Handle incoming audio
        pc.ontrack = (event) => {
            console.log('Received remote track');
            const audio = new Audio();
            audio.srcObject = event.streams[0];
            audio.play();
        };

        // Handle connection state changes
        pc.onconnectionstatechange = () => {
            console.log('Connection state:', pc.connectionState);
            if (pc.connectionState === 'connected') {
                updateStatus('connected', 'Connected - Listening');
                stopBtn.disabled = false;
            } else if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
                stopConversation();
            }
        };

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Send offer to server
        const response = await fetch('/offer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                offer: {
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type
                }
            })
        });

        if (!response.ok) {
            throw new Error('Failed to connect to server');
        }

        const data = await response.json();

        // Set remote description
        await pc.setRemoteDescription(new RTCSessionDescription(data.answer));

        isConnected = true;

        // Start polling for status updates
        startStatusPolling();

    } catch (error) {
        console.error('Error starting conversation:', error);
        updateStatus('disconnected', 'Error: ' + error.message);
        startBtn.disabled = false;
        stopConversation();
    }
}

function stopConversation() {
    // Stop status polling
    stopStatusPolling();

    // Close peer connection
    if (pc) {
        pc.close();
        pc = null;
    }

    // Stop local stream
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }

    isConnected = false;
    updateStatus('disconnected', 'Disconnected');
    startBtn.disabled = false;
    stopBtn.disabled = true;
}

function updateStatus(state, message) {
    statusDiv.className = `status ${state}`;
    statusDiv.textContent = message;
}

// Status polling
let statusInterval = null;

function startStatusPolling() {
    statusInterval = setInterval(async () => {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            // Update status based on conversation state
            if (data.state === 'user_speaking') {
                updateStatus('speaking', '🎤 You are speaking...');
            } else if (data.state === 'agent_speaking') {
                updateStatus('speaking', '🔊 Agent is speaking...');
            } else if (data.state === 'agent_thinking') {
                updateStatus('speaking', '💭 Agent is thinking...');
            } else {
                updateStatus('connected', '👂 Listening...');
            }

            // Update transcript if available
            if (data.current_transcript) {
                // This is simplified - in a real implementation,
                // you'd want to use WebSocket for real-time updates
            }

        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 500); // Poll every 500ms
}

function stopStatusPolling() {
    if (statusInterval) {
        clearInterval(statusInterval);
        statusInterval = null;
    }
}

// Transcript display functions
function addTranscriptItem(speaker, text) {
    // Remove empty state if present
    const emptyState = transcriptDiv.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    const item = document.createElement('div');
    item.className = `transcript-item ${speaker}`;

    const label = document.createElement('div');
    label.className = 'speaker-label';
    label.textContent = speaker === 'user' ? '👤 You' : '🤖 Agent';

    const text_elem = document.createElement('div');
    text_elem.className = 'transcript-text';
    text_elem.textContent = text;

    item.appendChild(label);
    item.appendChild(text_elem);

    transcriptDiv.appendChild(item);

    // Scroll to bottom
    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
}

// For demo purposes - in production, use WebSocket for real-time updates
// Example usage:
// addTranscriptItem('user', 'Hello, how are you?');
// addTranscriptItem('agent', 'I\'m doing well, thank you for asking!');
