# webrtc_server.py

import asyncio
from cgi import print_arguments
import logging
import threading
import time
import uuid
import queue
import io 

import numpy as np
import pyttsx3 # For Text-to-Speech
import requests
# REMOVED: from av import AudioFrame # This was causing the ImportError
from typing import Dict, List
from av import AudioFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, AudioStreamTrack, RTCIceCandidate # CORRECTED: AudioFrame is imported here
from aiortc.contrib.media import MediaBlackhole, MediaRecorder, MediaPlayer 
from aiortc.rtcrtpsender import RTCRtpSender
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.signal import resample # Ensure
from aiortc.sdp import candidate_from_sdp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-server")

# Configuration
MAIN_FASTAPI_SERVER_URL = "http://localhost:8000" # Your existing main.py server
WEBRTC_SERVER_PORT = 8001

# Global store for active peer connections and TTS engines
pcs = set()
tts_engines: Dict[str, pyttsx3.Engine] = {}

# Thread-safe queue for TTS text and audio data
tts_queue = queue.Queue() # For text to be spoken by TTS engine
audio_output_queues: Dict[str, queue.Queue] = {} # For PCM audio data from TTS to WebRTC sender

# Audio configuration for Whisper compatibility
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1
WHISPER_DTYPE = np.float32

# Silence detection for WebRTC input stream
WEBRTC_AUDIO_CHUNK_SIZE = WHISPER_SAMPLE_RATE // 2 # 0.5 seconds of audio at 16kHz
SILENCE_THRESHOLD_RMS = 0.015
CONSECUTIVE_SILENT_CHUNKS = 6 # 3 seconds of silence (6 * 0.5s chunks) before committing

class Offer(BaseModel):
    sdp: str
    type: str

class RTCIceCandidateJSON(BaseModel):
    candidate: str
    sdpMid: str = None
    sdpMLineIndex: int = None

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Store server's local ICE candidates for each session
server_ice_candidates: Dict[str, List[RTCIceCandidateJSON]] = {}


# --- TTS Worker Thread ---
def tts_engine_worker():
    """
    Dedicated worker for pyttsx3 to avoid blocking.
    Manages TTS engine lifecycle and puts generated audio into session-specific queues.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 180) # Speed of speech
    logger.info("TTS Engine worker started.")

    def on_word(name, loc, length):
        pass 

    def on_end(name, completed):
        pass 

    engine.connect('started-word', on_word)
    engine.connect('finished-utterance', on_end)

    while True:
        task = tts_queue.get()
        if task is None: # Sentinel for shutdown
            logger.info("TTS Engine worker received shutdown signal.")
            break

        session_id = task["session_id"]
        text = task["text"]
        
        try:
            # Simulate PCM audio generation
            dummy_pcm_data = np.random.rand(WHISPER_SAMPLE_RATE * len(text) // 5, 1).astype(WHISPER_DTYPE) * 0.1 
            num_frames_per_chunk = WHISPER_SAMPLE_RATE // 10 
            for i in range(0, len(dummy_pcm_data), num_frames_per_chunk):
                chunk = dummy_pcm_data[i:i + num_frames_per_chunk]
                if session_id in audio_output_queues:
                    audio_output_queues[session_id].put(chunk)
                time.sleep(0.01) # Simulate real-time audio output
            
            if session_id in audio_output_queues:
                audio_output_queues[session_id].put(None) # Signal end of TTS for this utterance

            logger.info(f"TTS Engine worker generated audio for session {session_id}: '{text}'")

        except Exception as e:
            logger.error(f"TTS Engine worker error for session {session_id}: {e}")
        finally:
            tts_queue.task_done()

    engine.stop()
    logger.info("TTS Engine worker stopped.")

tts_worker_thread = threading.Thread(target=tts_engine_worker, daemon=True)
tts_worker_thread.start()


# --- Custom WebRTC Audio Track for Sending TTS ---
class WebRTCAudioSenderTrack(AudioStreamTrack):
    """
    A custom AudioStreamTrack that continuously sends audio data
    from a queue to the WebRTC peer.
    """
    def __init__(self, session_id):
        super().__init__()
        self.session_id = session_id
        self.queue = audio_output_queues.get(session_id)
        if not self.queue:
            self.queue = queue.Queue()
            audio_output_queues[session_id] = self.queue
        self.last_sync_time = None
        self.samples_sent = 0
        logger.info(f"WebRTCAudioSenderTrack initialized for session {session_id}")

    async def recv(self):
        """
        Read new audio data from the queue and return an AudioFrame.
        This method is called by aiorttc when it needs more audio data to send.
        """
        if self.queue.empty() and self.last_sync_time is not None:
            await asyncio.sleep(0.01) 

        chunk = await asyncio.to_thread(self.queue.get) 
        
        if chunk is None: 
            self.queue.put(None) 
            raise EOFError 

        if len(chunk.shape) == 1: 
            chunk = chunk.reshape(-1, 1)

        frame = AudioFrame(format="s16", layout="mono", samples=chunk.shape[0], sample_rate=WHISPER_SAMPLE_RATE)
        frame.samples = (chunk * (2**15 - 1)).astype(np.int16).tobytes() 

        # Simulate real-time by controlling timestamp
        if self.last_sync_time is None:
            self.last_sync_time = time.time()
            self.samples_sent = 0
        
        self.samples_sent += chunk.shape[0]
        expected_time = self.samples_sent / WHISPER_SAMPLE_RATE
        current_time = time.time() - self.last_sync_time
        
        if expected_time > current_time:
            await asyncio.sleep(expected_time - current_time)

        return frame

# --- Callbacks for LLM Responses ---
@app.post("/webrtc_server_callback")
async def webrtc_server_callback(request: Request):
    """
    Endpoint for the main FastAPI server (localhost:8000) to call
    with the LLM's text response.
    """
    try:
        response_data = await request.json()
        session_id = response_data.get("session_id")
        llm_response_text = response_data.get("response", "No response from LLM.")
        final_transcription = response_data.get("final_transcription", "")

        logger.info(f"Server: Received LLM response for session {session_id}. Transcription: '{final_transcription}', LLM Response: '{llm_response_text}'")

        if session_id in pcs_by_id: 
            tts_queue.put({"session_id": session_id, "text": llm_response_text})
            
            if "goodbye" in llm_response_text.lower():
                logger.info(f"Server: Session {session_id} - LLM said goodbye.")

        return {"status": "received", "message": "Response queued for TTS."}
    except Exception as e:
        logger.error(f"Server: Error receiving LLM response callback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Store peer connections by ID to manage them across requests
pcs_by_id: Dict[str, RTCPeerConnection] = {}
audio_buffers: Dict[str, bytearray] = {}
silent_chunk_counts: Dict[str, int] = {}

# --- WebRTC Signaling Endpoints ---
@app.post("/offer")
async def offer(offer: Offer):
    session_id = str(uuid.uuid4())
    pc = RTCPeerConnection()
    pcs.add(pc) 
    pcs_by_id[session_id] = pc
    audio_buffers[session_id] = bytearray()
    silent_chunk_counts[session_id] = 0
    server_ice_candidates[session_id] = [] # Initialize list for server's local candidates
    
    logger.info(f"Server: Session {session_id} - Created RTCPeerConnection")

    @pc.on("track")
    async def on_track(track):
        logger.info(f"Server: Session {session_id} - Track {track.kind} received")
        if track.kind == "audio":
            asyncio.create_task(process_webrtc_audio(session_id, track))

    @pc.on("icecandidate")
    async def on_ice_candidate(candidate: RTCIceCandidate):
        if candidate:
            logger.info(f"Server: Session {session_id} - Local ICE candidate generated: {candidate.sdpMid} {candidate.candidate}")
            server_ice_candidates[session_id].append(RTCIceCandidateJSON(
                candidate=candidate.candidate,
                sdpMid=candidate.sdpMid,
                sdpMLineIndex=candidate.sdpMLineIndex
            ))

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"Server: Session {session_id} - Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            if session_id in pcs_by_id: del pcs_by_id[session_id]
            if session_id in audio_buffers: del audio_buffers[session_id]
            if session_id in silent_chunk_counts: del silent_chunk_counts[session_id]
            if session_id in audio_output_queues:
                audio_output_queues[session_id].put(None) 
                del audio_output_queues[session_id]
            if session_id in server_ice_candidates: del server_ice_candidates[session_id] 
            logger.warning(f"Server: Session {session_id} - PeerConnection failed and closed.")
        elif pc.connectionState == "closed":
            pcs.discard(pc)
            if session_id in pcs_by_id: del pcs_by_id[session_id]
            if session_id in audio_buffers: del audio_buffers[session_id]
            if session_id in silent_chunk_counts: del silent_chunk_counts[session_id]
            if session_id in audio_output_queues:
                audio_output_queues[session_id].put(None) 
                del audio_output_queues[session_id]
            if session_id in server_ice_candidates: del server_ice_candidates[session_id] 
            logger.info(f"Server: Session {session_id} - PeerConnection closed.")

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

    audio_sender = pc.addTrack(WebRTCAudioSenderTrack(session_id))
    logger.info(f"Server: Session {session_id} - Added WebRTCAudioSenderTrack for TTS.")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp, 
        "type": pc.localDescription.type, 
        "session_id": session_id,
        "server_ice_candidates": server_ice_candidates[session_id] 
    }

def parse_sdp_candidate(sdp: str) -> dict:
    """
    Parse a candidate string from SDP into a dict compatible with RTCIceCandidate.
    Example input:
    candidate:842163049 1 udp 1677729535 192.168.1.10 52123 typ srflx raddr 0.0.0.0 rport 0
    """
    parts = sdp.split()
    if parts[0].startswith("candidate:"):
        foundation = parts[0].split(":")[1]
    else:
        raise ValueError(f"Invalid candidate line: {sdp}")

    candidate = {
        "foundation": foundation,
        "component": int(parts[1]),
        "protocol": parts[2].lower(),
        "priority": int(parts[3]),
        "ip": parts[4],
        "port": int(parts[5]),
        "type": parts[7],
    }

    # Optional extensions
    for i in range(8, len(parts) - 1, 2):
        attr = parts[i]
        val = parts[i + 1]
        if attr == "raddr":
            candidate["relatedAddress"] = val
        elif attr == "rport":
            candidate["relatedPort"] = int(val)
        elif attr == "tcptype":
            candidate["tcpType"] = val

    return candidate
# Endpoint to receive ICE candidates from the browser
@app.post("/candidate/{session_id}")
async def candidate_from_browser(session_id: str, candidate_json: RTCIceCandidateJSON): 
    pc = pcs_by_id.get(session_id)
    if not pc:
        logger.warning(f"Server: Session {session_id} - Received candidate for unknown PC.")
        raise HTTPException(status_code=404, detail="Session not found")
    print(pc)

    try:
        cand_dict = parse_sdp_candidate(candidate_json.candidate)
        cand = RTCIceCandidate(
            component=cand_dict["component"],
            foundation=cand_dict["foundation"],
            ip=cand_dict["ip"],
            port=cand_dict["port"],
            priority=cand_dict["priority"],
            protocol=cand_dict["protocol"],
            type=cand_dict["type"],
            relatedAddress=cand_dict.get("relatedAddress"),
            relatedPort=cand_dict.get("relatedPort"),
            sdpMid=candidate_json.sdpMid,
            sdpMLineIndex=candidate_json.sdpMLineIndex,
            tcpType=cand_dict.get("tcpType"),
        )

        await pc.addIceCandidate(cand)
        logger.info(f"Server: Session {session_id} - Successfully added remote ICE candidate: {candidate_json.candidate}")
    except Exception as e:
        logger.error(f"Server: Session {session_id} - Error adding remote ICE candidate {candidate_json.candidate}: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding ICE candidate: {e}")
    
    return {"status": "ok"}


# --- Audio Processing for WebRTC Stream ---
async def process_webrtc_audio(session_id: str, track: AudioStreamTrack):
    """
    Receives audio frames from the WebRTC track, buffers them,
    performs silence detection, and sends to the main FastAPI server for LLM.
    """
    logger.info(f"Server: Session {session_id} - Started processing incoming audio track.")
    pc = pcs_by_id.get(session_id)
    if not pc:
        logger.warning(f"Server: Session {session_id} - PeerConnection not found for audio processing.")
        return

    audio_buffer = audio_buffers[session_id]
    silent_chunk_count = silent_chunk_counts[session_id]

    try:
        while True:
            frame = await track.recv() 

            # Robust check for frame.samples type
            if not isinstance(frame.samples, (bytes, bytearray)):
                logger.error(f"Server: Session {session_id} - Received non-bytes-like frame.samples ({type(frame.samples)}). Skipping frame.")
                continue # Skip this malformed frame

            samples = np.frombuffer(frame.samples, dtype=np.int16).astype(WHISPER_DTYPE) / (2**15)
            
            if frame.sample_rate != WHISPER_SAMPLE_RATE:
                # `resample` is imported at the top-level
                num_samples = int(len(samples) * WHISPER_SAMPLE_RATE / frame.sample_rate)
                samples = resample(samples, num_samples)
            
            if frame.channels > WHISPER_CHANNELS:
                samples = samples.reshape(-1, frame.channels).mean(axis=1)

            # Robust type check before extending audio_buffer
            if not isinstance(samples, np.ndarray):
                logger.error(f"Server: Session {session_id} - 'samples' is not a NumPy array after processing ({type(samples)}). Cannot append to buffer. Skipping frame.")
                continue # Skip this frame if it's not a numpy array

            bytes_to_append = samples.tobytes()
            
            if not isinstance(bytes_to_append, (bytes, bytearray)):
                logger.error(f"Server: Session {session_id} - Expected bytes-like object from samples.tobytes(), got {type(bytes_to_append)}. Skipping frame.")
                continue # Skip if tobytes() returns something unexpected

            audio_buffer.extend(bytes_to_append)

            while len(audio_buffer) >= WEBRTC_AUDIO_CHUNK_SIZE * WHISPER_DTYPE().itemsize:
                chunk_bytes = audio_buffer[:WEBRTC_AUDIO_CHUNK_SIZE * WHISPER_DTYPE().itemsize]
                current_audio_chunk = np.frombuffer(chunk_bytes, dtype=WHISPER_DTYPE)
                
                rms = np.sqrt(np.mean(current_audio_chunk**2))
                is_silent = rms < SILENCE_THRESHOLD_RMS
                
                if is_silent:
                    silent_chunk_count += 1
                else:
                    silent_chunk_count = 0 

                del audio_buffer[:WEBRTC_AUDIO_CHUNK_SIZE * WHISPER_DTYPE().itemsize]

                if silent_chunk_count >= CONSECUTIVE_SILENT_CHUNKS:
                    if len(audio_buffer) > 0: 
                        logger.info(f"Server: Session {session_id} - Detected {CONSECUTIVE_SILENT_CHUNKS * 0.5}s silence. Committing audio to main FastAPI.")
                        
                        full_audio_for_whisper = np.frombuffer(audio_buffer, dtype=WHISPER_DTYPE)
                        
                        audio_buffer[:] = b''
                        silent_chunk_count = 0
                        
                        if len(full_audio_for_whisper) > 0:
                            logger.info(f"Server: Session {session_id} - Sending {len(full_audio_for_whisper)} samples to main FastAPI /commit_audio.")
                            
                            audio_file = io.BytesIO(full_audio_for_whisper.tobytes())
                            files = {"chunk": ("audio.wav", audio_file, "application/octet-stream")}
                            data = {
                                "session_id": session_id,
                                "callback_url": f"http://localhost:{WEBRTC_SERVER_PORT}/webrtc_server_callback"
                            }
                            
                            try:
                                await asyncio.to_thread(requests.post, f"{MAIN_FASTAPI_SERVER_URL}/commit_audio", files=files, data=data, timeout=10)
                                logger.info(f"Server: Session {session_id} - Successfully sent commit to main FastAPI.")
                            except requests.exceptions.RequestException as e:
                                logger.error(f"Server: Session {session_id} - Error sending commit to main FastAPI: {e}")
                            except Exception as e:
                                logger.error(f"Server: Session {session_id} - Unexpected error during commit: {e}")
                    else:
                        silent_chunk_count = 0

            silent_chunk_counts[session_id] = silent_chunk_count

    except Exception as e:
        logger.error(f"Server: Session {session_id} - Error processing WebRTC audio: {e}")
    finally:
        logger.info(f"Server: Session {session_id} - Stopped processing incoming audio track.")
        if pc and pc.connectionState != "closed":
            await pc.close()

# --- Cleanup on shutdown ---
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server: Shutting down WebRTC server...")
    for pc in pcs:
        await pc.close()
    tts_queue.put(None)
    tts_worker_thread.join(timeout=5)
    logger.info("Server: WebRTC server shut down.")


if __name__ == "__main__":
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WEBRTC_SERVER_PORT)
