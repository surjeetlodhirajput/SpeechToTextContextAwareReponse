# webrtc_server.py

import asyncio
import logging
import os
import threading
import time
import uuid
import queue
import io
import wave
import numpy as np
import pyttsx3
import requests
from typing import Dict, List
from av import AudioFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, AudioStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.signal import resample
import soundfile as sf
from gtts import gTTS
import fractions

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-server")

# --- Config ---
MAIN_FASTAPI_SERVER_URL = "http://localhost:8000"
WEBRTC_SERVER_PORT = 8001

# --- Globals ---
pcs = set()
pcs_by_id: Dict[str, RTCPeerConnection] = {}
tts_queue = queue.Queue()
audio_output_queues: Dict[str, queue.Queue] = {}
pc_outbound_send_ttrack: Dict[str, "OutboundAudioTrack"] = {}
audio_buffers: Dict[str, bytearray] = {}
silent_chunk_counts: Dict[str, int] = {}
server_ice_candidates: Dict[str, List] = {}

# Recording manager globals (managed by class)
# Audio config
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1
WHISPER_DTYPE = np.float32

WEBRTC_SAMPLE_RATE = 48000
WEBRTC_CHANNELS = 1
WEBRTC_FRAME_DURATION = 0.02  # 20 ms frame duration

# RMS threshold for silence detection
SILENCE_THRESHOLD_RMS = 0.015

# VAD: 5 seconds of silence => number of 20ms frames
SILENCE_SECONDS = 5.0
FRAMES_PER_SECOND = int(1.0 / WEBRTC_FRAME_DURATION)  # 50
CONSECUTIVE_SILENT_FRAMES_TO_ROTATE = int(SILENCE_SECONDS * FRAMES_PER_SECOND)

# Minimum length for upload (>=2s)
MIN_UPLOAD_SECONDS = 2.0

# --- Models ---
class Offer(BaseModel):
    sdp: str
    type: str

class RTCIceCandidateJSON(BaseModel):
    candidate: str
    sdpMid: str = None
    sdpMLineIndex: int = None

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- TTS worker (pyttsx3) - same pattern as before ----------------
def create_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    return engine

def tts_engine_worker():
    logger.info("TTS Engine worker started.")
    engine = None
    while True:
        task = tts_queue.get()
        if task is None:
            logger.info("TTS worker shutdown signal received.")
            tts_queue.task_done()
            if engine:
                try:
                    engine.stop()
                except:
                    pass
            break

        session_id = task["session_id"]
        text = task["text"]
        temp_file_name = f"tts_uuid_{uuid.uuid4()}.wav"
        try:
            if engine:
                try:
                    engine.stop()
                except:
                    pass
            engine = create_tts_engine()
            engine.save_to_file(text, temp_file_name)

            def run_tts():
                engine.runAndWait()

            tts_thread = threading.Thread(target=run_tts, daemon=True)
            tts_thread.start()
            tts_thread.join(timeout=10)

            if tts_thread.is_alive():
                try:
                    engine.stop()
                except:
                    pass
                continue

            if not os.path.exists(temp_file_name):
                continue

            with wave.open(temp_file_name, 'rb') as wf:
                orig_rate = wf.getframerate()
                channels = wf.getnchannels()
                nframes = wf.getnframes()
                pcm_data = wf.readframes(nframes)
                audio = np.frombuffer(pcm_data, dtype=np.int16)
                if channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
                if orig_rate != WEBRTC_SAMPLE_RATE:
                    num_samples = int(len(audio) * WEBRTC_SAMPLE_RATE / orig_rate)
                    audio = resample(audio, num_samples)
                audio = audio.astype(np.float32) / (2 ** 15)

                chunk_size = int(WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_DURATION)
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    if session_id in audio_output_queues:
                        audio_output_queues[session_id].put(chunk)
                        # sleep to emulate realtime
                        time.sleep(WEBRTC_FRAME_DURATION)
        except Exception as e:
            logger.exception("TTS worker error: %s", e)
        finally:
            tts_queue.task_done()
            if os.path.exists(temp_file_name):
                try:
                    os.remove(temp_file_name)
                except:
                    pass
            if engine:
                try:
                    engine.stop()
                except:
                    pass
            engine = None

tts_worker_thread = threading.Thread(target=tts_engine_worker, daemon=True)
tts_worker_thread.start()

# ---------------- Utility helpers ----------------
def convert_48khz_audio_to_16khz(samples_48k: np.ndarray) -> np.ndarray:
    """Simple downsample by picking every 3rd sample (48k -> 16k).
       Input expected in float32 normalized [-1,1]. Returns int16."""
    if samples_48k.ndim > 1:
        samples_48k = samples_48k.mean(axis=-1)
    # decimate by 3
    samples_16k = samples_48k[::5]
    samples_16k = np.clip(samples_16k, -1.0, 1.0)
    return (samples_16k * 32767.0).astype(np.int16)


def float_samples_to_int16(samples: np.ndarray) -> np.ndarray:
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16)

# ---------------- Recording Manager ----------------
class RecordingManager:
    """
    Manages per-session wave file, writing samples, deciding when to finalize/upload.
    Thread-safe operations using a lock per session.
    """
    def __init__(self, recordings_dir="recordings"):
        self.recordings_dir = recordings_dir
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.lock = threading.Lock()
        # maps session_id -> dict with file handle and metadata
        self.sessions: Dict[str, Dict] = {}

    def ensure_session(self, session_id: str):
        with self.lock:
            if session_id not in self.sessions:
                filename = self._new_filename(session_id)
                wf = wave.open(filename, 'wb')
                wf.setnchannels(WHISPER_CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(WHISPER_SAMPLE_RATE)
                self.sessions[session_id] = {
                    "wave": wf,
                    "filename": filename,
                    "created_at": time.time(),
                    "frames_written": 0,  # number of int16 samples written (per channel)
                    "silent_frames": 0,  # consecutive silent 20ms frames
                }
                logger.info("RecordingManager: Created new wav %s for %s", filename, session_id)
            return self.sessions[session_id]

    def _new_filename(self, session_id: str) -> str:
        ts = int(time.time())
        unique = uuid.uuid4().hex[:8]
        return os.path.join(self.recordings_dir, f"rc_{session_id}_{ts}_{unique}.wav")

    def write_samples(self, session_id: str, audio_int16: np.ndarray):
        """
        audio_int16: int16 numpy array (mono) at 16k sample rate.
        """
        sess = self.ensure_session(session_id)
        wf = sess["wave"]
        # write bytes
        try:
            wf.writeframes(audio_int16.tobytes())
            sess["frames_written"] += len(audio_int16)
        except Exception:
            logger.exception("RecordingManager: Error writing samples for %s", session_id)

    def get_duration_seconds(self, session_id: str) -> float:
        sess = self.sessions.get(session_id)
        if not sess:
            return 0.0
        frames = sess["frames_written"]
        return frames / float(WHISPER_SAMPLE_RATE)

    def mark_speech(self, session_id: str):
        """
        Reset silent counter on speech frame.
        """
        sess = self.ensure_session(session_id)
        sess["silent_frames"] = 0

    def mark_silence_and_maybe_rotate(self, session_id: str) -> bool:
        """
        Increment silent frame counter. If counter reaches threshold,
        decide whether to finalize/upload and create a new file.
        Returns True if rotation (new file created) happened, else False.
        """
        sess = self.ensure_session(session_id)
        sess["silent_frames"] += 1
        if sess["silent_frames"] >= CONSECUTIVE_SILENT_FRAMES_TO_ROTATE:
            # check duration
            duration = self.get_duration_seconds(session_id)
            filename = sess["filename"]
            logger.info("RecordingManager: Detected %d silent frames for %s (duration %.2fs).",
                        sess["silent_frames"], session_id, duration)
            if duration >= MIN_UPLOAD_SECONDS:
                # finalize current file and upload
                logger.info("RecordingManager: Finalizing and uploading %s (%.2fs) for %s",
                            filename, duration, session_id)
                self._close_session_wave(session_id)  # close handle before upload
                # upload in background thread
                t = threading.Thread(target=self._upload_and_delete, args=(session_id, filename), daemon=True)
                t.start()
                # create new file for next segment
                self._create_new_session_wave(session_id)
                return True
            else:
                # If file shorter than MIN_UPLOAD_SECONDS -> keep writing to same file (do not rotate)
                logger.info("RecordingManager: File %s for %s is < %.1fs, will continue writing to it.",
                            filename, session_id, MIN_UPLOAD_SECONDS)
                # reset silent counter so we don't repeatedly log
                sess["silent_frames"] = 0
                return False
        return False

    def _create_new_session_wave(self, session_id: str):
        with self.lock:
            # create new file replacing session
            filename = self._new_filename(session_id)
            try:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(WHISPER_CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(WHISPER_SAMPLE_RATE)
                self.sessions[session_id] = {
                    "wave": wf,
                    "filename": filename,
                    "created_at": time.time(),
                    "frames_written": 0,
                    "silent_frames": 0,
                }
                logger.info("RecordingManager: Created rotated wav %s for %s", filename, session_id)
            except Exception:
                logger.exception("RecordingManager: Error creating rotated wav for %s", session_id)

    def _close_session_wave(self, session_id: str):
        with self.lock:
            sess = self.sessions.get(session_id)
            if sess:
                wf = sess.get("wave")
                if wf:
                    try:
                        wf.close()
                    except:
                        logger.exception("RecordingManager: Error closing wave for %s", session_id)

    def finalize_and_upload_on_close(self, session_id: str):
        """
        Called when session ends. Ensure current file closed and uploaded.
        """
        with self.lock:
            sess = self.sessions.get(session_id)
            if not sess:
                return
            filename = sess["filename"]
            # close wave
            try:
                sess["wave"].close()
            except:
                logger.exception("RecordingManager: Error closing final wave for %s", session_id)
            # remove session entry
            del self.sessions[session_id]
        # Upload final file in background
        logger.info("RecordingManager: Finalizing session %s -> uploading %s", session_id, filename)
        t = threading.Thread(target=self._upload_and_delete, args=(session_id, filename), daemon=True)
        t.start()

    def _upload_and_delete(self, session_id: str, filename: str):
        """
        Upload file to MAIN_FASTAPI_SERVER_URL /upload_wav.
        Delete only after successful upload.
        """
        try:
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                logger.warning("RecordingManager: Upload skipped because file missing/empty: %s", filename)
                return
            # double check duration on disk (more robust)
            try:
                with wave.open(filename, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
            except Exception:
                duration = 0.0

            if duration < MIN_UPLOAD_SECONDS:
                logger.info("RecordingManager: Upload skipped because duration %.2fs < %.2fs for %s",
                            duration, MIN_UPLOAD_SECONDS, filename)
                # If file is too short, keep file on disk (no deletion). This ensures caller can append later
                return

            with open(filename, 'rb') as f:
                files = {'file': (os.path.basename(filename), f, 'audio/wav')}
                data = {
                    'session_id': session_id,
                    'callback_url': f"http://localhost:{WEBRTC_SERVER_PORT}/webrtc_server_callback"
                }
                logger.info("RecordingManager: Uploading %s for session %s", filename, session_id)
                resp = requests.post(f"{MAIN_FASTAPI_SERVER_URL}/upload_wav", files=files, data=data, timeout=30)
                resp.raise_for_status()
                logger.info("RecordingManager: Upload succeeded for %s, response: %s", filename, resp.text)
            # delete only after success
            try:
                os.remove(filename)
                logger.info("RecordingManager: Deleted local file %s after upload", filename)
            except OSError:
                logger.exception("RecordingManager: Error deleting local file %s", filename)
        except requests.exceptions.RequestException as e:
            logger.exception("RecordingManager: Upload failed for %s: %s", filename, e)
        except Exception:
            logger.exception("RecordingManager: Unexpected error while uploading %s", filename)


recording_manager = RecordingManager()

# ---------------- Outbound audio track (server->client TTS streaming) ----------------
class WebRTCAudioSenderTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id
        self.queue = audio_output_queues.get(session_id)
        if not self.queue:
            self.queue = queue.Queue()
            audio_output_queues[session_id] = self.queue
        self.last_sync_item = None
        self.sample_sent = 0
        logger.info(f"WEBRTCAudioSender Track Initialise for session {session_id}")

    async def recv(self) -> AudioFrame:
        try:
            try:
                chunk = await asyncio.wait_for(asyncio.to_thread(self.queue.get, timeout=0.1), timeout=0.1)
            except (asyncio.TimeoutError, queue.Empty):
                silence_samples = int(WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_DURATION)
                chunk = np.zeros(silence_samples, dtype=WHISPER_DTYPE)

            if chunk is None:
                raise EOFError("End of audio stream")

            if not isinstance(chunk, np.ndarray):
                logger.error(f"session {self.session_id}: Expected numpy array, got {type(chunk)}")
                chunk = np.zeros(int(WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_DURATION), dtype=WHISPER_DTYPE)

            if len((chunk.shape)) == 1:
                chunk = chunk.reshape(-1, 1)

            audio_data = (chunk * (2 ** 15 - 1)).astype(np.int16)

            frame = AudioFrame(format="s16", layout="mono", samples=audio_data.shape[0])
            frame.sample_rate = WEBRTC_SAMPLE_RATE
            frame.planes[0].update(audio_data.tobytes())

            if self.last_sync_item is None:
                self.last_sync_item = time.time()
                self.sample_sent = 0

            frame.pts = self.sample_sent
            frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
            self.sample_sent += audio_data.shape[0]

            expected_time = self.sample_sent / WEBRTC_SAMPLE_RATE
            current_time = time.time() - self.last_sync_item

            if expected_time > current_time:
                await asyncio.sleep(expected_time - current_time)
            return frame

        except Exception as e:
            logger.exception("%s error in recv()", self.session_id)
            silence_samples = int(WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_DURATION)
            silence_data = np.zeros(silence_samples, dtype=WHISPER_DTYPE)
            frame = AudioFrame(format="s16", layout="mono", samples=silence_samples)
            frame.sample_rate = WEBRTC_SAMPLE_RATE
            frame.pts = 0
            frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
            frame.planes[0].update(silence_data.tobytes())
            return frame

# ---------------- LLM callback endpoint ----------------
@app.post("/webrtc_server_callback")
async def webrtc_server_callback(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        llm_response_text = data.get("response", "No response")
        if session_id in pcs_by_id:
            tts_queue.put({"session_id": session_id, "text": llm_response_text})
        return {"status": "received"}
    except Exception as e:
        logger.exception("Callback error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- SDP / ICE helpers ----------------
def parse_sdp_candidate(sdp: str) -> dict:
    parts = sdp.split()
    foundation = parts[0].split(":")[1]
    cand = {
        "foundation": foundation,
        "component": int(parts[1]),
        "protocol": parts[2].lower(),
        "priority": int(parts[3]),
        "ip": parts[4],
        "port": int(parts[5]),
        "type": parts[7],
    }
    for i in range(8, len(parts) - 1, 2):
        if parts[i] == "raddr":
            cand["relatedAddress"] = parts[i + 1]
        elif parts[i] == "rport":
            cand["relatedPort"] = int(parts[i + 1])
        elif parts[i] == "tcptype":
            cand["tcpType"] = parts[i + 1]
    return cand

@app.post("/candidate/{session_id}")
async def candidate_from_browser(session_id: str, candidate_json: RTCIceCandidateJSON):
    pc = pcs_by_id.get(session_id)
    if not pc:
        raise HTTPException(status_code=404, detail="Session not found")
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
    return {"status": "ok"}

# ---------------- Offer / PeerConnection handling ----------------
@app.post("/offer")
async def offer(offer: Offer):
    session_id = str(uuid.uuid4())
    pc = RTCPeerConnection(
        configuration=RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"])
            ]
        )
    )
    pcs.add(pc)
    pcs_by_id[session_id] = pc
    audio_buffers[session_id] = bytearray()
    silent_chunk_counts[session_id] = 0
    server_ice_candidates[session_id] = []

    @pc.on("icecandidate")
    def on_icecandidate(event):
        if event.candidate:
            server_ice_candidates[session_id].append({
                "candidate": event.candidate.sdp,
                "sdpMid": event.candidate.sdpMid,
                "sdpMLineIndex": event.candidate.sdpMLineIndex
            })

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            asyncio.create_task(process_webrtc_audio(session_id, track))

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed"):
            logger.info(f"Server: Connection state changed to {pc.connectionState} for session {session_id}. Finalizing session.")
            # finalize and upload final file
            recording_manager.finalize_and_upload_on_close(session_id)

            # cleanup
            pcs.discard(pc)
            pcs_by_id.pop(session_id, None)
            pc_outbound_send_ttrack.pop(session_id, None)
            audio_output_queues.pop(session_id, None)
            audio_buffers.pop(session_id, None)
            silent_chunk_counts.pop(session_id, None)
            server_ice_candidates.pop(session_id, None)
            logger.info(f"Server: Session {session_id} closed and resources cleared.")

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
    pc.addTrack(WebRTCAudioSenderTrack(session_id))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    sdp_lines = pc.localDescription.sdp.split('\n')
    ice_candidate_in_sdp = [line for line in sdp_lines if line.startswith('a=candidate')]

    for candidate_line in ice_candidate_in_sdp:
        candidate_string = candidate_line[2:].replace('\r', '').strip()
        candidate_data = {
            "candidate": candidate_string,
            "sdpMid": "0",
            "sdpMLineIndex": 0,
            "usernameFragment": None
        }
        server_ice_candidates[session_id].append(candidate_data)
    await asyncio.sleep(0.2)
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": session_id,
        "server_ice_candidates": server_ice_candidates[session_id],
    }

# ---------------- Audio processing (incoming) ----------------
async def process_webrtc_audio(session_id: str, track: AudioStreamTrack):
    """
    Receives frames from the browser. For each 20ms frame:
      - compute RMS
      - if speech: convert to 16k int16 and write to current wav (RecordingManager)
      - if silence: increment silent counter and if >= threshold, ask RecordingManager to rotate/upload
    """
    try:
        # ensure a file exists for this session
        recording_manager.ensure_session(session_id)

        while True:
            frame = await track.recv()  # av.AudioFrame
            raw = frame.to_ndarray()
            # If stereo, collapse to mono
            if raw.ndim > 1:
                raw = raw.mean(axis=0)
            # Normalize to float32 in [-1,1]
            samples = raw.astype(np.float32) / 32768.0
            # compute RMS
            rms = float(np.sqrt(np.mean(samples ** 2)))
            logger.debug("session %s: rms=%.6f", session_id, rms)

            if rms >= SILENCE_THRESHOLD_RMS:
                # speech
                recording_manager.mark_speech(session_id)
                # convert to 16k int16 and write
                if frame.sample_rate == WEBRTC_SAMPLE_RATE:
                    audio_int16 = convert_48khz_audio_to_16khz(samples)
                else:
                    # frame at some other rate (e.g., already 16k)
                    if samples.dtype in (np.float32, np.float64):
                        audio_int16 = float_samples_to_int16(samples)
                    else:
                        audio_int16 = samples.astype(np.int16)
                recording_manager.write_samples(session_id, audio_int16)
            else:
                # silence
                rotated = recording_manager.mark_silence_and_maybe_rotate(session_id)
                if rotated:
                    # if rotated, silent counter already reset inside manager
                    logger.info("session %s: rotated file due to prolonged silence", session_id)
                else:
                    # no rotation: continue (maybe file was <2s)
                    pass

    except Exception:
        logger.exception("Audio processing error for session %s", session_id)
    finally:
        # When track ends: finalize
        try:
            logger.info("process_webrtc_audio: track ended for %s, finalizing.", session_id)
            recording_manager.finalize_and_upload_on_close(session_id)
        except Exception:
            logger.exception("Error finalizing on track end for %s", session_id)

# ---------------- Shutdown ----------------
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutdown event: closing peer connections and TTS worker.")
    for pc in list(pcs):
        try:
            await pc.close()
        except Exception:
            logger.exception("Error closing pc")
    # shutdown tts worker
    tts_queue.put(None)
    tts_worker_thread.join(timeout=5)

# ---------------- Run server ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WEBRTC_SERVER_PORT)
