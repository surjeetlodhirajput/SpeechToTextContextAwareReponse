# webrtc_server.py

import asyncio
import logging
import os
import string
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
from fractions import Fraction
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
tts_engines: Dict[str, pyttsx3.Engine] = {}
tts_queue = queue.Queue()
audio_output_queues: Dict[str, queue.Queue] = {}
pc_outbound_send_ttrack: Dict[str, "OutboundAudioTrack"] = {}
audio_buffers: Dict[str, bytearray] = {}
silent_chunk_counts: Dict[str, int] = {}
server_ice_candidates: Dict[str, List["RTCIceCandidateJSON"]] = {}
current_wav_file: Dict[str, wave.Wave_write] = {}
# Audio config
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1
WHISPER_DTYPE = np.float32

WEBRTC_SAMPLE_RATE = 48000
WEBRTC_CHANNELS = 1
WEBRTC_FRAME_DURATION = 0.02 #20 ms frame duration

WEBRTC_AUDIO_CHUNK_SIZE = WHISPER_SAMPLE_RATE // 2
SILENCE_THRESHOLD_RMS = 0.015
CONSECUTIVE_SILENT_CHUNKS = 3
FRAME_MS = 20
FRAME_SIZE = int(WHISPER_SAMPLE_RATE * (FRAME_MS / 1000.0))


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

def create_tts_engine():
    """create the tts engine instance"""
    engine = pyttsx3.init()
    engine.setProperty("rate",180)
    return engine
# --- pyttsx3 TTS Worker (thread) ---
def tts_engine_worker():
    logger.info("TTS Engine worker started.")
    engine = None
    while True:
        logger.info("engine started for the next teask")
        task = tts_queue.get()
        logger.info(f"tts engine Get Task {task}")

        session_id = task["session_id"]
        text = task["text"]
        temp_file_name = f"tts_uuid_{uuid.uuid4()}.wav"
        try:
            logger.info(f"TTS Engined creating fresg engine for the session {session_id}")
            if engine:
                try:
                    engine.stop()
                except:
                    pass
            engine = create_tts_engine()
            engine.save_to_file(text, temp_file_name)
            import threading
            def run_tts():
                engine.runAndWait()
            tts_thread = threading.Thread(target = run_tts)
            tts_thread.daemon = True
            tts_thread.start()
            tts_thread.join(timeout=10)#10second time out

            if tts_thread.is_alive():
                try:
                    engine.stop()
                except:
                    pass
                continue
            if not os.path.exists(temp_file_name):
                continue
            with wave.open(temp_file_name,'rb') as wf:
                orig_rate = wf.getframerate()
                channels = wf.getnchannels()
                nframes = wf.getnframes()

                logger.info(f"TTS: Worket Audio file state:- Rate {orig_rate}, Channels: {channels} Frame: {nframes}")
                pcm_data = wf.readframes(nframes)
                audio = np.frombuffer(pcm_data, dtype = np.int16)
                if channels == 2:
                    audio = audio.reshape(-1,2).mean(axis=1) # convert stereo to mono
                if orig_rate != WEBRTC_SAMPLE_RATE:
                    num_samples = int(len(audio)* WEBRTC_SAMPLE_RATE/orig_rate)
                    audio = resample(audio, num_samples)
                audio = audio.astype(np.float32) / (2**15)

                chunk_size = int(WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_DURATION) #20 ms chunks
                chunk_sent = 0  
                logger.info(f"TTS: Worker: Starting to streo {len(audio)} samples in chunks of {chunk_size}")

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    if session_id in audio_output_queues:
                        audio_output_queues[session_id].put(chunk)
                        chunk_sent+=1
                        print("chunk pushed into the array")
                    time.sleep(WEBRTC_FRAME_DURATION)
                
                if session_id in audio_output_queues:
                    logger.info(f"Finished for streaming session id{session_id}")

            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
        except Exception as e:
            logger.error(f"error in the file conversion {e}")
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
        finally:
            tts_queue.task_done()
            if engine:
                try:
                    engine.stop()
                except:
                    pass
            engine = None


tts_worker_thread = threading.Thread(target=tts_engine_worker, daemon=True)
tts_worker_thread.start()


# --- Utility: PCM normalization ---
def _ensure_mono_int16_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    if samples.dtype in (np.float32, np.float64):
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32768.0).astype(np.int16)
    elif samples.dtype != np.int16:
        samples = samples.astype(np.int16)
    if sr != WHISPER_SAMPLE_RATE:
        num_samples = int(round(len(samples) * WHISPER_SAMPLE_RATE / sr))
        f = samples.astype(np.float32)
        x_old = np.linspace(0, 1, len(f), endpoint=False)
        x_new = np.linspace(0, 1, num_samples, endpoint=False)
        samples = np.interp(x_new, x_old, f).astype(np.int16)
    return samples


def text_to_pcm_array(text: str) -> np.ndarray:
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    data, sr = sf.read(buf, always_2d=False)
    return _ensure_mono_int16_16k(data, sr)


# --- Outbound track ---
class WebRTCAudioSenderTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, session_id: string):
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
                chunk = await asyncio.wait_for(asyncio.to_thread(self.queue.get, timeout = 0.1),
                timeout = 0.1)
            except(asyncio.TimeoutError, queue.Empty):
                silence_samples = int(WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_DURATION)
                chunk = np.zeros(silence_samples, dtype = WHISPER_DTYPE)
            
            if chunk is None:
                raise EOFError("End of audio stream")
            
            if not isinstance(chunk, np.ndarray):
                logger.error(f"session {self.session_id}: Expected numpy error , got {type(chunk)}")
                chunk = np.zeros(int(WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_DURATION), dtype=WHISPER_DTYPE)

            if len((chunk.shape)) == 1:
                chunk = chunk.reshape(-1, 1)

            audio_data =  (chunk * (2 ** 15 -1)).astype(np.int16)

            frame = AudioFrame(format = "s16", layout = "mono", samples = audio_data.shape[0])
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
            logger.error(f"{self.session_id} error in recv() {e}")
            silence_samples = int(WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_DURATION)
            silence_data = np.zeros(silence_samples, dtype = WHISPER_DTYPE)
            frame = AudioFrame(format = "s16", layout = "mono", samples = silence_samples)
            frame.sample_rate = WEBRTC_SAMPLE_RATE
            frame.pts = 0
            frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
            frame.planes[0].update(silence_data.tobytes())

            return frame



# --- LLM Callback ---
@app.post("/webrtc_server_callback")
async def webrtc_server_callback(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        llm_response_text = data.get("response", "No response")
        final_transcription = data.get("final_transcription", "")
        if session_id in pcs_by_id:
            tts_queue.put({"session_id": session_id, "text": llm_response_text})
        return {"status": "received"}
    except Exception as e:
        logger.error(f"Callback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- SDP / ICE ---
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


# --- Offer ---
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
            pcs.discard(pc)
            pcs_by_id.pop(session_id, None)
            pc_outbound_send_ttrack.pop(session_id, None)
            tts_engines.pop(session_id, None)
            audio_output_queues.pop(session_id, None)
            audio_buffers.pop(session_id, None)
            current_wav_file[session_id].close()
            del current_wav_file[session_id]
            silent_chunk_counts.pop(session_id, None)
            logger.info(f"Server: Session {session_id} closed and resources cleared.")

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
    pc.addTrack(WebRTCAudioSenderTrack(session_id))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    sdp_lines = pc.localDescription.sdp.split('\n');
    ice_candidate_in_sdp = [line for line in sdp_lines if line.startswith('a=candidate')]

    for candidate_line in ice_candidate_in_sdp:
        candidate_string = candidate_line[2:].replace('\r','').strip()
        candidate_data = {
            "candidate": candidate_string,
            "sdpMid": "0",
            "sdpMLineIndex": 0,
            "usernameFragment": None
        }
        server_ice_candidates[session_id].append(candidate_data)
    await asyncio.sleep(0.5)
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": session_id,
        "server_ice_candidates": server_ice_candidates[session_id],
    }

def convert_48khz_audio_to_16khz(samples_48k: np.ndarray) -> np.ndarray:
    if samples_48k.ndim > 1:
        samples_48k = samples_48k.mean(axis=-1)
    samples_16k_float = samples_48k[::3]

    #convert to int16
    samples_16k_float = np.clip(samples_16k_float, -1.0, 1.0)
    samples_16k_int16 = (samples_16k_float * 32767.0).astype(np.int16)
    return samples_16k_int16

def create_new_wav_file(session_id)->str:
    recordin_dir = "recordings"
    if not os.path.exists(recordin_dir):
        os.makedirs(recordin_dir)
    filename = f"{recordin_dir}/rc_{session_id}.wav"
    try:
        if session_id in current_wav_file:
            current_wav_file[session_id].close()
        wf = wave.open(filename,'wb')
        wf.setnchannels(WHISPER_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(WHISPER_SAMPLE_RATE)
        current_wav_file[session_id] = wf
        logger.info(f"Created new wav file for {session_id}")
        return filename
    except Exception as e:
        logger.error(f"Error creating new wav file for {session_id}: {e}")
        return None

def write_audio_to_current_file(session_id: str, audio_samples:np.ndarray, original_sample_rate: int = WEBRTC_SAMPLE_RATE):
    if session_id not in current_wav_file:
        return
    try:
        if audio_samples.dtype in (np.float32, np.float64):
            audio_samples = np.clip(audio_samples, -1.0, 1.0)
            audio_16k = (audio_samples * 32767.0).astype(np.int16)
        else:
            audio_16k = audio_samples.astype(np.int16)
        
        # Ensure audio is 16kHz before writing
        if original_sample_rate != WHISPER_SAMPLE_RATE:
            audio_16k = resample(audio_16k, int(len(audio_16k) * WHISPER_SAMPLE_RATE / original_sample_rate)).astype(np.int16)

        current_wav_file[session_id].writeframes(audio_16k.tobytes())
    except Exception as e:
        logger.error(f"write audio file error {session_id}: {e}")

# --- Audio Processing (incoming) ---
async def process_webrtc_audio(session_id: str, track: AudioStreamTrack):
    try:
        # Create new wav file for this session at the start
        create_new_wav_file(session_id)

        while True:
            frame = await track.recv()
            raw = frame.to_ndarray()
            if raw.ndim > 1:
                raw = raw.mean(axis=0)
            samples = raw.astype(np.float32) / 32768.0

            rms = float(np.sqrt(np.mean(samples ** 2)))
            if rms >= SILENCE_THRESHOLD_RMS:
                if frame.sample_rate == WEBRTC_SAMPLE_RATE:
                    audio_16k = convert_48khz_audio_to_16khz(samples)
                else:
                    if samples.dtype in (np.float32, np.float64):
                        samples = np.clip(samples, -1.0, 1.0)
                        audio_16k = (samples * 32767.0).astype(np.int16)
                    else:
                        audio_16k = samples.astype(np.int16)

                # ✅ Append audio to the current wav file
                write_audio_to_current_file(session_id, audio_16k, frame.sample_rate)

                # Send chunk to main server
                files = {'chunk': ("audio.wav", audio_16k.tobytes(), "audio/wav")}
                data = {'session_id': session_id}
                try:
                    # response = requests.post(
                    #     f"{MAIN_FASTAPI_SERVER_URL}/stream_audio",
                    #     files=files,
                    #     data=data,
                    #     timeout=3,
                    # )
                    print(f"Server: Sent processed chunk for session {session_id}. Response: {''}")
                except requests.exceptions.ConnectionError as e:
                    print(f"Server: Could not connect to /stream_audio endpoint: {e}")

            else:
                silent_chunk_counts[session_id] += 1
                if silent_chunk_counts[session_id] >= CONSECUTIVE_SILENT_CHUNKS:
                    silent_chunk_counts[session_id] = 0
                    data = {
                        "session_id": session_id,
                        "callback_url": f"http://localhost:{WEBRTC_SERVER_PORT}/webrtc_server_callback",
                    }
                    # await asyncio.to_thread(
                    #     requests.post,
                    #     f"{MAIN_FASTAPI_SERVER_URL}/commit_audio",
                    #     data=data,
                    #     timeout=3,
                    # )
    except Exception as e:
        logger.error(f"Audio error: {e}")
    finally:
        # ✅ Close wav file when track ends
        if session_id in current_wav_file:
            try:
                current_wav_file[session_id].close()
                logger.info(f"WAV file closed for session {session_id}")
            except Exception as e:
                logger.error(f"Error closing WAV file for {session_id}: {e}")
            del current_wav_file[session_id]

# --- Shutdown ---
@app.on_event("shutdown")
async def shutdown_event():
    for pc in pcs:
        await pc.close()
    tts_queue.put(None)
    tts_worker_thread.join(timeout=5)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=WEBRTC_SERVER_PORT)
