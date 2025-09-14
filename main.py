import time
import uuid
from pathlib import Path
from typing import Dict, Optional
import threading
import queue
import wave
import io

import numpy as np
import uvicorn
import requests
from fastapi import FastAPI, UploadFile, Form
from faster_whisper import WhisperModel
from gpt4all import GPT4All
import soundfile as sf
from scipy.signal import resample

############################
# Config
############################
WHISPER_MODEL = "large-v2"
GPT4ALL_MODEL = "orca-mini-3b-gguf2-q4_0.gguf"

# Target audio format for Whisper
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# Hardcoded global TRAVEL_DATA
TRAVEL_DATA = [
    {"transactionId": "TX123", "travelDate": "2025-09-01", "destination": "Paris", "city": "Paris"},
    {"transactionId": "TX456", "travelDate": "2025-10-15", "destination": "New York", "city": "New York"},
    {"transactionId": "TX789", "travelDate": "2025-11-20", "destination": "Tokyo", "city": "Tokyo"},
    {"transactionId": "TX001", "travelDate": "2025-12-25", "destination": "London", "city": "London",
     "totalCost": "1500", "currency": "GBP", "status": "Confirmed", "Checked In Baggage Weight Limit": "20kg",
     "E-Ticket Number": "0987654321", "Passenger Name": "Jane Doe", "Flight Number": "BA456",
     "Departure Date": "2025-12-25", "Departure Time": "08:00 AM", "Arrival Date": "2025-12-25",
     "Arrival Time": "10:00 AM", "Airline": "British Airways", "Flight Status": "On Time",
     "Gate Number": "50", "Seat Number": "22C", "Baggage Tag Number": "0987654321"}
]

############################
# Globals
############################
app = FastAPI()

print("Server: Loading models...")
whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

gpt4all_model = GPT4All(
    model_name=GPT4ALL_MODEL,
    model_path=Path.home() / "Documents/GPT4All/",
    allow_download=True
)
print("Server: Models loaded.")

# Sessions hold audio buffers and locks
sessions: Dict[str, dict] = {}

# Worker queues
stt_request_queue: "queue.Queue[dict]" = queue.Queue()
llm_request_queue: "queue.Queue[dict]" = queue.Queue()

############################
# FIXED: Audio Processing Helpers with proper type conversion
############################
def convert_numpy_types(obj):
    """
    FIXED: Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def validate_and_convert_audio(audio_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    FIXED: Properly validate and convert WAV files with JSON-serializable output
    """
    try:
        audio_info = {"original_format": "unknown", "channels": 0, "sample_rate": 0, "duration": 0.0}
        
        try:
            # Method 1: Parse WAV file using wave module (handles int16 PCM correctly)
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                
                audio_info.update({
                    "original_format": "WAV",
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "duration": frames / sample_rate if sample_rate > 0 else 0.0,
                    "sample_width": sampwidth
                })
                
                # Read PCM data
                pcm_data = wf.readframes(frames)
                
                # FIXED: Convert based on actual sample width
                if sampwidth == 1:  # 8-bit unsigned
                    audio_np = np.frombuffer(pcm_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                elif sampwidth == 2:  # 16-bit signed (most common)
                    audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:  # 32-bit signed or float
                    try:
                        # Try as float32 first
                        audio_np = np.frombuffer(pcm_data, dtype=np.float32)
                        if np.max(np.abs(audio_np)) > 10:  # Likely int32, not float32
                            audio_np = np.frombuffer(pcm_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                    except:
                        audio_np = np.frombuffer(pcm_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")
                    
        except wave.Error:
            # Method 2: Try with soundfile for other formats
            try:
                audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
                
                audio_info.update({
                    "original_format": "Other",
                    "channels": 1 if audio_np.ndim == 1 else audio_np.shape[1],
                    "sample_rate": sample_rate,
                    "duration": len(audio_np) / sample_rate if sample_rate > 0 else 0.0
                })
                
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)
                    
            except Exception as sf_error:
                raise ValueError(f"Failed to decode audio: {sf_error}")
        
        # Convert stereo to mono if needed
        if audio_np.ndim > 1:
            if audio_np.shape[1] == 2:  # Stereo
                audio_np = np.mean(audio_np, axis=1)
            else:  # Multi-channel
                audio_np = audio_np[:, 0]  # Take first channel
        
        # FIXED: Proper resampling to target sample rate
        if audio_info["sample_rate"] != TARGET_SAMPLE_RATE:
            target_length = int(len(audio_np) * TARGET_SAMPLE_RATE / audio_info["sample_rate"])
            audio_np = resample(audio_np, target_length)
            print(f"Resampled audio from {audio_info['sample_rate']}Hz to {TARGET_SAMPLE_RATE}Hz")
        
        # Ensure audio is in correct range [-1.0, 1.0]
        audio_np = np.clip(audio_np, -1.0, 1.0)
        
        # FIXED: Check for silent audio and convert to native Python float
        rms = np.sqrt(np.mean(audio_np ** 2))
        audio_info["rms_level"] = float(rms)  # Convert numpy.float32 to Python float
        
        if audio_info["rms_level"] < 0.001:
            print(f"Warning: Audio RMS level is very low ({audio_info['rms_level']:.6f}). Audio might be silent.")
        
        # FIXED: Convert all numpy types in audio_info to Python types
        audio_info = convert_numpy_types(audio_info)
        
        return audio_np, audio_info
        
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {e}")

############################
# Helpers
############################
def get_session(session_id: str):
    """Retrieve or create a session with a per-session lock."""
    if session_id not in sessions:
        sessions[session_id] = {
            "audio_buffer": bytearray(),
            "chat_history": [],
            "last_response": "",
            "last_chunk_time": time.time(),
            "transcription_buffer": "",
            "lock": threading.Lock(),
        }
    return sessions[session_id]

def build_system_prompt(travel_data) -> str:
    """Builds the full system prompt with formatted travel dataset."""
    
    def format_entry(entry: dict, idx: int) -> str:
        return f"  - Travel Option {idx}: " + ", ".join(f"{k}: {v}" for k, v in entry.items())

    if isinstance(travel_data, list):
        dataset_str = "\n".join(format_entry(item, i+1) for i, item in enumerate(travel_data))
    elif isinstance(travel_data, dict):
        dataset_str = format_entry(travel_data, 1)
    else:
        dataset_str = str(travel_data)

    return f"""
You are a customer support assistant.
You must ONLY use the travel dataset below to answer.
If the user asks something outside the dataset, reply with:
"I don't have that information."
Keep answers short and clear.

Dataset:
{dataset_str}

Given the dataset above, answer the user's question strictly using only that information.
"""

############################
# FIXED Worker threads
############################
def stt_worker():
    """FIXED: Dedicated worker with proper audio format handling for Whisper."""
    print("Server: STT worker thread started.")
    while True:
        task = stt_request_queue.get()
        if task is None:
            print("Server: STT worker received exit signal.")
            break

        session_id: str = task["session_id"]
        audio_bytes: bytes = task["audio_bytes"]
        callback_url: Optional[str] = task["callback_url"]

        transcription = ""
        try:
            print(f"Server: STT worker - Session {session_id} - Processing {len(audio_bytes)} bytes of audio")
            
            # FIXED: Proper audio conversion
            audio_np, audio_info = validate_and_convert_audio(audio_bytes)
            
            print(f"Server: STT worker - Audio info: {audio_info}")
            
            # Check if audio has meaningful content
            if audio_info["duration"] < 0.1:
                print(f"Server: STT worker - Audio too short ({audio_info['duration']:.2f}s)")
                transcription = ""
            elif audio_info["rms_level"] < 0.001:
                print(f"Server: STT worker - Audio too quiet (RMS: {audio_info['rms_level']:.6f})")
                transcription = ""
            else:
                # Run Whisper transcription with optimized parameters
                print(f"Server: STT worker - Running Whisper on {len(audio_np)} samples")
                segments, info = whisper_model.transcribe(
                    audio_np,
                    word_timestamps=True,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        min_speech_duration_ms=250
                    ),
                    beam_size=5,
                    temperature=0.0
                )
                
                transcription = " ".join(s.text.strip() for s in segments)
                print(f"Server: STT worker - Session {session_id} - Transcription: '{transcription}'")
                print(f"Server: STT worker - Language: {info.language} (confidence: {info.language_probability:.2f})")
                
        except Exception as e:
            print(f"Server: STT worker - Error during Whisper transcription: {e}")
            import traceback
            traceback.print_exc()
            transcription = ""

        if not transcription.strip():
            # Send immediate callback with "didn't catch that"
            payload = {
                "final_transcription": "",
                "response": "Sorry, I didn't catch that.",
                "session_id": session_id,
            }
            if callback_url:
                try:
                    requests.post(callback_url, json=payload, timeout=5)
                    print(f"Server: STT worker - Sent empty transcription response to {callback_url}")
                except requests.exceptions.RequestException as e:
                    print(f"Server: STT worker - Callback error: {e}")
            stt_request_queue.task_done()
            continue

        # Pass to LLM queue
        llm_request_queue.put({
            "session_id": session_id,
            "transcription": transcription,
            "callback_url": callback_url
        })
        stt_request_queue.task_done()

def llm_worker():
    """Dedicated worker to process LLM requests (single-threaded for GPT4All safety)."""
    print("Server: LLM worker thread started.")

    while True:
        task = llm_request_queue.get()
        if task is None:
            print("Server: LLM worker thread received exit signal.")
            break

        session_id = task["session_id"]
        transcription = task["transcription"]
        callback_url = task["callback_url"]

        session = sessions.get(session_id)
        if not session:
            response_text = f"Server: LLM worker - Session {session_id} not found."
            print(response_text)
            payload = {
                "final_transcription": transcription,
                "response": response_text,
                "session_id": session_id
            }
            if callback_url:
                try:
                    requests.post(callback_url, json=payload, timeout=5)
                except requests.exceptions.RequestException as e:
                    print(f"Server: LLM worker - Failed to send error callback: {e}")
            llm_request_queue.task_done()
            continue
            
        full_system_prompt = build_system_prompt(TRAVEL_DATA)
        print(f"Server: LLM worker - Session {session_id} - Generating response for: '{transcription}'")
        
        try:
            with gpt4all_model.chat_session(full_system_prompt):
                response_text = gpt4all_model.generate(
                    prompt=transcription,
                    max_tokens=200,
                    temp=0.2
                )
            print(f"Server: LLM worker - Session {session_id} - Response: '{response_text}'")
        except Exception as e:
            response_text = f"Server: LLM worker - Error during GPT4All generation: {e}"
            print(response_text)

        # Save to chat history
        try:
            session["chat_history"].append({"role": "user", "content": transcription})
            session["chat_history"].append({"role": "assistant", "content": response_text})
            session["last_response"] = response_text
        except Exception as e:
            print(f"Server: LLM worker - failed to update session history: {e}")

        # Send callback
        payload = {
            "final_transcription": transcription,
            "response": response_text,
            "session_id": session_id
        }
        if callback_url:
            try:
                requests.post(callback_url, json=payload, timeout=5)
                print(f"Server: LLM worker - Sent response to {callback_url}")
            except requests.exceptions.RequestException as e:
                print(f"Server: LLM worker - Callback error: {e}")

        llm_request_queue.task_done()

# endpoint to handle the file upload
from fastapi import File

@app.post("/upload_wav")
async def upload_wav(
    session_id: str = Form(...),
    callback_url: str = Form(...),
    file: UploadFile = File(...)
):
    """
    FIXED: Accept a WAV file with proper validation and JSON-serializable responses
    """
    session = get_session(session_id)
    audio_bytes = await file.read()

    print(f"Server: upload_wav - Session {session_id} - Received file: {file.filename} ({len(audio_bytes)} bytes)")

    if not audio_bytes:
        payload = {
            "final_transcription": "",
            "response": "Sorry, I didn't catch that (empty WAV).",
            "session_id": session_id,
        }
        try:
            requests.post(callback_url, json=payload, timeout=5)
            print(f"Server: upload_wav - Empty file, sent default response to {callback_url}")
        except requests.exceptions.RequestException as e:
            print(f"Server: upload_wav - Callback error: {e}")
        return {"message": "Uploaded file was empty.", "session_id": session_id}

    # ADDED: Quick validation before queuing
    try:
        audio_np, audio_info = validate_and_convert_audio(audio_bytes)
        print(f"Server: upload_wav - Audio validation successful: {audio_info}")
        
        if audio_info["duration"] < 0.1:
            payload = {
                "final_transcription": "",
                "response": "Sorry, the audio was too short.",
                "session_id": session_id,
            }
            try:
                requests.post(callback_url, json=payload, timeout=5)
            except requests.exceptions.RequestException as e:
                print(f"Server: upload_wav - Callback error: {e}")
            return {"message": "Audio too short.", "session_id": session_id}
            
    except Exception as e:
        print(f"Server: upload_wav - Audio validation failed: {e}")
        payload = {
            "final_transcription": "",
            "response": "Sorry, there was an issue with the audio format.",
            "session_id": session_id,
        }
        try:
            requests.post(callback_url, json=payload, timeout=5)
        except requests.exceptions.RequestException as e:
            print(f"Server: upload_wav - Callback error: {e}")
        return {"message": "Audio validation failed.", "session_id": session_id}

    # Queue it for STT → LLM pipeline
    stt_request_queue.put({
        "session_id": session_id,
        "audio_bytes": audio_bytes,
        "callback_url": callback_url
    })
    print(f"Server: upload_wav - Session {session_id} - Queued WAV for STT processing.")

    # FIXED: Return JSON-serializable response (audio_info already converted)
    return {
        "message": "WAV file accepted and validated. Transcription + response will be sent to callback URL.",
        "session_id": session_id,
        "audio_info": audio_info  # Now contains only Python native types
    }

# Rest of the endpoints remain the same...
@app.post("/stream_audio")
async def stream_audio(
    session_id: str = Form(...),
    chunk: UploadFile = Form(...),
):
    """Accept incoming audio chunks and buffer them without blocking."""
    session = get_session(session_id)
    data = await chunk.read()

    with session["lock"]:
        session["audio_buffer"].extend(data)
        session["last_chunk_time"] = time.time()

    print(f"Server: Session {session_id} - Audio buffer size: {len(session['audio_buffer'])} bytes")
    return {"message": "Listening...", "session_id": session_id}

@app.post("/commit_audio")
async def commit_audio(session_id: str = Form(...), callback_url: str = Form(...)):
    """Snapshot the buffered audio and queue it for STT in a background thread."""
    session = get_session(session_id)

    with session["lock"]:
        if len(session["audio_buffer"]) == 0:
            print(f"Server: commit_audio - No audio, skipping for {session_id}")
            return {"message": "No audio to process.", "session_id": session_id}

        audio_snapshot = bytes(session["audio_buffer"])
        session["audio_buffer"].clear()
        session["transcription_buffer"] = ""

    stt_request_queue.put({
        "session_id": session_id,
        "audio_bytes": audio_snapshot,
        "callback_url": callback_url
    })
    print(f"Server: commit_audio - Session {session_id} - Queued audio ({len(audio_snapshot)} bytes) for STT.")

    return {"message": "Audio committed. STT and LLM will run in background; response will be sent to your callback.", "session_id": session_id}

@app.post("/start_session")
def start_session():
    """Initialize a new session and return a unique session ID."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "audio_buffer": bytearray(),
        "chat_history": [],
        "last_response": "",
        "last_chunk_time": time.time(),
        "transcription_buffer": "",
        "lock": threading.Lock(),
    }
    print(f"Server: New session started: {session_id}")
    return {"session_id": session_id}

@app.post("/set_session_context")
def set_session_context(
    session_id: str = Form(...),
    system_prompt: str = Form(...),
    travel_data: str = Form(...)
):
    """Kept for compatibility; using hardcoded global context instead."""
    print(f"Server: Received context for session {session_id}; using global context.")
    return {"message": "Session context updated successfully.", "session_id": session_id}

@app.post("/end_session")
def end_session(session_id: str = Form(...)):
    """End and clear a specific session from memory."""
    if session_id in sessions:
        del sessions[session_id]
        print(f"Server: Session {session_id} ended.")
        return {"message": "Session ended", "session_id": session_id}
    return {"message": "Session not found", "session_id": session_id}

############################
# Run
############################
if __name__ == "__main__":
    # Start worker threads before the server
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()
    print("Server: Started STT worker thread.")

    llm_thread = threading.Thread(target=llm_worker, daemon=True)
    llm_thread.start()
    print("Server: Started LLM worker thread.")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        # Graceful shutdown of workers
        stt_request_queue.put(None)
        llm_request_queue.put(None)
        stt_thread.join()
        llm_thread.join()
        print("Server: Worker threads gracefully shut down.")
