import time
import uuid
from pathlib import Path
from typing import Dict, Optional
import threading
import queue

import numpy as np
import uvicorn
import requests
from fastapi import FastAPI, UploadFile, Form
from faster_whisper import WhisperModel
from gpt4all import GPT4All

############################
# Config
############################
WHISPER_MODEL = "large-v2"  # Recommended for better accuracy
GPT4ALL_MODEL = "orca-mini-3b-gguf2-q4_0.gguf"

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
stt_request_queue: "queue.Queue[dict]" = queue.Queue()   # STT (Whisper) queue
llm_request_queue: "queue.Queue[dict]" = queue.Queue()   # LLM (GPT4All) queue


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
"I don’t have that information."
Keep answers short and clear.

Dataset:
{dataset_str}

Given the dataset above, answer the user's question strictly using only that information.
"""



############################
# Worker threads
############################
def stt_worker():
    """Dedicated worker to run Whisper transcription off the FastAPI loop."""
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
            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            segments, _ = whisper_model.transcribe(
                audio_np,
                word_timestamps=True,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            transcription = " ".join(s.text.strip() for s in segments)
            print(f"Server: STT worker - Session {session_id} - Transcription: {transcription}")
        except Exception as e:
            print(f"Server: STT worker - Error during Whisper transcription: {e}")
            transcription = ""

        if not transcription.strip():
            # Send immediate callback with "didn't catch that"
            payload = {
                "final_transcription": "",
                "response": "Sorry, I didn’t catch that.",
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
        print(f"Server: LLM worker - Session {session_id} prompt-{full_system_prompt} - Generating response for: '{transcription}'")
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


############################
# API Endpoints
############################
@app.post("/stream_audio")
async def stream_audio(
    session_id: str = Form(...),
    chunk: UploadFile = Form(...),
):
    """
    Accept incoming audio chunks and buffer them without blocking.
    """
    session = get_session(session_id)
    data = await chunk.read()

    # Lock to avoid race with commit that snapshots & clears buffer
    with session["lock"]:
        session["audio_buffer"].extend(data)
        session["last_chunk_time"] = time.time()

    print(f"Server: Session {session_id} - Audio buffer size: {len(session['audio_buffer'])} bytes")
    return {"message": "Listening...", "session_id": session_id}


@app.post("/commit_audio")
async def commit_audio(session_id: str = Form(...), callback_url: str = Form(...)):
    """
    Snapshot the buffered audio and queue it for STT in a background thread.
    Return immediately so audio streaming isn't blocked.
    """
    session = get_session(session_id)

    # Snapshot & clear buffer atomically
    with session["lock"]:
        if len(session["audio_buffer"]) == 0:
            # Nothing to transcribe; send immediate callback
            payload = {
                "final_transcription": "",
                "response": "Sorry, I didn’t catch that.",
                "session_id": session_id
            }
            try:
                requests.post(callback_url, json=payload, timeout=5)
                print(f"Server: commit_audio - No audio, sent default response to {callback_url}")
            except requests.exceptions.RequestException as e:
                print(f"Server: commit_audio - Callback error: {e}")
            return {"message": "No audio to process.", "session_id": session_id}

        audio_snapshot = bytes(session["audio_buffer"])
        session["audio_buffer"].clear()
        session["transcription_buffer"] = ""

    # Queue STT work
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
    """
    Kept for compatibility; using hardcoded global context instead.
    """
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
