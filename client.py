import requests
import sounddevice as sd
import numpy as np
import io
import pyttsx3
import time
import threading
import queue
import uvicorn
from fastapi import FastAPI, Request
import json

# --- Main Server Configuration ---
SERVER_URL = "http://localhost:8000"

# --- Client's own server configuration ---
CLIENT_HOST = "127.0.0.1"
CLIENT_PORT = 8001
CLIENT_CALLBACK_URL = f"http://{CLIENT_HOST}:{CLIENT_PORT}/receive_response"

# --- Silence Detection ---
SILENCE_THRESHOLD_RMS = 0.015
CONSECUTIVE_SILENT_CHUNKS = 3  # 2 sec of silence (4 × 0.5s chunks)

# --- Queues ---
audio_queue = queue.Queue()
response_queue = queue.Queue()
tts_queue = queue.Queue()

# --- Stop signal shared across threads ---
stop_event = threading.Event()

# --- Client's FastAPI App ---
client_app = FastAPI()

############################
# TEST DATA
############################
TEST_TRAVEL_DATA = [
    {"transactionId": "TX123", "travelDate": "2025-09-01", "destination": "Paris", "city": "Paris"},
    {"transactionId": "TX456", "travelDate": "2025-10-15", "destination": "New York", "city": "New York"},
    {"transactionId": "TX789", "travelDate": "2025-11-20", "destination": "Tokyo", "city": "Tokyo"},
]

TEST_SYSTEM_PROMPT = (
    "You are a travel assistant for a customer using the provided travel dataset. "
    "You must only answer using the travel dataset. "
    "If a user asks something outside the dataset, reply with 'Sorry, I don’t have that information.' "
    "Keep answers short and clear. If user asks to end, say 'Goodbye!'."
)

############################
# Client Endpoint (Callback)
############################
@client_app.post("/receive_response")
async def receive_response_from_server(request: Request):
    """Main server calls this endpoint with its response."""
    try:
        response_data = await request.json()
        print(f"\nClient: Received response from main server: {response_data.get('response', 'No text')}")
        response_queue.put(response_data)  # Non-blocking handoff to the sender loop
        return {"status": "received"}
    except Exception as e:
        print(f"Client: Error receiving response: {e}")
        return {"status": "error", "message": str(e)}

############################
# Helpers
############################
def start_session():
    try:
        resp = requests.post(f"{SERVER_URL}/start_session", timeout=5)
        resp.raise_for_status()
        return resp.json()["session_id"]
    except Exception as e:
        print(f"Client: Error starting session: {e}")
        return None

def set_session_context(session_id: str, system_prompt: str, travel_data: list):
    try:
        travel_data_json = json.dumps(travel_data)
        data = {"session_id": session_id, "system_prompt": system_prompt, "travel_data": travel_data_json}
        resp = requests.post(f"{SERVER_URL}/set_session_context", data=data, timeout=5)
        resp.raise_for_status()
        print(f"Client: Context set -> {resp.json().get('message')}")
        return resp.json()
    except Exception as e:
        print(f"Client: Error setting context: {e}")
        return {"message": "Error setting context"}

def end_session(session_id):
    try:
        resp = requests.post(f"{SERVER_URL}/end_session", data={"session_id": session_id}, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Client: Error ending session: {e}")
        return {"message": "Error ending session"}

def send_chunk(session_id, audio_chunk: np.ndarray):
    # Convert to bytes (float32 PCM)
    buffer = io.BytesIO(audio_chunk.astype(np.float32).tobytes())
    files = {"chunk": ("chunk.raw", buffer, "application/octet-stream")}
    data = {"session_id": session_id}
    try:
        resp = requests.post(f"{SERVER_URL}/stream_audio", files=files, data=data, timeout=4)
        resp.raise_for_status()
        print("Client: Sent audio chunk to server")
        return resp.json()
    except requests.exceptions.Timeout:
        return {"message": "Timeout, still listening..."}
    except Exception as e:
        print(f"Client: Error sending chunk: {e}")
        return {"message": "Error sending chunk"}

def commit_audio_to_server(session_id: str, callback_url: str):
    data = {"session_id": session_id, "callback_url": callback_url}
    try:
        # Server returns immediately; keep timeout small
        resp = requests.post(f"{SERVER_URL}/commit_audio", data=data, timeout=3)
        resp.raise_for_status()
        print("commited audio to server")
        return resp.json()
    except requests.exceptions.Timeout:
        # Not fatal: server most likely started processing anyway
        print("Client: Commit request short-timeout hit; continuing (server processes in background).")
        return {"message": "Commit short-timeout"}
    except Exception as e:
        print(f"Client: Error committing audio: {e}")
        return {"message": "Error committing"}

def is_silent_chunk(audio_chunk, threshold_rms):
    # Avoid NaN in case of all zeros
    if audio_chunk.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio_chunk, dtype=np.float64))))
    return rms < threshold_rms

############################
# Text-to-Speech Worker
############################
def tts_worker():
    engine = pyttsx3.init()
    print("Client: TTS worker running.")
    while not stop_event.is_set():
        try:
            text = tts_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if text is None:
            break
        try:
            print(f"Assistant (speaking): {text}")
            engine.say(text)
            engine.runAndWait()
        finally:
            tts_queue.task_done()
    engine.stop()

############################
# Background Audio Sender
############################
def chunk_sender(session_id):
    silent_chunks = 0
    is_speaking = False
    waiting_response = False

    while not stop_event.is_set():
        try:
            # Use short timeout so we can also react to responses/stop_event quickly
            audio_chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            # No audio pulled this cycle; still check responses & continue
            audio_chunk = None

        # If no new audio, move on
        if audio_chunk is None:
            continue

        # --- Voice Activity Detection ---
        if is_silent_chunk(audio_chunk, SILENCE_THRESHOLD_RMS):
            silent_chunks += 1
        else:
            silent_chunks = 0
            if not is_speaking and not waiting_response:
                print("Client: Speech detected -> sending chunks.")
                is_speaking = True

        # --- Send or Commit ---
        if is_speaking and not waiting_response:
            if silent_chunks < CONSECUTIVE_SILENT_CHUNKS:
                send_chunk(session_id, audio_chunk)
            else:
                # End of utterance
                print("Client: Silence threshold reached. Committing.")
                is_speaking = False
                waiting_response = True
                commit_audio_to_server(session_id, CLIENT_CALLBACK_URL)
                waiting_response = False
                silent_chunks = 0
                is_speaking = False

############################
# Client Main
############################
def main():
    session_id = start_session()
    if not session_id:
        print("Client: Failed to start session.")
        return

    set_session_context(session_id, TEST_SYSTEM_PROMPT, TEST_TRAVEL_DATA)

    samplerate = 16000
    blocksize = int(samplerate * 0.5)

    # Start local FastAPI server for receiving callbacks
    threading.Thread(
        target=uvicorn.run,
        args=(client_app,),
        kwargs={"host": CLIENT_HOST, "port": CLIENT_PORT, "log_level": "warning"},
        daemon=True
    ).start()
    time.sleep(0.8)  # brief warm-up

    # Start workers
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    sender_thread = threading.Thread(target=chunk_sender, args=(session_id,), daemon=True)
    sender_thread.start()

    print("Client: Recording... Speak into your microphone. Say 'goodbye' to exit.")

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", blocksize=blocksize) as stream:
            while not stop_event.is_set():
                chunk, overflowed = stream.read(blocksize)
                if overflowed:
                    print("Client: Buffer overflow!")
                # Push raw audio continuously so VAD & timing remain responsive
                audio_queue.put(np.squeeze(chunk))
    except KeyboardInterrupt:
        print("\nClient: Stopped by user.")
    except Exception as e:
        print(f"Client: Audio error: {e}")
    finally:
        # Signal threads to stop and drain
        stop_event.set()
        audio_queue.put(None)
        tts_queue.put(None)

        # Give threads a moment to exit gracefully
        try:
            sender_thread.join(timeout=2.0)
        except RuntimeError:
            pass
        try:
            tts_thread.join(timeout=2.0)
        except RuntimeError:
            pass

        end_resp = end_session(session_id)
        print("Client: Session closed:", end_resp)

if __name__ == "__main__":
    main()
