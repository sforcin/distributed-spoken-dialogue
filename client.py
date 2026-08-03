"""
client.py
WebSocket dialogue client — server-controlled recording (no keypress).
"""

import io
import json
import os
import time
import argparse

import numpy as np
import sounddevice as sd
import soundfile as sf
from websocket import create_connection

DEFAULT_SERVER = os.environ.get("DIALOGUE_SERVER", "ws://localhost:5050/dialogue_ws")

SAMPLE_RATE = 16000
CHANNELS    = 1
MIC_DEVICE  = 2   # C922 Pro Stream Webcam mic, from earlier `python3 -m sounddevice` check


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


class AudioPlayer:
    def __init__(self):
        self.stream = sd.RawOutputStream(
            samplerate=24000,
            channels=1,
            dtype="int16",
        )
        self.stream.start()

    def write(self, pcm_bytes: bytes):
        if len(pcm_bytes) % 2 == 1:
            pcm_bytes = pcm_bytes[:-1]
        self.stream.write(pcm_bytes)

    def close(self):
        self.stream.stop()
        self.stream.close()


class DialogueClient:
    def __init__(self, server_url: str):
        self.ws = create_connection(server_url)
        print("[client] WebSocket connected")
        self.audio_player = AudioPlayer()

        self.recording = False
        self.frames = []
        self.input_stream = None

    # recording control — triggered by messages from the server now
    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        self.frames = []

        def callback(indata, frame_count, time_info, status):
            if status:
                print("[audio]", status)
            self.frames.append(indata.copy())

        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=MIC_DEVICE,
            callback=callback,
        )
        self.input_stream.start()
        print("🎤 Recording started (server-triggered)")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.input_stream.stop()
        self.input_stream.close()
        self.input_stream = None

        audio = np.concatenate(self.frames, axis=0) if self.frames else np.array([], dtype=np.float32)
        print("🎤 Recording stopped")

        if len(audio) == 0:
            print("[skipped — no audio recorded]")
            return

        print("[client] Sending audio...")
        self.send_audio(audio)

    def send_audio(self, audio: np.ndarray):
        wav = audio_to_wav_bytes(audio)
        self.ws.send_binary(wav)

    # incoming message dispatch
    def handle_message(self, msg):
        if isinstance(msg, bytes):
            self.audio_player.write(msg)
            return

        if not msg or not isinstance(msg, str):
            return

        msg = msg.strip()
        if not msg:
            return

        try:
            event = json.loads(msg)
        except json.JSONDecodeError:
            print(f"[WARN] Bad message from server: {repr(msg[:80])}")
            return

        t = event.get("type")

        if t == "start_recording":
            self.start_recording()
        elif t == "stop_recording":
            self.stop_recording()
        elif t == "sentence":
            print(f"[client] {event['text']}")
        elif t == "end_turn":
            print("[client] (turn complete, waiting for next start)")

    def recv_loop(self):
        """Runs forever: audio playback + control messages both arrive here."""
        while True:
            msg = self.ws.recv()
            self.handle_message(msg)


def main(server: str, topic: str):
    url = f"ws://{server.replace('ws://','').replace('http://','')}/dialogue_ws"
    client = DialogueClient(url)

    print(f"[client] Sending preferences (topic={topic})...")
    client.ws.send(json.dumps({
        "type": "set_preferences",
        "topic": topic,
    }))

    while True:
        msg = client.ws.recv()
        event = json.loads(msg)
        if event.get("type") == "preferences_set":
            print("[client] Server ready")
            break

    print("⏳ Waiting for intro...\n")

    print("Ready — waiting for start/stop signals from Pastoral.\n")
    try:
        client.recv_loop()
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dialogue client")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--topic", default="function_machine")
    args = parser.parse_args()

    main(args.server, args.topic)