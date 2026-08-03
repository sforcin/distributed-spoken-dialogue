"""
app.py — combined survey site + dialogue server, all on Pastoral.
"""

import os
import re
import json
import time
import uuid
import sqlite3
import tempfile
import subprocess
import threading
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sock import Sock
from kokoro import KPipeline
import ollama

from prompts import (
    SYSTEM_PROMPTS,
    OPENING_PROMPTS,
    get_task,
    check_number,
    get_fresh_examples,
)

# --- Config ---
APP_SECRET = os.environ.get("APP_SECRET", "change-this-to-something-random")
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "responses.sqlite")

WHISPER_BIN   = os.environ.get("WHISPER_BIN", "/home/lydia/docs/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "/home/lydia/docs/whisper.cpp/models/ggml-large-v3-turbo.bin")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL",  "llama3.1")
KOKORO_VOICE  = os.environ.get("KOKORO_VOICE",  "af_heart")
KOKORO_LANG   = os.environ.get("KOKORO_LANG",   "a")

app = Flask(__name__)
app.secret_key = APP_SECRET
sock = Sock(app)

print("Loading Kokoro TTS pipeline...")
tts_pipeline = KPipeline(lang_code=KOKORO_LANG)
print("Kokoro ready.")

latest_game_state = {"works": [], "doesnt_work": []}
active_session = None


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS participants (
            participant_id    TEXT PRIMARY KEY,
            age               INTEGER,
            sex               TEXT,
            gender            TEXT,
            ethnicity         TEXT,
            education         TEXT,
            robot_experience  TEXT,
            created_at        TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id  TEXT NOT NULL,
            survey          TEXT NOT NULL,
            question        TEXT NOT NULL,
            answer          TEXT,
            created_at      TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def db_connect():
    return sqlite3.connect(DB_PATH, timeout=10)


def get_participant_id():
    if "participant_id" not in session:
        session["participant_id"] = str(uuid.uuid4())[:8]
    return session["participant_id"]


def save_participant(answers):
    pid = get_participant_id()
    conn = db_connect()
    conn.execute(
        """INSERT OR IGNORE INTO participants
           (participant_id, age, sex, gender, ethnicity, education, robot_experience, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pid,
            answers.get("age"),
            answers.get("sex"),
            answers.get("gender"),
            answers.get("ethnicity"),
            answers.get("education"),
            answers.get("robot_experience"),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def save_answers(survey_key, answers):
    pid = get_participant_id()
    conn = db_connect()
    cur = conn.cursor()
    for q, a in answers.items():
        cur.execute(
            "SELECT id FROM responses WHERE participant_id = ? AND survey = ? AND question = ?",
            (pid, survey_key, q),
        )
        if cur.fetchone():
            continue
        cur.execute(
            "INSERT INTO responses (participant_id, survey, question, answer, created_at) VALUES (?, ?, ?, ?, ?)",
            (pid, survey_key, q, a, datetime.utcnow().isoformat()),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Dialogue Session
# ---------------------------------------------------------------------------

STATUS_WORDS = ("correct", "incorrect", "give_up", "in_progress")


class DialogueSession:
    def __init__(self, ws):
        self.ws = ws
        self.history = []
        self.lock = threading.Lock()
        self.topic = None
        self.system_prompt = None
        self.opening_prompt = None

        # Calculator-tracked game state
        self.task_index = 0
        self.shown = set()   # numbers already offered to the model this task

    def send_json(self, obj):
        self.ws.send(json.dumps(obj))

    def send_audio(self, pcm_bytes: bytes):
        self.ws.send(pcm_bytes)

    def set_preferences(self, topic: str):
        self.topic = topic
        self.system_prompt = SYSTEM_PROMPTS[topic]
        self.opening_prompt = OPENING_PROMPTS.get(topic, "Hi, let's start a conversation.")

    def send_intro(self):
        print("[server] Generating intro...")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.opening_prompt},
        ]
        text, _status = self._run_turn(messages)
        self.history.append({"role": "assistant", "content": text})
        self._stream_text(text)

    def _build_reminder(self, user_text: str) -> str:
        examples = get_fresh_examples(self.task_index, self.shown)
        for n, _label in examples:
            self.shown.add(n)
        example_lines = "\n".join(f"{n} -> {label}" for n, label in examples) or "(none left — reuse earlier ones from the conversation)"

        number_note = ""
        match = re.search(r'\b(\d+)\b', user_text or "")
        if match:
            number = int(match.group(1))
            label = check_number(self.task_index, number)
            number_note = f"\nThe student mentioned the number {number}. Its real verified answer: {number} {label}."

        return (
            "[INTERNAL — do not read this aloud]\n"
            "Pre-verified example numbers available this turn (use only these, never invent your own):\n"
            f"{example_lines}\n"
            f"{number_note}\n\n"
            "Respond with exactly two lines:\n"
            "STATUS: correct | incorrect | give_up | in_progress\n"
            "REPLY: <what to say out loud to the student>"
        )

    def _run_turn(self, messages):
        """One LLM call per turn — no round-trips. Every number the model
        could need is already provided, pre-verified, in the reminder block."""
        messages = list(messages)
        last_user_text = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        reminder = self._build_reminder(last_user_text)

        if messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": messages[-1]["content"] + "\n\n" + reminder,
            }
        else:
            messages.append({"role": "user", "content": reminder})

        res = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        content = res["message"]["content"].strip()

        # 1. Proper "STATUS: x" / "REPLY: y" format
        status_match = re.search(r"STATUS:\s*(\w+)", content, re.IGNORECASE)
        reply_match = re.search(r"REPLY:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        if status_match and reply_match and reply_match.group(1).strip():
            return reply_match.group(1).strip(), status_match.group(1).lower()

        # 2. Model dropped the labels but still put the bare status word on
        # its own first line, followed by the reply text — accept that shape.
        first_line, _, rest = content.partition("\n")
        first_word = first_line.strip().lower().rstrip(".:")
        if first_word in STATUS_WORDS and rest.strip():
            return rest.strip(), first_word

        # 3. Totally malformed — strip any stray labels/status words so
        # nothing broken gets spoken, then speak whatever's left.
        cleaned = re.sub(r'^\s*STATUS:.*$', '', content, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*REPLY:\s*', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(
            rf'^\s*({"|".join(STATUS_WORDS)})\s*\.?\s*$',
            '', cleaned, flags=re.IGNORECASE | re.MULTILINE
        ).strip()
        if not cleaned:
            cleaned = content
        print(f"[WARN] Reply didn't match protocol, speaking cleaned version: {cleaned!r}")
        return cleaned, "in_progress"

    def handle_user_audio(self, audio_bytes: bytes):
        turn_start = time.time()

        t0 = time.time()
        transcript = transcribe(audio_bytes)
        stt_time = time.time() - t0
        print(f"[USER] {transcript}")
        print(f"[timing] STT: {stt_time:.2f}s")

        self.history.append({"role": "user", "content": transcript})

        t0 = time.time()
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        text, status = self._run_turn(messages)
        llm_time = time.time() - t0
        print(f"[timing] LLM: {llm_time:.2f}s")
        print(f"[game] status={status}, task_index={self.task_index} ({get_task(self.task_index)['name']})")

        self.history.append({"role": "assistant", "content": text})

        if status in ("correct", "give_up"):
            self.task_index += 1
            self.shown = set()

        t0 = time.time()
        self._stream_text(text)
        tts_time = time.time() - t0
        print(f"[timing] TTS (total, incl. streaming): {tts_time:.2f}s")

        total_time = time.time() - turn_start
        print(f"[timing] TOTAL turn: {total_time:.2f}s  (STT {stt_time:.2f}s / LLM {llm_time:.2f}s / TTS {tts_time:.2f}s)")

    def _stream_text(self, text: str):
        for i, sentence in enumerate(split_sentences(text)):
            sentence = sentence.strip()
            if not sentence:
                continue
            self.send_json({"type": "sentence", "i": i, "text": sentence})

            t0 = time.time()
            for _, _, audio in tts_pipeline(sentence, voice=KOKORO_VOICE):
                self.send_audio(to_pcm(audio))
                time.sleep(0.01)
            sentence_tts_time = time.time() - t0
            print(f"[timing]   sentence {i} TTS: {sentence_tts_time:.2f}s  ({len(sentence)} chars)")

        self.send_json({"type": "end_turn"})


@sock.route("/dialogue_ws")
def ws_handler(ws):
    global active_session
    dsession = DialogueSession(ws)
    active_session = dsession
    print("[server] Blossom client connected")

    while True:
        msg = ws.receive()
        if msg is None:
            active_session = None
            return
        try:
            event = json.loads(msg)
            if event.get("type") == "set_preferences":
                topic = event.get("topic", "hello")
                dsession.set_preferences(topic)
                dsession.send_json({"type": "preferences_set", "topic": topic})
                print(f"[server] Topic set to '{topic}'")
                break
        except (json.JSONDecodeError, KeyError):
            print("[server] Ignored bad JSON during preference setting")

    dsession.send_intro()

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, bytes):
                dsession.handle_user_audio(msg)
            else:
                try:
                    event = json.loads(msg)
                    if event.get("type") == "reset":
                        dsession.history = []
                        print("[server] Conversation history reset")
                except json.JSONDecodeError:
                    pass
    finally:
        print("[server] Blossom client disconnected — saving transcript")
        save_transcript(dsession.history, dsession.topic)
        active_session = None


def to_pcm(audio) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


def transcribe(wav_bytes: bytes) -> str:
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        cmd = [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", tmp_path, "--output-txt"]
        subprocess.run(cmd, capture_output=True)
        txt_path = tmp_path + ".txt"
        if os.path.exists(txt_path):
            text = open(txt_path).read().strip()
            os.remove(txt_path)
            print(f"[timing]   whisper subprocess: {time.time() - t0:.2f}s")
            return text
    finally:
        os.remove(tmp_path)
    return ""


def split_sentences(text: str) -> list[str]:
    return re.findall(r'[^.!?]+[.!?]?', text)


def save_transcript(history: list, topic: str = None, save_dir: str = "transcripts"):
    os.makedirs(save_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    topic_str = topic or "unknown"
    path = os.path.join(save_dir, f"{date_str}_{topic_str}.txt")
    with open(path, "w") as f:
        for turn in history:
            f.write(f"{turn['role'].upper()}: {turn['content']}\n\n")
    print(f"[server] Transcript saved → {path}")


# ---------------------------------------------------------------------------
# Survey / participant flow
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    get_participant_id()
    return render_template("welcome.html")


@app.route("/survey/demographics", methods=["GET", "POST"])
def demographics():
    if request.method == "POST":
        answers = {
            "age": request.form.get("age"),
            "sex": request.form.get("sex"),
            "gender": request.form.get("gender_other") or request.form.get("gender"),
            "ethnicity": ", ".join(request.form.getlist("ethnicity")) + (
                (", " + request.form.get("ethnicity_other")) if request.form.get("ethnicity_other") else ""
            ),
            "education": request.form.get("education"),
            "robot_experience": request.form.get("robot_experience"),
        }
        save_participant(answers)
        return redirect(url_for("bfi10"))
    return render_template("demographics.html")


@app.route("/survey/bfi10", methods=["GET", "POST"])
def bfi10():
    if request.method == "POST":
        answers = {f"q{i}": request.form.get(f"q{i}") for i in range(1, 11)}
        save_answers("bfi10", answers)
        return redirect(url_for("nars"))
    return render_template("personalities.html")


@app.route("/survey/nars", methods=["GET", "POST"])
def nars():
    if request.method == "POST":
        answers = {f"q{i}": request.form.get(f"q{i}") for i in range(1, 15)}
        save_answers("nars", answers)
        return redirect(url_for("study"))
    return render_template("nars.html")


@app.route("/study")
def study():
    return render_template("index.html")


@app.route("/study/finish")
def study_finish():
    return redirect(url_for("rosas"))


@app.route("/survey/rosas", methods=["GET", "POST"])
def rosas():
    if request.method == "POST":
        answers = {f"q{i}": request.form.get(f"q{i}") for i in range(1, 7)}
        save_answers("rosas", answers)
        return redirect(url_for("sus"))
    return render_template("rosas.html")


@app.route("/survey/sus", methods=["GET", "POST"])
def sus():
    if request.method == "POST":
        answers = {f"q{i}": request.form.get(f"q{i}") for i in range(1, 11)}
        save_answers("sus", answers)
        return redirect(url_for("tam"))
    return render_template("sus.html")


@app.route("/survey/tam", methods=["GET", "POST"])
def tam():
    if request.method == "POST":
        answers = {f"q{i}": request.form.get(f"q{i}") for i in range(1, 7)}
        save_answers("tam", answers)
        return redirect(url_for("done"))
    return render_template("tam.html")


@app.route("/done")
def done():
    return render_template("done.html")


@app.route("/start")
def start():
    if active_session is None:
        return "no blossom client connected", 503
    active_session.send_json({"type": "start_recording"})
    print("[server] sent start_recording to Blossom")
    return "start sent"


@app.route("/stop")
def stop():
    if active_session is None:
        return "no blossom client connected", 503
    active_session.send_json({"type": "stop_recording"})
    print("[server] sent stop_recording to Blossom")
    return "stop sent"


@app.route("/update_game_state", methods=["POST"])
def update_game_state():
    global latest_game_state
    latest_game_state = request.get_json()
    print(f"[app] Game state updated: {latest_game_state}")
    return "ok"


@app.route("/game_state")
def game_state():
    return jsonify(latest_game_state)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

with app.app_context():
    init_db()


if __name__ == "__main__":
    print("Starting combined app + dialogue server on 0.0.0.0:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)