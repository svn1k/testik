import asyncio
import json
import os
import re
from datetime import datetime

import opengradient as og
import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_file('index.html')

# ─── Configuration ──────────────────────────────────────────────
# ПРЕДУПРЕЖДЕНИЕ: Обязательно установите эти переменные в окружении!
OG_PRIVATE_KEY   = os.environ.get("OG_PRIVATE_KEY", "")
MEMSYNC_API_KEY  = os.environ.get("MEMSYNC_API_KEY", "")
MEMSYNC_BASE_URL = "https://api.memchat.io/v1"
LLM_MODEL        = og.TEE_LLM.GPT_4_1_2025_04_14

llm = og.LLM(private_key=OG_PRIVATE_KEY)

def init_approval():
    try:
        llm.ensure_opg_approval(0.1)
    except:
        pass

# ─── Helpers ────────────────────────────────────────────
def memsync_headers():
    return {"X-API-Key": MEMSYNC_API_KEY, "Content-Type": "application/json"}

def save_memory(user_text: str, ai_response: str):
    if not MEMSYNC_API_KEY: return
    data = {
        "messages": [
            {"role": "user", "content": f"Multi-day reflection: {user_text}"},
            {"role": "assistant", "content": ai_response}
        ],
        "agent_id": "psych-diary"
    }
    try:
        requests.post(f"{MEMSYNC_BASE_URL}/memories", json=data, headers=memsync_headers(), timeout=5)
    except:
        pass

async def ai_respond(entry_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a warm AI psychologist. Analyze the user's weekly reflection. Be empathetic, notice trends over the days, and provide 1-2 practical tips. Use English."
        },
        {"role": "user", "content": f"Here is my reflection for the week:\n{entry_text}"}
    ]
    result = await llm.chat(model=LLM_MODEL, messages=messages, max_tokens=600, temperature=0.7)
    return result.chat_output.get("content", "")

async def ai_mood(entry_text: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": f"Analyze the mood of this text and return ONLY JSON:\n{entry_text}\n\n"
                       f"Format: {{\"mood\": \"word\", \"score\": 1-10, \"emoji\": \"😊\", \"tags\": [\"tag1\"]}}"
        }
    ]
    try:
        result = await llm.chat(model=LLM_MODEL, messages=messages, max_tokens=150, temperature=0.0)
        raw = result.chat_output.get("content", "{}")
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else {"mood": "neutral", "score": 5, "emoji": "😐", "tags": []}
    except:
        return {"mood": "neutral", "score": 5, "emoji": "😐", "tags": []}

# ─── Routes ─────────────────────────────────────────────────────
@app.route("/api/entry", methods=["POST"])
def new_entry():
    body = request.get_json()
    text = body.get("text", "").strip()
    if not text: return jsonify({"error": "Empty"}), 400

    async def process():
        return await asyncio.gather(ai_respond(text), ai_mood(text))

    ai_text, mood_data = asyncio.run(process())
    save_memory(text, ai_text)

    return jsonify({"response": ai_text, "mood": mood_data})

@app.route("/api/profile")
def profile():
    if not MEMSYNC_API_KEY: return jsonify({"bio": "MemSync key not configured."})
    try:
        r = requests.get(f"{MEMSYNC_BASE_URL}/users/profile", headers=memsync_headers(), timeout=5)
        return jsonify(r.json())
    except:
        return jsonify({"bio": "No profile data yet."})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    init_approval()
    app.run(host='0.0.0.0', port=5000)