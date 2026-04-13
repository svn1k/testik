import asyncio
import json
import os
import re
from datetime import datetime

import opengradient as og
import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Изменено: убраны static_folder и static_url_path
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # Изменено: отдает index.html из текущей директории (рядом со скриптом)
    return send_file('index.html')

# ─── Configuration ──────────────────────────────────────────────
OG_PRIVATE_KEY   = os.environ.get("OG_PRIVATE_KEY", "")
MEMSYNC_API_KEY  = os.environ.get("MEMSYNC_API_KEY", "")
MEMSYNC_BASE_URL = "https://api.memchat.io/v1"
LLM_MODEL        = og.TEE_LLM.GPT_4_1_2025_04_14

# ─── LLM Initialization ─────────────────────────────────────────
llm = og.LLM(private_key=OG_PRIVATE_KEY)

# One-time token approval (called on startup)
def init_approval():
    try:
        llm.ensure_opg_approval(0.1)
        print("✅ OPG approval ready")
    except Exception as e:
        print(f"⚠️ OPG approval warning: {e}")

# ─── MemSync Helpers ────────────────────────────────────────────
def memsync_headers():
    return {
        "X-API-Key": MEMSYNC_API_KEY,
        "Content-Type": "application/json"
    }

def save_memory(user_text: str, ai_response: str):
    """Saves entry + AI response to MemSync long-term memory."""
    if not MEMSYNC_API_KEY:
        return None
    data = {
        "messages": [
            {"role": "user",      "content": f"Diary entry: {user_text}"},
            {"role": "assistant", "content": ai_response}
        ],
        "agent_id":  "psych-diary",
        "thread_id": f"diary-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "source":    "chat"
    }
    try:
        r = requests.post(f"{MEMSYNC_BASE_URL}/memories",
                          json=data, headers=memsync_headers(), timeout=10)
        return r.json()
    except Exception as e:
        print(f"MemSync save error: {e}")
        return None

def search_memories(query: str, limit: int = 5):
    """Searches relevant memories to personalize the response."""
    if not MEMSYNC_API_KEY:
        return []
    data = {"query": query, "limit": limit, "rerank": True}
    try:
        r = requests.post(f"{MEMSYNC_BASE_URL}/memories/search",
                          json=data, headers=memsync_headers(), timeout=10)
        result = r.json()
        return result.get("memories", [])
    except Exception as e:
        print(f"MemSync search error: {e}")
        return []

def get_user_profile():
    """Fetches user profile from MemSync."""
    if not MEMSYNC_API_KEY:
        return {}
    try:
        r = requests.get(f"{MEMSYNC_BASE_URL}/users/profile",
                         headers=memsync_headers(), timeout=10)
        return r.json()
    except Exception as e:
        print(f"MemSync profile error: {e}")
        return {}

# ─── LLM Helpers ────────────────────────────────────────────────
def build_context_from_memories(memories: list) -> str:
    if not memories:
        return ""
    lines = ["From previous user entries:"]
    for m in memories[:4]:
        lines.append(f"  • {m.get('memory', '')}")
    return "\n".join(lines) + "\n\n"

async def ai_respond(entry_text: str, memories: list) -> str:
    context = build_context_from_memories(memories)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm, attentive psychological diary assistant. "
                "Your task: analyze user entries, reflect their emotions, "
                "notice patterns (anxiety, fatigue, joy, stress) and give "
                "gentle, concrete advice. Rules:\n"
                "- Answer only in English\n"
                "- Be empathetic and non-judgmental\n"
                "- Name emotions directly\n"
                "- If you notice worrying patterns over several days, gently point it out\n"
                "- Give 1-2 practical tips\n"
                "- Use the context of past entries for personalization\n"
                "- Response structure: [Reflect feelings] -> [Analyze pattern] -> [Advice]\n"
                "- Length: 3-5 paragraphs"
            )
        },
        {
            "role": "user",
            "content": f"{context}Today's entry:\n{entry_text}"
        }
    ]
    result = await llm.chat(model=LLM_MODEL, messages=messages, max_tokens=600, temperature=0.7)
    return result.chat_output.get("content", "")

async def ai_mood(entry_text: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": (
                f"Analyze the mood of this diary entry.\n"
                f"Text: {entry_text}\n\n"
                f"Reply ONLY with valid JSON without explanations:\n"
                f'{{\"mood\": \"one word in english\", \"score\": number from 1 to 10, '
                f'\"emoji\": \"one emoji\", \"tags\": [\"tag1\", \"tag2\"]}}'
            )
        }
    ]
    result = await llm.chat(model=LLM_MODEL, messages=messages, max_tokens=100, temperature=0.0)
    raw = result.chat_output.get("content", "{}")
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except Exception:
        return {"mood": "neutral", "score": 5, "emoji": "😐", "tags": []}

# ─── Routes ─────────────────────────────────────────────────────
@app.route("/api/entry", methods=["POST"])
def new_entry():
    """Receives a diary entry, returns AI response + mood analysis."""
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Entry text is empty"}), 400

    # Search context from past entries
    memories = search_memories(text)

    # Run both LLM requests in parallel
    async def run_all():
        response_task = ai_respond(text, memories)
        mood_task     = ai_mood(text)
        return await asyncio.gather(response_task, mood_task)

    ai_text, mood_data = asyncio.run(run_all())

    # Save to long-term memory
    save_memory(text, ai_text)

    return jsonify({
        "response":  ai_text,
        "mood":      mood_data,
        "memories_used": len(memories),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/profile", methods=["GET"])
def profile():
    """Returns user psychological profile from MemSync."""
    data = get_user_profile()
    return jsonify(data)

@app.route("/api/memories", methods=["GET"])
def memories():
    """Search memories by query."""
    query = request.args.get("q", "user mood")
    mems  = search_memories(query, limit=10)
    return jsonify({"memories": mems})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "llm":      bool(OG_PRIVATE_KEY),
        "memsync":  bool(MEMSYNC_API_KEY),
        "time":     datetime.now().isoformat()
    })

# ─── Start ──────────────────────────────────────────────────────
if __name__ == "__main__":
    init_approval()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)