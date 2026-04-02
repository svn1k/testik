import asyncio
import json
import os
import re
from datetime import datetime

import opengradient as og
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

# ─── Конфигурация ───────────────────────────────────────────────
OG_PRIVATE_KEY   = os.environ.get("OG_PRIVATE_KEY", "")
MEMSYNC_API_KEY  = os.environ.get("MEMSYNC_API_KEY", "")
MEMSYNC_BASE_URL = "https://api.memchat.io/v1"
LLM_MODEL        = og.TEE_LLM.GPT_4_1_2025_04_14

# ─── Инициализация LLM ──────────────────────────────────────────
llm = og.LLM(private_key=OG_PRIVATE_KEY)

# Одноразовое одобрение токенов (вызывается при старте)
def init_approval():
    try:
        llm.ensure_opg_approval(0.1)
        print("✅ OPG approval ready")
    except Exception as e:
        print(f"⚠️  OPG approval warning: {e}")

# ─── MemSync хелперы ────────────────────────────────────────────
def memsync_headers():
    return {
        "X-API-Key": MEMSYNC_API_KEY,
        "Content-Type": "application/json"
    }

def save_memory(user_text: str, ai_response: str):
    """Сохраняет запись + ответ AI в долгосрочную память MemSync."""
    if not MEMSYNC_API_KEY:
        return None
    data = {
        "messages": [
            {"role": "user",      "content": f"Запись в дневнике: {user_text}"},
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
    """Ищет релевантные воспоминания для персонализации ответа."""
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
    """Получает профиль пользователя из MemSync."""
    if not MEMSYNC_API_KEY:
        return {}
    try:
        r = requests.get(f"{MEMSYNC_BASE_URL}/users/profile",
                         headers=memsync_headers(), timeout=10)
        return r.json()
    except Exception as e:
        print(f"MemSync profile error: {e}")
        return {}

# ─── LLM хелперы ────────────────────────────────────────────────
def build_context_from_memories(memories: list) -> str:
    if not memories:
        return ""
    lines = ["Из предыдущих записей пользователя:"]
    for m in memories[:4]:
        lines.append(f"  • {m.get('memory', '')}")
    return "\n".join(lines) + "\n\n"

async def ai_respond(entry_text: str, memories: list) -> str:
    context = build_context_from_memories(memories)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — тёплый, внимательный психологический дневник-ассистент. "
                "Твоя задача: анализировать записи пользователя, отражать его эмоции, "
                "замечать паттерны (тревога, усталость, радость, стресс) и давать "
                "мягкие, конкретные советы. Правила:\n"
                "- Отвечай только на русском\n"
                "- Будь эмпатичным и не осуждающим\n"
                "- Называй эмоции прямо\n"
                "- Если замечаешь тревожные паттерны несколько дней подряд — мягко об этом скажи\n"
                "- Давай 1–2 практичных совета\n"
                "- Используй контекст прошлых записей для персонализации\n"
                "- Структура ответа: [Отражение чувств] → [Анализ паттерна] → [Совет]\n"
                "- Длина: 3–5 абзацев"
            )
        },
        {
            "role": "user",
            "content": f"{context}Сегодняшняя запись:\n{entry_text}"
        }
    ]
    result = await llm.chat(model=LLM_MODEL, messages=messages, max_tokens=600, temperature=0.7)
    return result.chat_output.get("content", "")

async def ai_mood(entry_text: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": (
                f"Проанализируй настроение этой записи дневника.\n"
                f"Текст: {entry_text}\n\n"
                f"Ответь ТОЛЬКО валидным JSON без пояснений:\n"
                f'{{\"mood\": \"одно слово на русском\", \"score\": число от 1 до 10, '
                f'\"emoji\": \"один эмодзи\", \"tags\": [\"тег1\", \"тег2\"]}}'
            )
        }
    ]
    result = await llm.chat(model=LLM_MODEL, messages=messages, max_tokens=100, temperature=0.0)
    raw = result.chat_output.get("content", "{}")
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except Exception:
        return {"mood": "нейтральное", "score": 5, "emoji": "😐", "tags": []}

# ─── Роуты ──────────────────────────────────────────────────────
@app.route("/api/entry", methods=["POST"])
def new_entry():
    """Принимает запись дневника, возвращает ответ AI + анализ настроения."""
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Текст записи пуст"}), 400

    # Ищем контекст из прошлых записей
    memories = search_memories(text)

    # Запускаем оба LLM-запроса параллельно
    async def run_all():
        response_task = ai_respond(text, memories)
        mood_task     = ai_mood(text)
        return await asyncio.gather(response_task, mood_task)

    ai_text, mood_data = asyncio.run(run_all())

    # Сохраняем в долгосрочную память
    save_memory(text, ai_text)

    return jsonify({
        "response":  ai_text,
        "mood":      mood_data,
        "memories_used": len(memories),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/profile", methods=["GET"])
def profile():
    """Возвращает психологический профиль пользователя из MemSync."""
    data = get_user_profile()
    return jsonify(data)

@app.route("/api/memories", methods=["GET"])
def memories():
    """Поиск воспоминаний по запросу."""
    query = request.args.get("q", "настроение пользователя")
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

# ─── Старт ──────────────────────────────────────────────────────
if __name__ == "__main__":
    init_approval()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
