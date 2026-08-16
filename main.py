"""
Weg. Backend — echter Server mit Gedächtnis.

Löst zwei Dinge, die eine reine HTML-Datei nicht kann:
1. Alles wird dauerhaft gespeichert (SQLite-Datenbank statt Browser-Variable,
   die beim Schliessen verschwindet).
2. Der Chat/die Zielzerlegung läuft über die echte Anthropic API, mit der
   gesamten bisherigen Journal-/Ziel-/Aufgaben-Historie als Kontext — statt
   der 5 fest programmierten Kategorien aus dem Vorgänger-Prototyp.

Starten (lokal):
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=dein-api-key
    uvicorn main:app --reload --port 8000

Siehe README.md für Details, inkl. kostenloser Deployment-Optionen.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = os.environ.get("WEG_DB_PATH", "weg.db")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-6"

app = FastAPI(title="Weg. Backend")

# Für den Prototyp offen für alle Origins — vor echtem Launch einschränken
# auf die tatsächliche Domain der Frontend-Seite.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Datenbank
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,          -- 'goal' | 'priority' | 'task' | 'journal' | 'chat_user' | 'chat_assistant'
                content TEXT NOT NULL,
                done INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)


init_db()


# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------

class EntryIn(BaseModel):
    type: str
    content: str


class EntryUpdate(BaseModel):
    done: Optional[bool] = None
    content: Optional[str] = None


class ChatIn(BaseModel):
    message: str


class GoalIn(BaseModel):
    goal: str


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_entries(conn, type_filter: Optional[str] = None, limit: int = 200):
    if type_filter:
        rows = conn.execute(
            "SELECT * FROM entries WHERE type = ? ORDER BY id DESC LIMIT ?",
            (type_filter, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM entries ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def build_memory_context(conn) -> str:
    """
    Baut eine kompakte Zusammenfassung der bisherigen Historie, die als
    Kontext an Claude mitgegeben wird — das ist das eigentliche "Gedächtnis".
    """
    goals = fetch_entries(conn, "goal", limit=10)
    journal = fetch_entries(conn, "journal", limit=10)
    tasks = fetch_entries(conn, "task", limit=20)
    chat = conn.execute(
        "SELECT * FROM entries WHERE type IN ('chat_user','chat_assistant') ORDER BY id DESC LIMIT 20"
    ).fetchall()
    chat = list(reversed([dict(r) for r in chat]))

    parts = []
    if goals:
        parts.append("Bisherige Ziele:\n" + "\n".join(f"- {g['content']}" for g in goals))
    if tasks:
        offen = [t for t in tasks if not t["done"]]
        erledigt = [t for t in tasks if t["done"]]
        parts.append(
            f"Offene Aufgaben ({len(offen)}):\n" + "\n".join(f"- {t['content']}" for t in offen[:10])
        )
        if erledigt:
            parts.append(f"Kürzlich erledigt: {len(erledigt)} Aufgaben.")
    if journal:
        parts.append("Journal-Einträge (neueste zuerst):\n" + "\n".join(f"- {j['content']}" for j in journal))

    return "\n\n".join(parts) if parts else "Noch keine bisherige Historie vorhanden."


async def call_claude(system_prompt: str, user_message: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY ist nicht gesetzt. Siehe README.md.",
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Anthropic API Fehler: {response.text}")

    data = response.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(text_blocks) if text_blocks else "(keine Antwort erhalten)"


MENTOR_SYSTEM_PROMPT = """Du bist der "Weg."-Mentor, ein persönlicher Struktur-Begleiter für jemanden, \
der gerade den Übergang von einer Festanstellung in die Selbständigkeit in der Schweiz durchläuft.

Du kennst die bisherige Geschichte der Person (Ziele, Aufgaben, Journal-Einträge) — nutze das aktiv, \
um wirklich persönlich zu antworten, nicht generisch. Beziehe dich konkret auf das, was die Person \
bisher erwähnt hat, wenn es relevant ist.

Regeln:
- Antworte warm, aber sachlich - keine übertriebene Cheerleader-Sprache.
- Kurz und konkret, auf Deutsch.
- Bei RAV/AHV/Steuerfragen: allgemeine Informationen ja, aber keine verbindliche Rechts- oder \
Steuerberatung. Bei unklaren Einzelfällen auf RAV/Ausgleichskasse/Treuhänder verweisen.
- Erfinde keine Fakten über die Person, die nicht im Kontext stehen."""


GOAL_SYSTEM_PROMPT = """Du bist der "Weg."-Ziel-Agent. Die Person beschreibt ein Ziel für die Woche \
oder den Monat. Zerlege es in genau 3 konkrete Prioritäten und pro Priorität 1-2 sofort umsetzbare \
Aufgaben. Berücksichtige die bisherige Historie der Person, falls relevant (z.B. keine Aufgabe \
vorschlagen, die laut Historie schon erledigt ist).

Antworte AUSSCHLIESSLICH als JSON in diesem Format, ohne zusätzlichen Text:
{
  "kategorie": "kurze Bezeichnung, z.B. 'Akquise-Ziel'",
  "begruendung": "1 Satz, warum diese Einordnung",
  "prioritaeten": ["Priorität 1", "Priorität 2", "Priorität 3"],
  "aufgaben": ["Aufgabe 1", "Aufgabe 2", "Aufgabe 3", "Aufgabe 4"]
}"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/entries")
def list_entries(type: Optional[str] = None):
    with get_db() as conn:
        return fetch_entries(conn, type)


@app.post("/entries")
def create_entry(entry: EntryIn):
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO entries (type, content, done, created_at) VALUES (?, ?, 0, ?)",
            (entry.type, entry.content, now_iso()),
        )
        return {"id": cur.lastrowid}


@app.patch("/entries/{entry_id}")
def update_entry(entry_id: int, update: EntryUpdate):
    with get_db() as conn:
        if update.done is not None:
            conn.execute("UPDATE entries SET done = ? WHERE id = ?", (int(update.done), entry_id))
        if update.content is not None:
            conn.execute("UPDATE entries SET content = ? WHERE id = ?", (update.content, entry_id))
        return {"ok": True}


@app.delete("/entries/{entry_id}")
def delete_entry(entry_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        return {"ok": True}


@app.post("/chat")
async def chat(payload: ChatIn):
    with get_db() as conn:
        memory = build_memory_context(conn)
        conn.execute(
            "INSERT INTO entries (type, content, done, created_at) VALUES ('chat_user', ?, 0, ?)",
            (payload.message, now_iso()),
        )

    system_prompt = f"{MENTOR_SYSTEM_PROMPT}\n\n--- Bisherige Historie der Person ---\n{memory}"
    answer = await call_claude(system_prompt, payload.message)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO entries (type, content, done, created_at) VALUES ('chat_assistant', ?, 0, ?)",
            (answer, now_iso()),
        )

    return {"answer": answer}


@app.post("/goal/decompose")
async def decompose_goal(payload: GoalIn):
    import json

    with get_db() as conn:
        memory = build_memory_context(conn)
        conn.execute(
            "INSERT INTO entries (type, content, done, created_at) VALUES ('goal', ?, 0, ?)",
            (payload.goal, now_iso()),
        )

    system_prompt = f"{GOAL_SYSTEM_PROMPT}\n\n--- Bisherige Historie der Person ---\n{memory}"
    raw = await call_claude(system_prompt, payload.goal)

    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        raise HTTPException(status_code=502, detail=f"Konnte Antwort nicht parsen: {raw}")

    with get_db() as conn:
        for p in parsed.get("prioritaeten", []):
            conn.execute(
                "INSERT INTO entries (type, content, done, created_at) VALUES ('priority', ?, 0, ?)",
                (p, now_iso()),
            )
        for t in parsed.get("aufgaben", []):
            conn.execute(
                "INSERT INTO entries (type, content, done, created_at) VALUES ('task', ?, 0, ?)",
                (t, now_iso()),
            )

    return parsed


@app.get("/health")
def health():
    return {"status": "ok", "api_key_configured": bool(ANTHROPIC_API_KEY)}
