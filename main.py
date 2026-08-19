"""
Weg. Backend — echter Server mit Gedächtnis.

Löst zwei Dinge, die eine reine HTML-Datei nicht kann:
1. Alles wird dauerhaft gespeichert — über eine echte Postgres-Datenbank
   (z.B. Supabase), NICHT über die lokale Festplatte des Servers. Wichtig:
   Render's kostenloses Tier hat ein "flüchtiges" Dateisystem — jede lokale
   Datei (inkl. einer SQLite-Datenbank) geht bei Neustart/Redeploy verloren.
   Deshalb Postgres über DATABASE_URL, nicht SQLite, sobald deployed.
2. Der Chat/die Zielzerlegung läuft über die echte Anthropic API, mit der
   gesamten bisherigen Journal-/Ziel-/Aufgaben-Historie als Kontext.

Starten (lokal, ohne eigene Postgres-Datenbank — nutzt automatisch SQLite):
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=dein-api-key
    uvicorn main:app --reload --port 8000

Starten (mit echter Postgres-Datenbank, z.B. Supabase — für Render-Deploy):
    zusätzlich: export DATABASE_URL=postgres://... (von Supabase kopiert)

Siehe README.md für Details zum Supabase-Setup.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
import httpx
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = os.environ.get("WEG_DB_PATH", "weg.db")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_POSTGRES = bool(DATABASE_URL)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-6"

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 30

app = FastAPI(title="Sole. Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Datenbank — Postgres (production/Render) oder SQLite (lokal, Fallback)
# ---------------------------------------------------------------------------

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras

    @contextmanager
    def get_db():
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db() -> None:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    done BOOLEAN DEFAULT FALSE,
                    parent_id INTEGER,
                    due_date TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            # Migration-safe: falls die Tabelle schon vorher ohne diese Spalten existierte
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS parent_id INTEGER")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS due_date TEXT")

    def run_query(conn, query: str, params: tuple = ()):
        """Führt eine Query aus und gibt eine Liste von dicts zurück (SELECT)."""
        pg_query = query.replace("?", "%s")
        cur = conn.cursor()
        cur.execute(pg_query, params)
        try:
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.ProgrammingError:
            return []

    def run_write(conn, query: str, params: tuple = ()):
        """Führt INSERT/UPDATE/DELETE aus, gibt bei INSERT die neue id zurück."""
        pg_query = query.replace("?", "%s")
        if pg_query.strip().upper().startswith("INSERT"):
            pg_query += " RETURNING id"
        cur = conn.cursor()
        cur.execute(pg_query, params)
        if pg_query.strip().upper().startswith("INSERT"):
            return cur.fetchone()["id"]
        return None

else:
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
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    done INTEGER DEFAULT 0,
                    parent_id INTEGER,
                    due_date TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            # Migration-safe: falls die Tabelle schon vorher ohne diese Spalten existierte
            for col_def in ["parent_id INTEGER", "due_date TEXT"]:
                try:
                    conn.execute(f"ALTER TABLE entries ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # Spalte existiert schon

    def run_query(conn, query: str, params: tuple = ()):
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def run_write(conn, query: str, params: tuple = ()):
        cur = conn.execute(query, params)
        return cur.lastrowid


init_db()


# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------

class EntryIn(BaseModel):
    type: str
    content: str
    due_date: Optional[str] = None


class EntryUpdate(BaseModel):
    done: Optional[bool] = None
    content: Optional[str] = None


class ChatIn(BaseModel):
    message: str


class GoalIn(BaseModel):
    goal: str


class SignupIn(BaseModel):
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_token(user_id: int, email: str) -> str:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET ist nicht gesetzt. Siehe README.md.")
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: str = Header(None)) -> dict:
    """
    Liest den 'Authorization: Bearer <token>'-Header, prüft das Token,
    gibt {user_id, email} zurück. Wird als Depends() an jeden geschützten
    Endpoint gehängt, damit jede Anfrage weiss, wem die Daten gehören.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Nicht eingeloggt.")
    token = authorization.removeprefix("Bearer ").strip()
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET ist nicht gesetzt. Siehe README.md.")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Sitzung abgelaufen, bitte neu einloggen.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Ungültiges Login-Token.")
    return {"user_id": payload["user_id"], "email": payload["email"]}


def fetch_entries(conn, user_id: int, type_filter: Optional[str] = None, limit: int = 200):
    if type_filter:
        return run_query(
            conn,
            "SELECT * FROM entries WHERE user_id = ? AND type = ? ORDER BY id DESC LIMIT ?",
            (user_id, type_filter, limit),
        )
    return run_query(
        conn, "SELECT * FROM entries WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
    )


def fetch_entries_by_types(conn, user_id: int, types: list[str], limit: int = 200):
    placeholders = ",".join("?" for _ in types)
    query = f"SELECT * FROM entries WHERE user_id = ? AND type IN ({placeholders}) ORDER BY id DESC LIMIT ?"
    return run_query(conn, query, (user_id, *types, limit))


def build_memory_context(conn, user_id: int) -> str:
    """
    Baut eine kompakte Zusammenfassung der bisherigen Historie EINER Person,
    die als Kontext an Claude mitgegeben wird — das eigentliche "Gedächtnis".
    """
    journal = fetch_entries(conn, user_id, "journal", limit=10)
    tasks = fetch_entries(conn, user_id, "task", limit=30)
    vision = fetch_entries(conn, user_id, "vision", limit=1)
    projects = fetch_entries(conn, user_id, "project", limit=10)
    chat = fetch_entries_by_types(conn, user_id, ["chat_user", "chat_assistant"], limit=20)
    chat = list(reversed(chat))

    parts = []
    if vision:
        parts.append("Strategisches Ziel / Vision der Person:\n" + vision[0]["content"])
    if projects:
        parts.append(
            "Aktuelle Projekte (Teil der Strategie):\n"
            + "\n".join(f"- {p['content']}" for p in projects)
        )
    if tasks:
        offen = [t for t in tasks if not t["done"]]
        erledigt = [t for t in tasks if t["done"]]
        parts.append(
            f"Offene Aufgaben ({len(offen)}):\n" + "\n".join(f"- {t['content']}" for t in offen[:15])
        )
        if erledigt:
            parts.append(f"Kürzlich erledigt: {len(erledigt)} Aufgaben.")
    if journal:
        parts.append("Frühere Reflexions-Notizen:\n" + "\n".join(f"- {j['content']}" for j in journal))

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


MENTOR_SYSTEM_PROMPT = """Du bist der "Sole."-Mentor: eine Kombination aus persönlichem Chief of Staff \
und strategischem Sparring-Partner für jemanden, der gerade den Übergang von einer Festanstellung in \
die Selbständigkeit in der Schweiz durchläuft. Diese Chat-Seite ist die zentrale Startseite der Person \
— hier landet alles: spontane Gedanken (Braindump), zu erledigende Dinge, und strategische Fragen.

Du kennst die bisherige Geschichte der Person (strategisches Ziel/Vision, Projekte, offene Aufgaben, \
frühere Reflexionen) — nutze das aktiv, um wirklich persönlich zu antworten, nicht generisch.

DEINE ZWEI FUNKTIONEN IN JEDER NACHRICHT:

1. AUFGABEN ERKENNEN UND ORGANISIEREN: Wenn die Nachricht der Person konkrete To-dos, Pläne oder \
Dinge enthält, die erledigt werden müssen (auch beiläufig erwähnt, als Liste, oder mitten in einem \
längeren Text) - extrahiere diese als einzelne, klare Aufgaben. Für jede Aufgabe schätze grob ein, \
wann sie fällig sein sollte: "heute", "morgen", "diese_woche", oder null (kein klarer Zeitrahmen, \
kommt in die allgemeine Liste). Erfinde KEINE Aufgaben, die nicht wirklich in der Nachricht \
angedeutet wurden. Reine Reflexion, Fragen oder ein Gespräch ohne konkrete To-dos: leeres \
Aufgaben-Array, das ist normal und richtig so.

2. STRATEGISCHES SPARRING: Das ist deine wichtigste Rolle, nicht nur Nebensache:
- Du bist primär STRATEGISCH, nicht operativ. Die Frage "was steht heute an" beantwortet die \
Aufgaben-Extraktion oben bereits - deine eigentliche Stärke ist "was ist eigentlich wichtig, und warum".
- Wenn die Person Anzeichen zeigt, mehrere Dinge gleichzeitig anzufangen oder sich zu verzetteln, \
sprich das direkt und freundlich an - hilf, einen Fokus zu finden, statt jede neue Idee unterstützend \
zu bestätigen.
- Wenn die Person sehr euphorisch über eine neue Idee klingt, darfst du diese Euphorie sanft erden, \
mit einer ehrlichen, wohlwollenden Nachfrage - nicht bremsen um des Bremsens willen, sondern um echte \
Reflexion statt reinem Enthusiasmus anzuregen.
- Schiess nicht vorschnell auf eine einzelne Idee oder Lösung ein, nur weil die Person sie gerade \
erwähnt hat. Frag nach, biete Perspektiven, statt die erste Idee unhinterfragt zu bestärken.
- Erinnere die Person bei Gelegenheit an ihr strategisches Ziel/ihre Vision (falls im Kontext \
vorhanden), besonders wenn die aktuelle Nachricht davon abzuweichen scheint.

Weitere Regeln:
- Antworte warm, aber sachlich - keine übertriebene Cheerleader-Sprache.
- Kurz und konkret, auf Deutsch.
- Bei RAV/AHV/Steuerfragen: allgemeine Informationen ja, aber keine verbindliche Rechts- oder \
Steuerberatung. Bei unklaren Einzelfällen auf RAV/Ausgleichskasse/Treuhänder verweisen.
- Erfinde keine Fakten über die Person, die nicht im Kontext stehen.

Antworte AUSSCHLIESSLICH als JSON in diesem Format, ohne zusätzlichen Text:
{
  "antwort": "deine eigentliche Chat-Antwort als Sparring-Partner, kann Markdown enthalten (**fett**, > Zitate)",
  "neue_aufgaben": [
    {"inhalt": "konkrete Aufgabe", "faellig": "heute"},
    {"inhalt": "andere Aufgabe", "faellig": null}
  ]
}
Falls keine Aufgaben erkennbar sind: "neue_aufgaben": []"""


STRATEGY_SYSTEM_PROMPT = """Du hilfst dabei, die strategische Vision einer Person zu schärfen, die \
sich selbständig macht. Die Person gibt einen groben, evtl. unstrukturierten Text zu ihrer Vision. \
Formuliere daraus 2-4 klare, prägnante Sätze, die den Kern erfassen - nicht länger, nicht \
ausschmückender als nötig. Antworte NUR mit dem geschärften Text, ohne Anführungszeichen, ohne \
zusätzliche Erklärung."""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Auth-Endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/signup")
def signup(payload: SignupIn):
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Ungültige E-Mail-Adresse.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Passwort muss mindestens 8 Zeichen haben.")

    with get_db() as conn:
        existing = run_query(conn, "SELECT id FROM users WHERE email = ?", (email,))
        if existing:
            raise HTTPException(status_code=409, detail="Diese E-Mail ist bereits registriert.")

        password_hash = hash_password(payload.password)
        user_id = run_write(
            conn,
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, password_hash, now_iso()),
        )

    token = create_token(user_id, email)
    return {"token": token, "email": email}


@app.post("/auth/login")
def login(payload: LoginIn):
    email = payload.email.strip().lower()
    with get_db() as conn:
        rows = run_query(conn, "SELECT * FROM users WHERE email = ?", (email,))

    if not rows or not verify_password(payload.password, rows[0]["password_hash"]):
        raise HTTPException(status_code=401, detail="E-Mail oder Passwort falsch.")

    user = rows[0]
    token = create_token(user["id"], user["email"])
    return {"token": token, "email": user["email"]}


# ---------------------------------------------------------------------------
# Endpoints — jeweils mit user = Depends(get_current_user) geschützt,
# jede Anfrage ist automatisch auf die Daten der eingeloggten Person beschränkt
# ---------------------------------------------------------------------------

@app.get("/entries")
def list_entries(type: Optional[str] = None, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        return fetch_entries(conn, user["user_id"], type)


@app.post("/entries")
def create_entry(entry: EntryIn, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        new_id = run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, due_date, created_at) VALUES (?, ?, ?, FALSE, ?, ?)",
            (user["user_id"], entry.type, entry.content, entry.due_date, now_iso()),
        )
        return {"id": new_id}


@app.patch("/entries/{entry_id}")
def update_entry(entry_id: int, update: EntryUpdate, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        if update.done is not None:
            run_write(
                conn,
                "UPDATE entries SET done = ? WHERE id = ? AND user_id = ?",
                (update.done, entry_id, user["user_id"]),
            )
        if update.content is not None:
            run_write(
                conn,
                "UPDATE entries SET content = ? WHERE id = ? AND user_id = ?",
                (update.content, entry_id, user["user_id"]),
            )
        return {"ok": True}


@app.delete("/entries/{entry_id}")
def delete_entry(entry_id: int, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        run_write(conn, "DELETE FROM entries WHERE id = ? AND user_id = ?", (entry_id, user["user_id"]))
        return {"ok": True}


def due_date_from_label(label: Optional[str]) -> Optional[str]:
    """Wandelt 'heute'/'morgen'/'diese_woche'/None in ein echtes ISO-Datum um."""
    today = datetime.now(timezone.utc).date()
    if label == "heute":
        return today.isoformat()
    if label == "morgen":
        return (today + timedelta(days=1)).isoformat()
    if label == "diese_woche":
        return (today + timedelta(days=3)).isoformat()  # grobe Mitte der Woche
    return None


@app.post("/chat")
async def chat(payload: ChatIn, user: dict = Depends(get_current_user)):
    import json

    with get_db() as conn:
        memory = build_memory_context(conn, user["user_id"])
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'chat_user', ?, FALSE, ?)",
            (user["user_id"], payload.message, now_iso()),
        )

    system_prompt = f"{MENTOR_SYSTEM_PROMPT}\n\n--- Bisherige Historie der Person ---\n{memory}"
    raw = await call_claude(system_prompt, payload.message)

    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        antwort = parsed.get("antwort", raw)
        neue_aufgaben = parsed.get("neue_aufgaben", [])
    except (json.JSONDecodeError, AttributeError):
        # Falls das Parsen fehlschlägt, nutzen wir die Rohantwort ohne Aufgaben-Extraktion,
        # damit der Chat trotzdem funktioniert, statt komplett zu scheitern.
        antwort = raw
        neue_aufgaben = []

    with get_db() as conn:
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'chat_assistant', ?, FALSE, ?)",
            (user["user_id"], antwort, now_iso()),
        )
        erstellte_aufgaben = []
        for aufgabe in neue_aufgaben:
            inhalt = aufgabe.get("inhalt", "") if isinstance(aufgabe, dict) else str(aufgabe)
            faellig_label = aufgabe.get("faellig") if isinstance(aufgabe, dict) else None
            if not inhalt:
                continue
            due = due_date_from_label(faellig_label)
            run_write(
                conn,
                "INSERT INTO entries (user_id, type, content, done, due_date, created_at) VALUES (?, 'task', ?, FALSE, ?, ?)",
                (user["user_id"], inhalt, due, now_iso()),
            )
            erstellte_aufgaben.append(inhalt)

    return {"answer": antwort, "neue_aufgaben": erstellte_aufgaben}


# ---------------------------------------------------------------------------
# Strategy-Endpoints — Vision + Projekte, der "Kompass"
# ---------------------------------------------------------------------------

@app.get("/strategy")
def get_strategy(user: dict = Depends(get_current_user)):
    with get_db() as conn:
        vision = fetch_entries(conn, user["user_id"], "vision", limit=1)
        projects = fetch_entries(conn, user["user_id"], "project", limit=20)
    return {
        "vision": vision[0]["content"] if vision else "",
        "projects": projects,
    }


@app.post("/strategy/vision")
async def set_vision(payload: GoalIn, user: dict = Depends(get_current_user)):
    """Nimmt einen groben Vision-Text entgegen, lässt ihn von Claude schärfen, speichert ihn."""
    refined = await call_claude(STRATEGY_SYSTEM_PROMPT, payload.goal)
    with get_db() as conn:
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'vision', ?, FALSE, ?)",
            (user["user_id"], refined, now_iso()),
        )
    return {"vision": refined}


@app.post("/strategy/project")
def add_project(entry: EntryIn, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        new_id = run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'project', ?, FALSE, ?)",
            (user["user_id"], entry.content, now_iso()),
        )
    return {"id": new_id}


@app.delete("/strategy/project/{project_id}")
def delete_project(project_id: int, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        run_write(
            conn, "DELETE FROM entries WHERE id = ? AND user_id = ?", (project_id, user["user_id"])
        )
    return {"ok": True}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_configured": bool(ANTHROPIC_API_KEY),
        "jwt_secret_configured": bool(JWT_SECRET),
    }
