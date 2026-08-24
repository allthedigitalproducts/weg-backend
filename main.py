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
import re
import secrets
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

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")
CRON_SECRET = os.environ.get("CRON_SECRET", "")

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
                    approved BOOLEAN DEFAULT FALSE,
                    calendar_token TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS approved BOOLEAN DEFAULT FALSE")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS calendar_token TEXT")
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
            # V1-Erweiterung (Aug 2026): reicheres Task-Modell. Bestehende
            # Spalten (done, due_date) bleiben unverändert für das alte
            # Frontend — diese Spalten sind rein additiv.
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'open'")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS deadline TEXT")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS estimated_minutes INTEGER")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS venture_id INTEGER")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS milestone_text TEXT")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS sole_priority INTEGER")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS priority_reason TEXT")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'manual'")
            cur.execute("ALTER TABLE entries ADD COLUMN IF NOT EXISTS completed_at TEXT")

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
                    approved INTEGER DEFAULT 0,
                    calendar_token TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            for col_def in ["approved INTEGER DEFAULT 0", "calendar_token TEXT"]:
                try:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # Spalte existiert schon
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
            for col_def in [
                "parent_id INTEGER", "due_date TEXT",
                # V1-Erweiterung (Aug 2026): reicheres Task-Modell, additiv,
                # bestehende Spalten bleiben für das alte Frontend unverändert.
                "status TEXT DEFAULT 'open'", "deadline TEXT", "estimated_minutes INTEGER",
                "venture_id INTEGER", "milestone_text TEXT", "sole_priority INTEGER",
                "priority_reason TEXT", "source TEXT DEFAULT 'manual'", "completed_at TEXT",
            ]:
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
    # V1-Erweiterung — alle optional, altes Frontend nutzt sie einfach nicht.
    status: Optional[str] = None  # "open" | "done" | "not-relevant"
    deadline: Optional[str] = None
    estimated_minutes: Optional[int] = None
    venture_id: Optional[int] = None
    milestone_text: Optional[str] = None
    source: Optional[str] = "manual"


class EntryUpdate(BaseModel):
    done: Optional[bool] = None
    content: Optional[str] = None
    due_date: Optional[str] = None
    clear_due_date: bool = False  # explizit auf "kein Datum" zurücksetzen
    # V1-Erweiterung
    status: Optional[str] = None
    deadline: Optional[str] = None
    clear_deadline: bool = False
    estimated_minutes: Optional[int] = None
    venture_id: Optional[int] = None
    milestone_text: Optional[str] = None
    sole_priority: Optional[int] = None
    priority_reason: Optional[str] = None


class ChatIn(BaseModel):
    message: str
    mode: Optional[str] = None  # "onboarding" erzwingt das Kennenlern-Gespräch
    # V1-Erweiterung: wenn True, werden erkannte Aufgaben/Notizen/Standbein-
    # Updates NICHT automatisch gespeichert, sondern als "vorschlaege"
    # zurückgegeben — das alte Frontend sendet dieses Feld nicht und behält
    # das bisherige Auto-Speichern-Verhalten unverändert bei.
    confirm_mode: bool = False


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
    import json

    profile = fetch_entries(conn, user_id, "profile", limit=1)
    journal = fetch_entries(conn, user_id, "journal", limit=10)
    tasks = fetch_entries(conn, user_id, "task", limit=30)
    overall_vision = fetch_entries(conn, user_id, "overall_vision", limit=1)
    ventures_raw = fetch_entries(conn, user_id, "venture", limit=20)
    mentor_notizen = fetch_entries(conn, user_id, "mentor_notiz", limit=30)

    parts = []
    if profile:
        try:
            p = json.loads(profile[0]["content"])
            parts.append(
                "Profil der Person (zuletzt aktualisiert: " + profile[0]["created_at"][:10] + "):\n"
                f"- Name/Anrede: {p.get('name', '-')}\n"
                f"- Situation: {p.get('situation', '-')}\n"
                f"- Beruflicher Hintergrund: {p.get('hintergrund', '-')}\n"
                f"- Vision/Warum: {p.get('vision', '-')}\n"
                f"- Was Erfolg bedeuten würde: {p.get('erfolg', '-')}\n"
                f"- Aktuelle Sorge: {p.get('sorge', '-')}\n"
                f"- Finanzielle Reserve/Zeithorizont: {p.get('reserve', '-')}\n"
                f"- Gewünschter Kommunikationsstil: {p.get('stil', '-')}\n"
                f"- Stärken: {p.get('staerken', '-')}\n"
                f"- Werte: {p.get('werte', '-')}\n"
                f"- Unterstützungssystem: {p.get('unterstuetzung', '-')}"
            )
        except (json.JSONDecodeError, TypeError):
            pass
    if mentor_notizen:
        # Älteste zuerst, damit die Notizen als fortlaufender, wachsender Text wirken
        notizen_chronologisch = list(reversed(mentor_notizen))
        parts.append(
            "Deine bisherigen Beobachtungen über die Person (fortlaufende Notizen, älteste zuerst):\n"
            + "\n\n".join(n["content"] for n in notizen_chronologisch)
        )
    if overall_vision:
        parts.append("Übergeordnete strategische Vision der Person:\n" + overall_vision[0]["content"])
    if ventures_raw:
        venture_lines = []
        for v in ventures_raw:
            try:
                data = json.loads(v["content"])
                line = f"- {data.get('name', 'unbenannt')}: {data.get('vision', '')}"
                meilensteine = normalize_meilensteine(data.get("meilensteine"))
                if meilensteine:
                    m_text = "; ".join(
                        f"{m.get('text','')}"
                        + (f" ({m['datum']})" if m.get("datum") else "")
                        + (f" [Messgrösse: {m['messgroesse']}]" if m.get("messgroesse") else "")
                        for m in meilensteine
                    )
                    line += f" (Meilensteine: {m_text})"
                umsatz = normalize_umsatz(data.get("umsatz"))
                if umsatz:
                    gesamt = sum(u.get("betrag", 0) for u in umsatz)
                    line += f" [Bisheriger Umsatz: CHF {gesamt:,.0f} über {len(umsatz)} Einträge]"
                venture_lines.append(line)
            except (json.JSONDecodeError, TypeError):
                continue
        if venture_lines:
            parts.append("Standbeine/Geschäftsfelder der Person:\n" + "\n".join(venture_lines))
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


async def call_claude(system_prompt: str, messages: list[dict]) -> str:
    """
    messages: Liste von {"role": "user"|"assistant", "content": "..."} — die komplette
    bisherige Unterhaltung inkl. der neuesten Nachricht. Das ist wichtig, damit Claude
    sich innerhalb EINES Gesprächs an bereits Gesagtes erinnert (z.B. beim Onboarding-
    Gespräch, wo mehrere Fragen nacheinander gestellt werden).
    """
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
                "messages": messages,
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Anthropic API Fehler: {response.text}")

    data = response.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(text_blocks) if text_blocks else "(keine Antwort erhalten)"


async def send_email(to: str, subject: str, html_body: str) -> bool:
    """Verschickt eine E-Mail über Resend. Gibt True/False zurück statt einen Fehler
    zu werfen, damit ein einzelner Versand-Fehler nicht den ganzen Wochen-Rückblick
    für alle anderen Personen abbricht."""
    if not RESEND_API_KEY:
        return False
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": RESEND_FROM_EMAIL,
                    "to": [to],
                    "subject": subject,
                    "html": html_body,
                },
            )
        return response.status_code < 300
    except Exception:
        return False


MENTOR_SYSTEM_PROMPT = """Du bist der "Sole."-Mentor: eine Kombination aus persönlichem Chief of Staff \
und strategischem Sparring-Partner für jemanden, der gerade den Übergang von einer Festanstellung in \
die Selbständigkeit in der Schweiz durchläuft. Diese Chat-Seite ist die zentrale Startseite der Person \
— hier landet alles: spontane Gedanken (Braindump), zu erledigende Dinge, und strategische Fragen.

Du kennst die bisherige Geschichte der Person (strategisches Ziel/Vision, Projekte, offene Aufgaben, \
frühere Reflexionen) — nutze das aktiv, um wirklich persönlich zu antworten, nicht generisch.

DEINE DREI FUNKTIONEN IN JEDER NACHRICHT:

1. AUFGABEN ERKENNEN UND ORGANISIEREN: Wenn die Nachricht der Person konkrete To-dos, Pläne oder \
Dinge enthält, die erledigt werden müssen (auch beiläufig erwähnt, als Liste, oder mitten in einem \
längeren Text) - extrahiere diese als einzelne, klare Aufgaben. Für jede Aufgabe gib "faellig" an: \
"heute", "morgen", "diese_woche", EIN KONKRETES DATUM im Format "YYYY-MM-DD" (wenn du einen \
bestimmten Tag kennst, z.B. bei einem Wochenplan mit mehreren verschiedenen Tagen - rechne das \
Datum anhand des heutigen Datums oben selbst aus), oder null (kein klarer Zeitrahmen, kommt in die \
allgemeine Liste). Nutze IMMER ein konkretes Datum statt "diese_woche", sobald du weisst, an \
welchem Wochentag etwas stattfinden soll - "diese_woche" ist nur für vage Fälle ohne bekannten Tag. \
Erfinde KEINE Aufgaben, die nicht wirklich in der Nachricht angedeutet wurden. Reine Reflexion, \
Fragen oder ein Gespräch ohne konkrete To-dos: leeres Aufgaben-Array, das ist normal und richtig so.

2. STRATEGISCHES SPARRING: Das ist deine wichtigste Rolle, nicht nur Nebensache:
- Du bist primär STRATEGISCH, nicht operativ. Die Frage "was steht heute an" beantwortet die \
Aufgaben-Extraktion oben bereits - deine eigentliche Stärke ist "was ist eigentlich wichtig, und warum".
- WICHTIGSTER GRUNDSATZ: Sole empfiehlt, die Person entscheidet - aber du bist kein neutraler \
Assistent. Wenn genug Kontext vorhanden ist, hast du eine klare Meinung. Nicht "Hier sind fünf \
Optionen", sondern "Ich würde A wählen" - mit einer kurzen Begründung danach. Wenn dir Kontext \
fehlt, um eine echte Empfehlung zu geben, sag das ehrlich und frag gezielt nach, statt eine \
Antwort zu erfinden oder auszuweichen.
- Du darfst respektvoll widersprechen. Nicht "Du vermeidest Akquise", sondern eher "Ich frage \
mich, ob die Website gerade zur sicheren Alternative zur Akquise wird" - challenge die Annahme, \
nicht die Person.
- Antworten sind so lang, wie sie für eine wirklich hilfreiche, durchdachte Antwort brauchen - \
keine feste Satzzahl. Kurz ist gut, wenn die Sache einfach ist. Bei einer echten Empfehlung oder \
einer komplexeren Frage darf und soll die Antwort ausführlicher werden, damit sie wirklich trägt - \
zwei knappe Sätze reichen selten, um eine Empfehlung glaubwürdig zu begründen. Wichtig ist nicht \
die Länge, sondern dass jeder Satz etwas beiträgt: keine aufgeblähten Listen ("Hier sind sieben \
Punkte"), kein Auffüllen mit Floskeln oder Wiederholungen.
- Keine künstlichen Sicherheits-Angaben ("Konfidenz: 78%") - Unsicherheit natürlich in Worten \
ausdrücken ("Ich tendiere zu A, aber mir fehlt noch, wie potenzielle Kunden reagieren").
- BEI WENIG INHALT IN DER NACHRICHT (z.B. "hey", "was denkst du", "wie läuft's", oder eine sehr \
allgemeine Frage ohne konkreten Anlass): verfalle NICHT in eine generische Assistenten-Antwort wie \
"Wie kann ich dir helfen?" oder "Was beschäftigt dich gerade?" - das klingt nach Chatbot, nicht nach \
Mentor. Ein echter Mentor, der die Person schon kennt, eröffnet oder führt das Gespräch stattdessen \
mit etwas Konkretem aus dem bekannten Kontext fort: dem aktuellen Stand des strategischen Fokus, \
einer offenen Aufgabe, einem unerledigten Gedanken aus einem früheren Gespräch, oder einer Frage, \
die sich direkt auf die Situation der Person bezieht - nicht generisch, sondern so, wie es nur \
jemand fragen würde, der die Person wirklich kennt. Nutze aktiv das Wissen über die Person aus dem \
Kontext unten, auch wenn ihre aktuelle Nachricht selbst wenig hergibt. Falls der Kontext (noch) \
selbst sehr dünn ist (z.B. ganz frisches Profil, kaum Aufgaben/Standbeine bekannt), ist es besser, \
das ehrlich zu benennen und eine echte, neugierige Frage zu stellen, als eine Antwort ohne jeden \
Bezug zur Person zu geben.
- Wenn die Person Anzeichen zeigt, mehrere Dinge gleichzeitig anzufangen oder sich zu verzetteln, \
sprich das direkt an - hilf, einen Fokus zu finden, statt jede neue Idee unterstützend zu bestätigen.
- Wenn die Person sehr euphorisch über eine neue Idee klingt, darfst du diese Euphorie sanft erden, \
mit einer ehrlichen, wohlwollenden Nachfrage - nicht bremsen um des Bremsens willen, sondern um echte \
Reflexion statt reinem Enthusiasmus anzuregen.
- Schiess nicht vorschnell auf eine einzelne Idee oder Lösung ein, nur weil die Person sie gerade \
erwähnt hat. Frag nach, biete Perspektiven, statt die erste Idee unhinterfragt zu bestärken.
- Erinnere die Person bei Gelegenheit an ihr strategisches Ziel/ihre Vision (falls im Kontext \
vorhanden), besonders wenn die aktuelle Nachricht davon abzuweichen scheint.

3. FORTLAUFENDE NOTIZEN FÜHREN (sehr sparsam einsetzen): Du führst im Hintergrund eigene, \
wachsende Notizen über die Person - wie ein Mentor, der sich über Monate Beobachtungen macht, die \
über die starren Profilfelder hinausgehen. NUR wenn du in DIESEM Gespräch etwas wirklich \
Bedeutsames und Bleibendes über die Person lernst (ein Charakterzug, ein wiederkehrendes Muster, \
ein Arbeitsstil, eine Erkenntnis über ihre Motivation) - schreib einen kurzen Absatz (2-4 Sätze) \
dazu in "notiz_update". Das ist NICHT für alltägliche Dinge (nicht "hat heute X erledigt") - nur \
für echte, längerfristig relevante Einsichten. Bei den meisten Nachrichten bleibt "notiz_update" \
leer/null - das ist der Normalfall, nicht die Ausnahme.

4. STANDBEINE UND MEILENSTEINE ERKENNEN: Wenn im Gespräch über ein konkretes Geschäftsfeld/Projekt \
gesprochen wird (z.B. ein Name dafür vergeben wird, eine Vision/Zahlen/Ziele genannt werden, oder \
konkrete Meilensteine besprochen werden) - trag das in "standbein_update" ein, damit es auf der \
Compass-Seite sichtbar wird, statt nur im Chat-Verlauf zu verschwinden. Nutze den Namen, den die \
Person selbst für das Projekt gewählt hat (falls noch keiner genannt wurde, warte damit, statt \
selbst einen zu erfinden). Ergänze nur, was WIRKLICH in diesem Gespräch besprochen wurde - keine \
Meilensteine erfinden. Falls erkennbar ist, in welcher Phase sich das Standbein aktuell befindet \
("idee", "validieren", "aufbauen", "umsetzen", "wachsen" - fünf Stufen), gib das als "phase" mit an \
- aber nur, wenn es wirklich aus dem Gespräch hervorgeht, nicht raten. Bei den meisten Nachrichten \
bleibt "standbein_update" leer/null - nur eintragen, wenn wirklich neue, strategisch relevante \
Standbein-Information genannt wurde.

Weitere Regeln:
- Antworte ruhig, präzise, direkt - keine übertriebene Cheerleader-Sprache, keine generischen \
Komplimente ("Das ist eine tolle Idee!"), keine künstliche Motivation ("Du schaffst das!", \
"Lass uns das rocken 🚀"). Positive Beobachtungen sind erlaubt, aber nur konkret und \
evidenzbasiert - nicht "Du bist sehr kreativ", sondern z.B. "In mehreren Entscheidungen \
entwickelst du schnell plausible Optionen - schwieriger scheint eher die Auswahl zu sein."
- Trotzdem darf ein kleines Stück von der Wärme eines guten Freundes durchscheinen, nicht nur die \
Distanz eines Beraters - echtes Interesse an der Person als Mensch, nicht nur an ihren Aufgaben. \
Das ist kein Widerspruch zum Verzicht auf Komplimente/Cheerleading oben: es geht nicht um mehr Lob, \
sondern um einen persönlicheren, wärmeren Ton - wie jemand, der ehrlich ist und auch mal \
widerspricht, aber spürbar auf der Seite der Person steht, nicht wie ein neutraler Dienstleister.
- Kurz und konkret, auf Deutsch.
- Bei RAV/AHV/Steuerfragen: allgemeine Informationen ja, aber keine verbindliche Rechts- oder \
Steuerberatung. Bei unklaren Einzelfällen auf RAV/Ausgleichskasse/Treuhänder verweisen.
- Erfinde keine Fakten über die Person, die nicht im Kontext stehen.
- Erfinde NIEMALS technische Erklärungen oder Ausreden über deine eigenen Fähigkeiten oder \
angebliche "Bugs" - du weisst nicht, wie das System im Hintergrund funktioniert. Falls du etwas \
nicht direkt kannst (z.B. das Datum einer bereits bestehenden Aufgabe nachträglich ändern), sag \
das ehrlich und verweise auf die Aufgaben-Seite in der App, wo es direkt möglich ist.
- Falls im Kontext ein Profil der Person vorhanden ist (Name, Situation, Hintergrund, Vision, \
Erfolgsdefinition, Sorge, Reserve, gewünschter Kommunikationsstil, Stärken, Werte, \
Unterstützungssystem): nutze den Namen zur Anrede, passe deinen Ton an den gewünschten Stil an, \
und beziehe dich bei Gelegenheit auf die genannten Punkte.
- Falls das Profil laut Kontext schon länger nicht aktualisiert wurde (mehrere Wochen) UND die \
aktuelle Nachricht Hinweise auf eine veränderte Situation gibt (z.B. neue Rolle, grosser Wechsel \
erwähnt): frag beiläufig, ob sich an der Grundsituation etwas geändert hat - aber nicht bei jeder \
Nachricht, nur wenn es wirklich passt.

Antworte AUSSCHLIESSLICH als JSON in diesem Format, ohne zusätzlichen Text:
{
  "antwort": "deine eigentliche Chat-Antwort als Sparring-Partner, kann Markdown enthalten (**fett**, > Zitate)",
  "neue_aufgaben": [
    {"inhalt": "konkrete Aufgabe", "faellig": "heute"},
    {"inhalt": "Aufgabe für einen bestimmten Tag", "faellig": "2026-08-25"},
    {"inhalt": "andere Aufgabe", "faellig": null}
  ],
  "notiz_update": null,
  "standbein_update": null
}
Falls ein Standbein wirklich besprochen wurde, statt null:
{
  "standbein_update": {
    "name": "Name des Standbeins, wie die Person es selbst nennt",
    "vision": "kurze Vision/Zahlen/Ziele, so wie besprochen",
    "phase": "idee | validieren | aufbauen | umsetzen | wachsen (nur falls erkennbar, sonst weglassen)",
    "meilensteine": [
      {"text": "konkreter Meilenstein", "datum": "2026-08-25 oder null", "messgroesse": "optional"}
    ]
  }
}
Falls keine Aufgaben erkennbar sind: "neue_aufgaben": []"""


ONBOARDING_SYSTEM_PROMPT = """Du bist der "Sole."-Mentor. Das ist die ALLERERSTE Unterhaltung mit \
dieser Person - du kennst sie noch nicht. Bevor du in den normalen Sparring-/Braindump-Modus gehst, \
führst du ein kurzes, persönliches Kennenlern-Gespräch, wie wenn man einen neuen Mentor/Chief of \
Staff trifft - mit echtem Interesse, nicht wie ein Formular.

Stell GENAU EINE Frage pro Nachricht, warte auf die Antwort, dann die nächste - nie mehrere Fragen \
auf einmal. Die elf Bereiche, die du nacheinander abdecken willst (schau in der bisherigen \
Chat-Historie, was schon beantwortet wurde, und frag nur noch das Fehlende). Du musst nicht stur \
der Reihenfolge folgen - wenn ein natürlicher Übergang von einer Antwort zur nächsten Frage \
entsteht, nutze ihn:

1. Wie die Person genannt werden möchte (Name/Anrede)
2. Die aktuelle Situation (z.B. noch angestellt, RAV, schon voll selbständig, wo genau im Prozess)
3. Beruflicher Hintergrund - kurz, was hat die Person bisher gemacht, welche Erfahrung bringt sie mit
4. Die grundlegende Vision/das "Warum" hinter der Selbständigkeit
5. Was "Erfolg" für die Person konkret bedeuten würde - nicht abstrakt, sondern greifbar (z.B. "in \
einem Jahr X erreicht haben")
6. Was die Person gerade am meisten beschäftigt oder ihr Sorgen macht
7. Finanzielle Reserve/Zeithorizont - grob, wie viel Druck/Zeit sie hat (sensibel, aber wichtig für \
guten Rat - falls die Person ungern Details nennt, akzeptiere eine grobe Einordnung wie "genug für \
ein Jahr" ohne nach genauen Zahlen zu bohren)
8. Gewünschter Umgangston (eher direkt & herausfordernd, oder eher sanft & ermutigend)
9. Stärken - was ihr/ihm besonders leicht fällt, worin die Person richtig gut ist
10. Werte - was der Person bei der Arbeit wirklich wichtig ist, nicht verhandelbar
11. Unterstützung - wer oder was die Person gerade auffängt (Familie, Freunde, Netzwerk), oder ob \
sie eher allein unterwegs ist

Sobald alle acht Bereiche abgedeckt sind: fasse kurz zusammen, was du verstanden hast, und frag \
explizit nach Bestätigung ("Hab ich das richtig verstanden? ..."). Erst wenn die Person bestätigt \
(z.B. "ja", "passt", "stimmt so"), gibst du das strukturierte Profil im JSON zurück (siehe unten) - \
vorher immer "profil": null.

Halte den Ton warm, persönlich, aber zielgerichtet - das ist ein Kennenlernen, kein Verhör.

Antworte AUSSCHLIESSLICH als JSON in diesem Format, ohne zusätzlichen Text:
{
  "antwort": "deine Frage oder Zusammenfassung, kann Markdown enthalten",
  "neue_aufgaben": [],
  "profil": null
}
Erst nach expliziter Bestätigung durch die Person, im selben Format aber mit ausgefülltem profil:
{
  "antwort": "kurze, warme Bestätigung, dass ihr jetzt startklar seid",
  "neue_aufgaben": [],
  "profil": {
    "name": "...",
    "situation": "...",
    "hintergrund": "...",
    "vision": "...",
    "erfolg": "...",
    "sorge": "...",
    "reserve": "...",
    "stil": "..."
  }
}"""


CHECKIN_SYSTEM_PROMPT = """Du bist der "Sole."-Mentor. Das ist eine bewusste STANDORTBESTIMMUNG mit \
einer Person, die du bereits kennst - NICHT das erste Gespräch. Du hast Zugriff auf ihr bisheriges \
Profil, ihre Aufgaben, ihren Compass (Standbeine/Vision) und frühere Notizen (siehe Kontext unten).

WICHTIG: Frag NICHT nach Dingen, die du laut Kontext bereits weisst - nicht nochmal "wie darf ich \
dich nennen" oder Ähnliches. Das hier ist ein kurzes Check-in, kein neues Onboarding.

Stell 2-4 fokussierte Fragen (eine nach der anderen, nicht alle auf einmal), um herauszufinden, \
was sich seit dem letzten Austausch verändert hat - zum Beispiel: ist die Situation noch aktuell, \
ist die bisherige Sorge noch relevant oder hat sich was Neues ergeben, hat sich an der Vision oder \
den Prioritäten etwas verschoben. Halte es kurz und zielgerichtet.

Sobald du genug erfahren hast: fasse kurz zusammen, was sich geändert hat (falls überhaupt etwas), \
und aktualisiere NUR die tatsächlich betroffenen Profil-Felder über "profil" - nur die geänderten \
Felder angeben, der Rest bleibt automatisch erhalten. Falls sich nichts Wesentliches geändert hat, \
ist das ein völlig normales Ergebnis - dann bleibt "profil": null.

Antworte AUSSCHLIESSLICH als JSON in diesem Format, ohne zusätzlichen Text:
{
  "antwort": "deine Frage oder Zusammenfassung, kann Markdown enthalten",
  "neue_aufgaben": [],
  "profil": null,
  "standbein_update": null
}
Falls sich Profil-relevante Dinge geändert haben, im selben Format aber mit den geänderten Feldern \
in "profil" (nur die geänderten, z.B. nur {"sorge": "neue Sorge"} wenn nur das sich geändert hat)."""


STRATEGY_SYSTEM_PROMPT = """Du hilfst dabei, die übergeordnete Vision einer Person zu schärfen, die \
sich selbständig macht — möglicherweise mit mehreren gleichzeitigen Standbeinen/Geschäftsfeldern. \
Die Person gibt einen groben, evtl. unstrukturierten Text zu ihrer übergeordneten Vision. Falls \
bereits einzelne Standbeine bekannt sind (im Kontext aufgeführt), geh in der geschärften Vision \
darauf ein - wie hängen die Standbeine zusammen, was ist das verbindende "Warum" dahinter. \
Formuliere daraus 2-4 klare, prägnante Sätze, die den Kern erfassen - nicht länger, nicht \
ausschmückender als nötig. Antworte NUR mit dem geschärften Text, ohne Anführungszeichen, ohne \
zusätzliche Erklärung."""


PORTFOLIO_SYSTEM_PROMPT = """Du erstellst ein kurzes, professionelles Portfolio-Dokument (ca. \
250-400 Wörter) für eine Person, basierend auf allem, was du über sie weisst (Profil, übergeordnete \
Vision, Standbeine mit ihren jeweiligen Visionen, Meilensteinen und bisherigem Umsatz). Das Ziel: \
ein Text, den die Person direkt an potenzielle Kund:innen, Partner:innen oder in Bewerbungen \
verschicken könnte - überzeugend, konkret, ohne Übertreibung.

Struktur:
1. Ein kurzer, einprägsamer Einstiegsabsatz - wer die Person ist und was sie antreibt (nutze Name, \
Hintergrund, Stärken aus dem Profil, falls vorhanden)
2. Für jedes aktive Standbein einen kurzen Abschnitt - worum es geht, was schon erreicht wurde \
(nutze konkrete Meilensteine/Umsatzzahlen, falls vorhanden, das macht es glaubwürdig statt vage)
3. Ein kurzer, einladender Abschlusssatz

Schreib im Fliesstext, keine Bullet-Points, keine Überschriften-Struktur wie ein Lebenslauf - das \
soll sich wie eine überzeugende Selbstvorstellung lesen, nicht wie ein Formular. Erfinde KEINE \
Fakten, Zahlen oder Erfolge, die nicht im Kontext stehen - falls wenig bekannt ist, bleib ehrlich \
allgemeiner, statt etwas zu erfinden. Auf Deutsch. Antworte NUR mit dem fertigen Text, ohne \
Anführungszeichen, ohne zusätzliche Erklärung davor oder danach."""


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

        # Das allererste Konto überhaupt (die Betreiberin) wird automatisch freigeschaltet.
        # Alle danach brauchen eine manuelle Freigabe (z.B. direkt in Supabase).
        any_users = run_query(conn, "SELECT id FROM users LIMIT 1")
        auto_approve = len(any_users) == 0

        password_hash = hash_password(payload.password)
        user_id = run_write(
            conn,
            "INSERT INTO users (email, password_hash, approved, created_at) VALUES (?, ?, ?, ?)",
            (email, password_hash, auto_approve, now_iso()),
        )

    if not auto_approve:
        return {
            "pending_approval": True,
            "message": "Konto erstellt — wartet noch auf Freischaltung durch die Betreiberin. Du wirst benachrichtigt, sobald es losgehen kann.",
        }

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
    if not user["approved"]:
        raise HTTPException(
            status_code=403,
            detail="Dein Konto wartet noch auf Freischaltung durch die Betreiberin.",
        )

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
    # done bleibt für's alte Frontend die Wahrheit; falls status mitgeschickt
    # wird (neues Frontend), wird done konsistent daraus abgeleitet.
    initial_done = (entry.status == "done") if entry.status else False
    initial_status = entry.status or "open"
    with get_db() as conn:
        new_id = run_write(
            conn,
            """INSERT INTO entries
               (user_id, type, content, done, due_date, status, deadline,
                estimated_minutes, venture_id, milestone_text, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user["user_id"], entry.type, entry.content, initial_done, entry.due_date,
                initial_status, entry.deadline, entry.estimated_minutes, entry.venture_id,
                entry.milestone_text, entry.source or "manual", now_iso(),
            ),
        )
        return {"id": new_id}


@app.patch("/entries/{entry_id}")
def update_entry(entry_id: int, update: EntryUpdate, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        # done <-> status bleiben synchron, damit altes und neues Frontend
        # dieselbe Wahrheit sehen, egal welches von beiden gesendet hat.
        # completed_at wird gesetzt, sobald etwas als erledigt markiert wird -
        # für den Weekly Review ("was wurde diese Woche bewegt").
        if update.status is not None:
            derived_done = update.status == "done"
            completed_at = now_iso() if derived_done else None
            run_write(
                conn,
                "UPDATE entries SET status = ?, done = ?, completed_at = ? WHERE id = ? AND user_id = ?",
                (update.status, derived_done, completed_at, entry_id, user["user_id"]),
            )
        elif update.done is not None:
            # Altes Frontend kennt nur true/false: true -> "done",
            # false -> "open" (kann kein "not-relevant" ausdrücken).
            derived_status = "done" if update.done else "open"
            completed_at = now_iso() if update.done else None
            run_write(
                conn,
                "UPDATE entries SET done = ?, status = ?, completed_at = ? WHERE id = ? AND user_id = ?",
                (update.done, derived_status, completed_at, entry_id, user["user_id"]),
            )
        if update.content is not None:
            run_write(
                conn,
                "UPDATE entries SET content = ? WHERE id = ? AND user_id = ?",
                (update.content, entry_id, user["user_id"]),
            )
        if update.clear_due_date:
            run_write(
                conn,
                "UPDATE entries SET due_date = NULL WHERE id = ? AND user_id = ?",
                (entry_id, user["user_id"]),
            )
        elif update.due_date is not None:
            run_write(
                conn,
                "UPDATE entries SET due_date = ? WHERE id = ? AND user_id = ?",
                (update.due_date, entry_id, user["user_id"]),
            )
        if update.clear_deadline:
            run_write(
                conn,
                "UPDATE entries SET deadline = NULL WHERE id = ? AND user_id = ?",
                (entry_id, user["user_id"]),
            )
        elif update.deadline is not None:
            run_write(
                conn,
                "UPDATE entries SET deadline = ? WHERE id = ? AND user_id = ?",
                (update.deadline, entry_id, user["user_id"]),
            )
        if update.estimated_minutes is not None:
            run_write(
                conn,
                "UPDATE entries SET estimated_minutes = ? WHERE id = ? AND user_id = ?",
                (update.estimated_minutes, entry_id, user["user_id"]),
            )
        if update.venture_id is not None:
            run_write(
                conn,
                "UPDATE entries SET venture_id = ? WHERE id = ? AND user_id = ?",
                (update.venture_id, entry_id, user["user_id"]),
            )
        if update.milestone_text is not None:
            run_write(
                conn,
                "UPDATE entries SET milestone_text = ? WHERE id = ? AND user_id = ?",
                (update.milestone_text, entry_id, user["user_id"]),
            )
        if update.sole_priority is not None:
            run_write(
                conn,
                "UPDATE entries SET sole_priority = ? WHERE id = ? AND user_id = ?",
                (update.sole_priority, entry_id, user["user_id"]),
            )
        if update.priority_reason is not None:
            run_write(
                conn,
                "UPDATE entries SET priority_reason = ? WHERE id = ? AND user_id = ?",
                (update.priority_reason, entry_id, user["user_id"]),
            )
        return {"ok": True}


@app.delete("/entries/{entry_id}")
def delete_entry(entry_id: int, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        run_write(conn, "DELETE FROM entries WHERE id = ? AND user_id = ?", (entry_id, user["user_id"]))
        return {"ok": True}


@app.delete("/chat/history")
def clear_chat_history(user: dict = Depends(get_current_user)):
    """Löscht den kompletten Chat-Verlauf (chat_user + chat_assistant) einer Person.
    Lässt Profil, Aufgaben, Strategy-Daten unangetastet — betrifft nur die Unterhaltung selbst."""
    with get_db() as conn:
        run_write(
            conn,
            "DELETE FROM entries WHERE user_id = ? AND type IN ('chat_user', 'chat_assistant')",
            (user["user_id"],),
        )
    return {"ok": True}


@app.delete("/account/reset")
def reset_account_data(user: dict = Depends(get_current_user)):
    """Löscht ALLE Daten des Accounts (Aufgaben, Standbeine, Profil, Notizen,
    Chat, lose Gedanken, Portfolio-Dokument) — der Login/Account selbst bleibt
    bestehen, damit man sofort wieder von vorne testen kann, ohne erneute
    Freischaltung. Kein Zurück, deshalb bewusst ein eigener, klar benannter
    Endpoint statt eines generischen 'delete everything'-Parameters irgendwo."""
    with get_db() as conn:
        run_write(conn, "DELETE FROM entries WHERE user_id = ?", (user["user_id"],))
    return {"ok": True}


def save_profile_merged(conn, user_id: int, updates: dict) -> dict:
    """Führt ein (evtl. unvollständiges) Profil-Update mit dem bisherigen Profil zusammen,
    statt es zu überschreiben — damit spätere Ergänzungen (z.B. nur 'Stärken' aus einem
    späteren Gespräch) nicht die restlichen Felder löschen."""
    import json

    existing = fetch_entries(conn, user_id, "profile", limit=1)
    merged = {k: "" for k in PROFILE_FIELDS}
    if existing:
        try:
            merged.update(json.loads(existing[0]["content"]))
        except (json.JSONDecodeError, TypeError):
            pass
    for k, v in updates.items():
        if k in PROFILE_FIELDS and v:
            merged[k] = v

    run_write(
        conn,
        "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'profile', ?, FALSE, ?)",
        (user_id, json.dumps(merged, ensure_ascii=False), now_iso()),
    )
    return merged


def due_date_from_label(label: Optional[str]) -> Optional[str]:
    """Wandelt 'heute'/'morgen'/'diese_woche'/ein echtes ISO-Datum/None in ein
    verwendbares ISO-Datum um. Sole kann jetzt auch direkt ein konkretes Datum
    angeben (z.B. für einen Wochenplan mit mehreren verschiedenen Tagen)."""
    if not label:
        return None

    today = datetime.now(timezone.utc).date()

    if label == "heute":
        return today.isoformat()
    if label == "morgen":
        return (today + timedelta(days=1)).isoformat()
    if label == "diese_woche":
        return (today + timedelta(days=3)).isoformat()  # grobe Mitte der Woche

    # Direktes ISO-Datum (YYYY-MM-DD), z.B. für einen Wochenplan mit mehreren Tagen
    if re.match(r"^\d{4}-\d{2}-\d{2}$", label):
        try:
            datetime.strptime(label, "%Y-%m-%d")  # validiert, dass es ein echtes Datum ist
            return label
        except ValueError:
            return None

    return None


class ChatStartIn(BaseModel):
    mode: str  # "checkin" oder "onboarding_full"


@app.post("/chat/start")
async def chat_start(payload: ChatStartIn, user: dict = Depends(get_current_user)):
    """Erzeugt die EINSTIEGSFRAGE für eine Standortbestimmung oder ein komplettes
    Neu-Onboarding — und speichert sie ECHT in der Datenbank (nicht nur im Browser
    angezeigt), damit Claude beim nächsten Turn weiss, worauf sich die Antwort der
    Person bezieht. Das war die eigentliche Ursache der Verwirrung beim letzten Mal."""
    import json

    with get_db() as conn:
        memory = build_memory_context(conn, user["user_id"])

    if payload.mode == "onboarding_full":
        base_prompt = ONBOARDING_SYSTEM_PROMPT
        kickoff = "Starte das Kennenlern-Gespräch von vorne mit der ersten Frage."
    else:
        base_prompt = CHECKIN_SYSTEM_PROMPT
        kickoff = "Starte jetzt die Standortbestimmung mit deiner ersten Frage, basierend auf dem, was du bereits über mich weisst."

    heute = datetime.now(timezone.utc).date()
    wochentage = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    heute_text = f"Heutiges Datum: {heute.isoformat()} ({wochentage[heute.weekday()]})"
    system_prompt = f"{base_prompt}\n\n{heute_text}\n\n--- Bekannte Eckdaten der Person ---\n{memory}"

    raw = await call_claude(system_prompt, [{"role": "user", "content": kickoff}])

    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        antwort = parsed.get("antwort", raw)
    except (json.JSONDecodeError, AttributeError):
        antwort = raw

    with get_db() as conn:
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'chat_assistant', ?, FALSE, ?)",
            (user["user_id"], antwort, now_iso()),
        )

    return {"answer": antwort}


def create_task_entry(conn, user_id: int, inhalt: str, faellig_label: Optional[str]) -> None:
    due = due_date_from_label(faellig_label)
    run_write(
        conn,
        "INSERT INTO entries (user_id, type, content, done, due_date, created_at) VALUES (?, 'task', ?, FALSE, ?, ?)",
        (user_id, inhalt, due, now_iso()),
    )


def create_notiz_entry(conn, user_id: int, text: str) -> None:
    run_write(
        conn,
        "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'mentor_notiz', ?, FALSE, ?)",
        (user_id, text.strip(), now_iso()),
    )


def apply_standbein_update(conn, user_id: int, standbein_update: dict) -> bool:
    """Legt ein neues Standbein an oder merged in ein bestehendes (nach Name).
    Enthält dieselbe Logik, die vorher inline in /chat stand — jetzt auch von
    /suggestions/confirm wiederverwendbar."""
    import json

    if not (isinstance(standbein_update, dict) and standbein_update.get("name")):
        return False

    neuer_name = standbein_update["name"].strip().lower()
    bestehende_ventures = fetch_entries(conn, user_id, "venture", limit=50)
    passendes_venture = None
    for v in bestehende_ventures:
        try:
            v_data = json.loads(v["content"])
            if v_data.get("name", "").strip().lower() == neuer_name:
                passendes_venture = (v["id"], v_data)
                break
        except (json.JSONDecodeError, TypeError):
            continue

    neue_meilensteine = standbein_update.get("meilensteine", []) or []

    if passendes_venture:
        venture_id, v_data = passendes_venture
        if standbein_update.get("vision"):
            v_data["vision"] = standbein_update["vision"]
        if standbein_update.get("phase") in VENTURE_PHASES:
            v_data["phase"] = standbein_update["phase"]
        elif v_data.get("phase") not in VENTURE_PHASES:
            v_data["phase"] = "idee"
        bestehende_meilensteine = normalize_meilensteine(v_data.get("meilensteine"))
        bestehende_texte = {m.get("text", "").strip().lower() for m in bestehende_meilensteine}
        for m in neue_meilensteine:
            if isinstance(m, dict) and m.get("text", "").strip().lower() not in bestehende_texte:
                bestehende_meilensteine.append({
                    "text": m.get("text", ""),
                    "datum": m.get("datum"),
                    "erledigt": False,
                    "messgroesse": m.get("messgroesse", ""),
                })
        v_data["meilensteine"] = bestehende_meilensteine
        v_data["umsatz"] = normalize_umsatz(v_data.get("umsatz"))
        run_write(
            conn,
            "UPDATE entries SET content = ? WHERE id = ? AND user_id = ?",
            (json.dumps(v_data, ensure_ascii=False), venture_id, user_id),
        )
    else:
        neues_venture = {
            "name": standbein_update["name"],
            "vision": standbein_update.get("vision", ""),
            "phase": standbein_update.get("phase") if standbein_update.get("phase") in VENTURE_PHASES else "idee",
            "role": standbein_update.get("role", ""),
            "focus": standbein_update.get("focus") if standbein_update.get("focus") in VENTURE_FOCUS_OPTIONS else "secondary",
            "umsatz": [],
            "meilensteine": [
                {
                    "text": m.get("text", ""),
                    "datum": m.get("datum"),
                    "erledigt": False,
                    "messgroesse": m.get("messgroesse", ""),
                }
                for m in neue_meilensteine if isinstance(m, dict)
            ],
        }
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'venture', ?, FALSE, ?)",
            (user_id, json.dumps(neues_venture, ensure_ascii=False), now_iso()),
        )
    return True


@app.post("/chat")
async def chat(payload: ChatIn, user: dict = Depends(get_current_user)):
    import json

    with get_db() as conn:
        memory = build_memory_context(conn, user["user_id"])
        has_profile = bool(fetch_entries(conn, user["user_id"], "profile", limit=1))

        # Bisherige Unterhaltung ALS ECHTE NACHRICHTEN abrufen, bevor wir die neue
        # Nachricht einfügen — das ist entscheidend, damit Claude sich innerhalb
        # des Gesprächs an bereits gestellte Fragen/Antworten erinnert.
        previous_turns = fetch_entries_by_types(
            conn, user["user_id"], ["chat_user", "chat_assistant"], limit=30
        )
        previous_turns = list(reversed(previous_turns))  # chronologisch aufsteigend

        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'chat_user', ?, FALSE, ?)",
            (user["user_id"], payload.message, now_iso()),
        )

    messages = [
        {"role": "user" if t["type"] == "chat_user" else "assistant", "content": t["content"]}
        for t in previous_turns
    ]
    messages.append({"role": "user", "content": payload.message})

    if payload.mode == "checkin":
        base_prompt = CHECKIN_SYSTEM_PROMPT
    elif not has_profile or payload.mode == "onboarding":
        base_prompt = ONBOARDING_SYSTEM_PROMPT
    else:
        base_prompt = MENTOR_SYSTEM_PROMPT
    heute = datetime.now(timezone.utc).date()
    wochentage = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    heute_text = f"Heutiges Datum: {heute.isoformat()} ({wochentage[heute.weekday()]})"
    system_prompt = f"{base_prompt}\n\n{heute_text}\n\n--- Bekannte Eckdaten der Person (Profil, Vision, Aufgaben) ---\n{memory}"
    raw = await call_claude(system_prompt, messages)

    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        antwort = parsed.get("antwort", raw)
        neue_aufgaben = parsed.get("neue_aufgaben", [])
        neues_profil = parsed.get("profil")
        notiz_update = parsed.get("notiz_update")
        standbein_update = parsed.get("standbein_update")
    except (json.JSONDecodeError, AttributeError):
        # Falls das Parsen fehlschlägt, nutzen wir die Rohantwort ohne Extraktion,
        # damit der Chat trotzdem funktioniert, statt komplett zu scheitern.
        antwort = raw
        neue_aufgaben = []
        neues_profil = None
        notiz_update = None
        standbein_update = None

    with get_db() as conn:
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'chat_assistant', ?, FALSE, ?)",
            (user["user_id"], antwort, now_iso()),
        )

        erstellte_aufgaben = []
        profil_gespeichert = False
        standbein_gespeichert = False
        vorschlaege = []  # nur befüllt, wenn confirm_mode=True

        # Profil aus dem Onboarding-Gespräch: hat schon eine eigene explizite
        # Bestätigung DURCH DAS GESPRÄCH SELBST ("Hab ich das richtig
        # verstanden?") bevor das JSON überhaupt ausgefüllt zurückkommt -
        # zählt als bereits bestätigt, wird unabhängig von confirm_mode
        # direkt gespeichert (Briefing: "Explizit im Onboarding angegebene
        # Informationen dürfen direkt gespeichert werden").
        if isinstance(neues_profil, dict) and neues_profil:
            save_profile_merged(conn, user["user_id"], neues_profil)
            profil_gespeichert = True

        if payload.confirm_mode:
            # Neues Verhalten: nichts automatisch speichern, nur vorschlagen.
            for aufgabe in neue_aufgaben:
                inhalt = aufgabe.get("inhalt", "") if isinstance(aufgabe, dict) else str(aufgabe)
                faellig_label = aufgabe.get("faellig") if isinstance(aufgabe, dict) else None
                if not inhalt:
                    continue
                vorschlaege.append({"kind": "task", "label": inhalt, "payload": {"inhalt": inhalt, "faellig": faellig_label}})

            if isinstance(notiz_update, str) and notiz_update.strip():
                text = notiz_update.strip()
                vorschlaege.append({"kind": "notiz", "label": text, "payload": {"text": text}})

            if isinstance(standbein_update, dict) and standbein_update.get("name"):
                vorschlaege.append({
                    "kind": "standbein",
                    "label": f"Standbein: {standbein_update['name']}",
                    "payload": standbein_update,
                })
        else:
            # Altes Verhalten, unverändert für das bestehende Frontend:
            # sofort automatisch speichern.
            for aufgabe in neue_aufgaben:
                inhalt = aufgabe.get("inhalt", "") if isinstance(aufgabe, dict) else str(aufgabe)
                faellig_label = aufgabe.get("faellig") if isinstance(aufgabe, dict) else None
                if not inhalt:
                    continue
                create_task_entry(conn, user["user_id"], inhalt, faellig_label)
                erstellte_aufgaben.append(inhalt)

            if isinstance(notiz_update, str) and notiz_update.strip():
                create_notiz_entry(conn, user["user_id"], notiz_update)

            standbein_gespeichert = apply_standbein_update(conn, user["user_id"], standbein_update)

    return {
        "answer": antwort,
        "neue_aufgaben": erstellte_aufgaben,
        "onboarding": (not has_profile) or (payload.mode == "onboarding"),
        "profil_gespeichert": profil_gespeichert,
        "standbein_gespeichert": standbein_gespeichert,
        "vorschlaege": vorschlaege,
    }


class SuggestionConfirmIn(BaseModel):
    kind: str  # "task" | "notiz" | "standbein"
    payload: dict


@app.post("/suggestions/confirm")
def confirm_suggestion(body: SuggestionConfirmIn, user: dict = Depends(get_current_user)):
    """Wendet einen einzelnen, vom User bestätigten Sole-Vorschlag an.
    Wird vom neuen Frontend aufgerufen, wenn auf 'Merken'/'Diese Woche'/
    'Übernehmen' o.ä. geklickt wird."""
    with get_db() as conn:
        if body.kind == "task":
            create_task_entry(conn, user["user_id"], body.payload.get("inhalt", ""), body.payload.get("faellig"))
        elif body.kind == "notiz":
            create_notiz_entry(conn, user["user_id"], body.payload.get("text", ""))
        elif body.kind == "standbein":
            apply_standbein_update(conn, user["user_id"], body.payload)
        else:
            raise HTTPException(status_code=400, detail=f"Unbekannte Vorschlagsart: {body.kind}")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Strategy-Endpoints — Vision + Projekte, der "Kompass"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Profil-Endpoints — für die editierbare Profil-Seite
# ---------------------------------------------------------------------------

PROFILE_FIELDS = [
    "name", "situation", "hintergrund", "vision", "erfolg", "sorge", "reserve", "stil",
    "staerken", "werte", "unterstuetzung",
    "arbeitsweise", "rahmen", "ziel",
]


class ProfileIn(BaseModel):
    name: str = ""
    situation: str = ""
    hintergrund: str = ""
    vision: str = ""
    erfolg: str = ""
    sorge: str = ""
    reserve: str = ""
    stil: str = ""
    staerken: str = ""
    werte: str = ""
    unterstuetzung: str = ""
    # V1-Erweiterung — zusätzliche Facts-Felder, altes Frontend nutzt sie nicht,
    # bestehende Felder oben bleiben unverändert erhalten.
    arbeitsweise: str = ""
    rahmen: str = ""
    ziel: str = ""


@app.get("/profile")
def get_profile(user: dict = Depends(get_current_user)):
    import json

    with get_db() as conn:
        profile = fetch_entries(conn, user["user_id"], "profile", limit=1)
    if not profile:
        return {"exists": False, **{k: "" for k in PROFILE_FIELDS}}
    try:
        data = json.loads(profile[0]["content"])
    except (json.JSONDecodeError, TypeError):
        data = {}
    return {"exists": True, **{k: data.get(k, "") for k in PROFILE_FIELDS}}


@app.post("/profile")
def set_profile(payload: ProfileIn, user: dict = Depends(get_current_user)):
    import json

    with get_db() as conn:
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'profile', ?, FALSE, ?)",
            (user["user_id"], json.dumps(payload.model_dump(), ensure_ascii=False), now_iso()),
        )
    return {"ok": True}


REFLEXION_SYSTEM_PROMPT = """Du bist der "Sole."-Mentor. Schreib eine kurze, warme Einschätzung \
(3-5 Sätze) der Person, basierend auf allem, was du über sie weisst (Profil, Vision, Standbeine, \
Aufgaben, bisherige Beobachtungen). Das ist KEINE reine Auflistung von Fakten, sondern eine echte, \
persönliche Synthese - was fällt dir an dieser Person auf, welches Muster siehst du, was schätzt du \
an ihr. Schreib direkt an die Person ("du"), warm und ehrlich, nicht übertrieben lobend. Falls noch \
sehr wenig über die Person bekannt ist, schreib das ehrlich statt etwas zu erfinden - z.B. dass ihr \
euch noch am Kennenlernen seid. Auf Deutsch. Antworte NUR mit dem Text, ohne Anführungszeichen, \
ohne Überschrift."""


@app.post("/profile/reflection")
async def generate_reflection(user: dict = Depends(get_current_user)):
    """Erzeugt sofort eine Sole-Einschätzung, statt darauf zu warten, dass sie organisch
    im Chat entsteht. Wird als normale 'mentor_notiz' gespeichert, taucht also direkt in
    derselben Liste wie die organisch entstandenen Beobachtungen auf."""
    with get_db() as conn:
        memory = build_memory_context(conn, user["user_id"])

    text = await call_claude(
        REFLEXION_SYSTEM_PROMPT,
        [{"role": "user", "content": f"Hier ist der bisherige Kontext über die Person:\n\n{memory}"}],
    )

    with get_db() as conn:
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'mentor_notiz', ?, FALSE, ?)",
            (user["user_id"], text.strip(), now_iso()),
        )

    return {"text": text.strip()}


WEEKLY_REVIEW_SYSTEM_PROMPT = """Du bist der "Sole."-Mentor. Die Person hat um einen Wochenrückblick \
gebeten. Du bekommst unten eine Liste, was diese Woche erledigt wurde, was liegen geblieben ist, und \
den aktuellen Stand ihrer Standbeine.

Schreib NUR den Abschnitt "Was jetzt zählt" - eine kurze, klare Empfehlung (2-4 Sätze) für die \
kommende Woche, mit Haltung, nicht als neutrale Liste. Nicht "Du könntest X oder Y tun", sondern \
"Für nächste Woche würde ich einen Schwerpunkt setzen: ...". Beziehe dich konkret auf das, was \
liegen geblieben ist oder was der aktuelle Meilenstein-Fokus nahelegt - keine generischen Ratschläge. \
Falls zu wenig bewegt wurde, um daraus etwas Sinnvolles abzuleiten, sag das ehrlich, statt eine \
Empfehlung zu erfinden. Auf Deutsch. Antworte NUR mit dem Text, ohne Anführungszeichen, ohne \
Überschrift."""


@app.post("/weekly-review/generate")
async def generate_weekly_review(user: dict = Depends(get_current_user)):
    """Nur auf Anfrage, nie automatisch. GEMACHT/NICHT BEWEGT/GELERNT/COMPASS sind
    echte, direkt abgefragte Daten - nur 'was_jetzt_zaehlt' kommt von Claude, als
    gezielte Synthese, nicht als erfundener Text."""
    import json

    sieben_tage_alt = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    with get_db() as conn:
        alle_tasks = fetch_entries(conn, user["user_id"], "task", limit=300)
        alle_notizen = fetch_entries(conn, user["user_id"], "mentor_notiz", limit=50)
        alle_ventures = fetch_entries(conn, user["user_id"], "venture", limit=20)

    gemacht = [
        t["content"] for t in alle_tasks
        if t.get("completed_at") and t["completed_at"] >= sieben_tage_alt
    ]
    heute = datetime.now(timezone.utc).date().isoformat()
    sieben_tage_alt_datum = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
    nicht_bewegt = [
        t["content"] for t in alle_tasks
        if t.get("status", "open") == "open"
        and t.get("due_date") and sieben_tage_alt_datum <= t["due_date"] < heute
    ]
    gelernt = [
        n["content"] for n in alle_notizen
        if n["created_at"] >= sieben_tage_alt
    ]

    compass_stand = []
    for v in alle_ventures:
        try:
            v_data = json.loads(v["content"])
            compass_stand.append({
                "name": v_data.get("name", ""),
                "phase": v_data.get("phase", "idee"),
                "focus": v_data.get("focus", "secondary"),
            })
        except (json.JSONDecodeError, TypeError):
            continue

    kontext_teile = []
    if gemacht:
        kontext_teile.append("Diese Woche erledigt: " + "; ".join(gemacht))
    if nicht_bewegt:
        kontext_teile.append("Liegen geblieben (war für diese Woche geplant): " + "; ".join(nicht_bewegt))
    if compass_stand:
        kontext_teile.append(
            "Aktueller Compass-Stand: "
            + "; ".join(f"{v['name']} ({v['phase']}, {v['focus']})" for v in compass_stand)
        )
    kontext = "\n".join(kontext_teile) if kontext_teile else "Diese Woche wenig Aktivität erfasst."

    was_jetzt_zaehlt = await call_claude(
        WEEKLY_REVIEW_SYSTEM_PROMPT, [{"role": "user", "content": kontext}]
    )

    return {
        "gemacht": gemacht,
        "nicht_bewegt": nicht_bewegt,
        "gelernt": gelernt,
        "compass_stand": compass_stand,
        "was_jetzt_zaehlt": was_jetzt_zaehlt.strip(),
    }


# ---------------------------------------------------------------------------
# Portfolio-Dokument — automatisch generiert aus Profil + Compass
# ---------------------------------------------------------------------------

@app.post("/portfolio/generate")
async def generate_portfolio(user: dict = Depends(get_current_user)):
    with get_db() as conn:
        memory = build_memory_context(conn, user["user_id"])

    text = await call_claude(
        PORTFOLIO_SYSTEM_PROMPT,
        [{"role": "user", "content": f"Hier ist der bisherige Kontext über die Person:\n\n{memory}"}],
    )

    with get_db() as conn:
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'portfolio_doc', ?, FALSE, ?)",
            (user["user_id"], text.strip(), now_iso()),
        )

    return {"text": text.strip()}


@app.get("/portfolio")
def get_portfolio(user: dict = Depends(get_current_user)):
    with get_db() as conn:
        rows = fetch_entries(conn, user["user_id"], "portfolio_doc", limit=1)
    return {
        "text": rows[0]["content"] if rows else "",
        "erstellt_am": rows[0]["created_at"][:10] if rows else None,
    }


@app.get("/profile/journey")
def get_journey(user: dict = Depends(get_current_user)):
    """Stellt Statistiken für die 'Deine Reise'-Übersicht zusammen: seit wann dabei,
    wie viel erledigt, die Entwicklung des Profils über Zeit, plus Soles fortlaufende Notizen."""
    import json

    with get_db() as conn:
        member_since_rows = run_query(
            conn, "SELECT created_at FROM users WHERE id = ?", (user["user_id"],)
        )
        tasks = fetch_entries(conn, user["user_id"], "task", limit=1000)
        ventures_raw = fetch_entries(conn, user["user_id"], "venture", limit=100)
        profile_history = fetch_entries(conn, user["user_id"], "profile", limit=20)
        mentor_notizen = fetch_entries(conn, user["user_id"], "mentor_notiz", limit=50)

    aufgaben_erledigt = len([t for t in tasks if t["done"]])
    standbeine = len(ventures_raw)

    meilensteine_erreicht = 0
    for v in ventures_raw:
        try:
            data = json.loads(v["content"])
            for m in normalize_meilensteine(data.get("meilensteine")):
                if isinstance(m, dict) and m.get("erledigt"):
                    meilensteine_erreicht += 1
        except (json.JSONDecodeError, TypeError):
            continue

    historie = []
    for p in profile_history:
        try:
            data = json.loads(p["content"])
            historie.append({
                "datum": p["created_at"][:10],
                "situation": data.get("situation", ""),
            })
        except (json.JSONDecodeError, TypeError):
            continue

    notizen_liste = [
        {"datum": n["created_at"][:10], "text": n["content"]}
        for n in reversed(mentor_notizen)  # älteste zuerst, wie ein wachsender Text
    ]

    return {
        "seit": member_since_rows[0]["created_at"][:10] if member_since_rows else None,
        "aufgaben_erledigt": aufgaben_erledigt,
        "standbeine": standbeine,
        "meilensteine_erreicht": meilensteine_erreicht,
        "profil_historie": list(reversed(historie)),  # älteste zuerst
        "mentor_notizen": notizen_liste,
    }


class MeilensteinIn(BaseModel):
    text: str
    datum: Optional[str] = None  # ISO-Datum YYYY-MM-DD, optional
    erledigt: bool = False
    messgroesse: str = ""  # "Wie misst du, ob's erreicht ist?" — optional


class UmsatzEintragIn(BaseModel):
    betrag: float
    datum: Optional[str] = None  # ISO-Datum YYYY-MM-DD, optional
    notiz: str = ""


VENTURE_PHASES = ["idee", "validieren", "aufbauen", "umsetzen", "wachsen"]
VENTURE_FOCUS_OPTIONS = ["primary", "secondary", "parked"]


class VentureIn(BaseModel):
    name: str
    vision: str = ""
    meilensteine: list[MeilensteinIn] = []
    umsatz: list[UmsatzEintragIn] = []
    phase: str = "idee"
    role: str = ""
    focus: str = "secondary"


def normalize_meilensteine(raw) -> list:
    """Alte Ventures hatten 'meilensteine' als einen einzigen Textblock ohne Datum.
    Wandelt das für die Anzeige in die neue Listenform um, ohne Daten zu verlieren."""
    if isinstance(raw, str):
        if not raw.strip():
            return []
        return [{"text": raw, "datum": None, "erledigt": False, "messgroesse": ""}]
    if isinstance(raw, list):
        for m in raw:
            if isinstance(m, dict):
                if "erledigt" not in m:
                    m["erledigt"] = False
                if "messgroesse" not in m:
                    m["messgroesse"] = ""
        return raw
    return []


def normalize_umsatz(raw) -> list:
    """Gibt eine leere Liste zurück, falls noch keine Umsatz-Einträge vorhanden sind
    (z.B. bei älteren Standbeinen, die vor diesem Feature angelegt wurden)."""
    if isinstance(raw, list):
        return raw
    return []


@app.get("/strategy")
def get_strategy(user: dict = Depends(get_current_user)):
    import json

    with get_db() as conn:
        overall_vision = fetch_entries(conn, user["user_id"], "overall_vision", limit=1)
        ventures_raw = fetch_entries(conn, user["user_id"], "venture", limit=20)

    ventures = []
    for v in ventures_raw:
        try:
            data = json.loads(v["content"])
            data["id"] = v["id"]
            data["meilensteine"] = normalize_meilensteine(data.get("meilensteine"))
            data["umsatz"] = normalize_umsatz(data.get("umsatz"))
            if data.get("phase") not in VENTURE_PHASES:
                data["phase"] = "idee"
            if data.get("focus") not in VENTURE_FOCUS_OPTIONS:
                data["focus"] = "secondary"
            if "role" not in data:
                data["role"] = ""
            ventures.append(data)
        except (json.JSONDecodeError, TypeError):
            continue

    return {
        "overall_vision": overall_vision[0]["content"] if overall_vision else "",
        "ventures": ventures,
    }


@app.post("/strategy/overall-vision")
async def set_overall_vision(payload: GoalIn, user: dict = Depends(get_current_user)):
    """Nimmt einen groben Vision-Text entgegen, lässt ihn von Claude schärfen, speichert ihn.
    Bezieht bestehende Standbeine als Kontext mit ein, damit die geschärfte Vision darauf eingeht."""
    import json

    with get_db() as conn:
        ventures_raw = fetch_entries(conn, user["user_id"], "venture", limit=20)
    venture_names = []
    for v in ventures_raw:
        try:
            venture_names.append(json.loads(v["content"]).get("name", ""))
        except (json.JSONDecodeError, TypeError):
            continue

    context = ""
    if venture_names:
        context = "\n\n(Bekannte Standbeine der Person: " + ", ".join(venture_names) + ")"

    refined = await call_claude(
        STRATEGY_SYSTEM_PROMPT, [{"role": "user", "content": payload.goal + context}]
    )
    with get_db() as conn:
        run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'overall_vision', ?, FALSE, ?)",
            (user["user_id"], refined, now_iso()),
        )
    return {"overall_vision": refined}


@app.post("/strategy/venture")
def add_venture(payload: VentureIn, user: dict = Depends(get_current_user)):
    import json

    content = json.dumps(payload.model_dump(exclude={"id"}, exclude_unset=False), ensure_ascii=False)
    with get_db() as conn:
        new_id = run_write(
            conn,
            "INSERT INTO entries (user_id, type, content, done, created_at) VALUES (?, 'venture', ?, FALSE, ?)",
            (user["user_id"], content, now_iso()),
        )
    return {"id": new_id}


@app.put("/strategy/venture/{venture_id}")
def update_venture(venture_id: int, payload: VentureIn, user: dict = Depends(get_current_user)):
    import json

    content = json.dumps(payload.model_dump(), ensure_ascii=False)
    with get_db() as conn:
        run_write(
            conn,
            "UPDATE entries SET content = ? WHERE id = ? AND user_id = ?",
            (content, venture_id, user["user_id"]),
        )
    return {"ok": True}


@app.delete("/strategy/venture/{venture_id}")
def delete_venture(venture_id: int, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        run_write(
            conn, "DELETE FROM entries WHERE id = ? AND user_id = ?", (venture_id, user["user_id"])
        )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Wöchentlicher E-Mail-Rückblick — wird NICHT von Personen direkt aufgerufen,
# sondern einmal pro Woche von einem zeitgesteuerten Auslöser (Cron Job).
# Geschützt durch ein Secret, damit nicht irgendjemand diesen Endpoint missbraucht.
# ---------------------------------------------------------------------------

DIGEST_SYSTEM_PROMPT = """Du schreibst einen kurzen, warmen wöchentlichen Rückblick-Absatz (3-4 Sätze) \
für eine Person im Übergang in die Selbständigkeit, als Teil einer E-Mail. Beziehe dich auf ihre \
erledigten Aufgaben diese Woche, ihre Vision/Situation falls bekannt, und ermutige sie ehrlich, ohne \
übertriebene Cheerleader-Sprache. Auf Deutsch. Antworte NUR mit dem Absatz-Text, kein JSON, keine \
Anführungszeichen, keine Überschrift."""


@app.post("/admin/send-weekly-digests")
async def send_weekly_digests(x_cron_secret: str = Header(None)):
    import json

    if not CRON_SECRET or x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Ungültiges oder fehlendes Cron-Secret.")

    with get_db() as conn:
        users = run_query(conn, "SELECT id, email FROM users")

    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    sent_count = 0

    for u in users:
        with get_db() as conn:
            tasks = fetch_entries(conn, u["id"], "task", limit=200)
            profile = fetch_entries(conn, u["id"], "profile", limit=1)

        erledigt_diese_woche = [
            t for t in tasks if t["done"] and t["created_at"] >= week_ago
        ]
        offen = [t for t in tasks if not t["done"]]

        name = ""
        situation = ""
        if profile:
            try:
                p = json.loads(profile[0]["content"])
                name = p.get("name", "")
                situation = p.get("situation", "")
            except (json.JSONDecodeError, TypeError):
                pass

        summary_input = (
            f"Name: {name or 'unbekannt'}\nSituation: {situation or 'unbekannt'}\n"
            f"Diese Woche erledigt ({len(erledigt_diese_woche)}): "
            + ", ".join(t["content"] for t in erledigt_diese_woche[:8])
            + f"\nOffene Aufgaben aktuell: {len(offen)}"
        )

        try:
            absatz = await call_claude(
                DIGEST_SYSTEM_PROMPT, [{"role": "user", "content": summary_input}]
            )
        except HTTPException:
            absatz = "Schön, dass du diese Woche dabei warst."

        html = f"""
        <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:24px;">
          <h2 style="font-family:serif;">Sole. — Dein Wochenrückblick</h2>
          <p>{absatz}</p>
          <p style="font-size:13px;color:#666;">
            {len(erledigt_diese_woche)} Aufgabe(n) erledigt diese Woche, {len(offen)} noch offen.
          </p>
          <a href="https://charming-moxie-8f36aa.netlify.app/sole-mentor.html"
             style="display:inline-block;background:#1C2E29;color:#fff;padding:10px 20px;
                    border-radius:999px;text-decoration:none;margin-top:12px;">
            Zu Sole. →
          </a>
        </div>
        """

        success = await send_email(u["email"], "Dein Sole.-Wochenrückblick", html)
        if success:
            sent_count += 1

    return {"ok": True, "sent": sent_count, "total_users": len(users)}


# ---------------------------------------------------------------------------
# Kalender-Abo (iCal / .ics) — funktioniert mit Google Kalender, Apple Kalender,
# Outlook. Nutzt einen geheimen Token in der URL statt Login, weil Kalender-Apps
# keine Bearer-Token-Header mitschicken können.
# ---------------------------------------------------------------------------

@app.get("/calendar/token")
def get_calendar_token(user: dict = Depends(get_current_user)):
    """Gibt den persönlichen Kalender-Token zurück, erzeugt ihn falls noch keiner existiert."""
    with get_db() as conn:
        rows = run_query(conn, "SELECT calendar_token FROM users WHERE id = ?", (user["user_id"],))
        token = rows[0]["calendar_token"] if rows else None
        if not token:
            token = secrets.token_urlsafe(24)
            run_write(conn, "UPDATE users SET calendar_token = ? WHERE id = ?", (token, user["user_id"]))
    return {"token": token}


def escape_ics_text(text: str) -> str:
    """Escaped Sonderzeichen gemäss iCalendar-Spezifikation."""
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


@app.get("/calendar/{token}.ics")
def get_ics_feed(token: str):
    """Öffentlicher (aber geheimer) Kalender-Feed — kein Login nötig, nur der Token
    in der URL. Enthält alle Aufgaben mit Fälligkeitsdatum als ganztägige Termine."""
    with get_db() as conn:
        rows = run_query(conn, "SELECT id FROM users WHERE calendar_token = ?", (token,))
        if not rows:
            raise HTTPException(status_code=404, detail="Ungültiger Kalender-Link.")
        user_id = rows[0]["id"]
        tasks = fetch_entries(conn, user_id, "task", limit=1000)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Sole.//Aufgaben-Kalender//DE",
        "CALSCALE:GREGORIAN",
        "X-WR-CALNAME:Sole. Aufgaben",
    ]

    for t in tasks:
        if not t["due_date"]:
            continue
        due_compact = t["due_date"].replace("-", "")
        prefix = "✓ " if t["done"] else ""
        summary = escape_ics_text(prefix + t["content"])
        lines += [
            "BEGIN:VEVENT",
            f"UID:sole-task-{t['id']}@sole-app",
            f"DTSTART;VALUE=DATE:{due_compact}",
            f"DTEND;VALUE=DATE:{due_compact}",
            f"SUMMARY:{summary}",
            f"DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "END:VEVENT",
        ]

    lines.append("END:VCALENDAR")
    ics_content = "\r\n".join(lines)

    from fastapi.responses import Response
    return Response(
        content=ics_content,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": "inline; filename=sole-aufgaben.ics"},
    )


# ---------------------------------------------------------------------------
# Datenexport — alle eigenen Daten als JSON-Datei zum Download.
# ---------------------------------------------------------------------------

@app.get("/export")
def export_data(user: dict = Depends(get_current_user)):
    import json as json_module

    with get_db() as conn:
        all_entries = run_query(
            conn, "SELECT * FROM entries WHERE user_id = ? ORDER BY id", (user["user_id"],)
        )
        user_rows = run_query(
            conn, "SELECT email, created_at FROM users WHERE id = ?", (user["user_id"],)
        )

    export = {
        "exportiert_am": now_iso(),
        "konto": user_rows[0] if user_rows else {},
        "eintraege": all_entries,
    }

    from fastapi.responses import Response
    return Response(
        content=json_module.dumps(export, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=sole-daten-export.json"},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_configured": bool(ANTHROPIC_API_KEY),
        "jwt_secret_configured": bool(JWT_SECRET),
    }
