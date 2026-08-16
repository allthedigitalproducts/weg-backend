# Weg. Backend

Ein echter kleiner Server mit Gedächtnis — löst zwei Dinge, die eine reine
HTML-Datei nicht kann:

1. **Alles wird dauerhaft gespeichert** (SQLite-Datenbank statt Browser-Variable, die beim Schliessen verschwindet).
2. **Der Chat und die Zielzerlegung laufen über die echte Claude API**, mit
   deiner gesamten bisherigen Historie als Kontext — statt der 5 fest
   programmierten Kategorien aus dem Vorgänger-Prototyp.

## Wichtiger Hinweis zu diesem Test

Ich konnte den Server in meiner aktuellen Umgebung **nicht live starten und
durchtesten**, weil hier kein Internetzugriff verfügbar ist (weder um die
Bibliotheken zu installieren noch um die Anthropic API aufzurufen). Die
Python-Syntax ist geprüft und sauber, aber der erste echte End-to-End-Test
passiert bei dir lokal. Plane dafür ruhig etwas Zeit für kleine Fehlerbehebung ein.

## Lokal starten

Voraussetzung: Python 3.10 oder neuer ist installiert.

```bash
cd weg-backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=dein-api-key    # auf Windows: set ANTHROPIC_API_KEY=...
uvicorn main:app --reload --port 8000
```

Der Server läuft dann auf `http://localhost:8000`. Test:

```bash
curl http://localhost:8000/health
```

Sollte `{"status":"ok","api_key_configured":true}` zurückgeben.

## Woher bekommst du einen API-Key?

Über die Anthropic Console (console.anthropic.com) — dort lässt sich ein
API-Key erstellen. Das ist ein separater Zugang von der normalen
Claude.ai-Nutzung und wird nach Verbrauch abgerechnet (bei diesem
Nutzungsvolumen vermutlich wenige Franken pro Monat für einen Test).

## Endpoints im Überblick

| Endpoint | Zweck |
|---|---|
| `GET /entries?type=...` | Alle Einträge eines Typs abrufen (goal, priority, task, journal, chat_user, chat_assistant) |
| `POST /entries` | Neuen Eintrag anlegen (z.B. Journal-Notiz) |
| `PATCH /entries/{id}` | Eintrag als erledigt markieren oder Text ändern |
| `DELETE /entries/{id}` | Eintrag löschen |
| `POST /chat` | Nachricht an den Mentor — bekommt volle Historie als Kontext |
| `POST /goal/decompose` | Ziel eingeben, wird per Claude in Prioritäten + Aufgaben zerlegt |

## Deployment, damit es nicht nur lokal läuft

Für einen echten Test mit anderen Personen muss der Server irgendwo laufen,
nicht nur auf deinem eigenen Rechner. Zwei einfache, kostengünstige Optionen:

- **Render.com** — kostenloses Tier zum Testen, `ANTHROPIC_API_KEY` als
  Umgebungsvariable im Dashboard setzen, Python-Service mit
  `uvicorn main:app --host 0.0.0.0 --port $PORT` als Startbefehl.
- **Railway.app** — ähnlich einfach, wenige Klicks, kleine monatliche Kosten
  nach der Testphase.

Wichtig: die SQLite-Datei (`weg.db`) liegt bei den meisten kostenlosen
Hosting-Optionen nicht dauerhaft — für einen längeren Test lohnt sich früher
oder später eine "echte" Datenbank (z.B. Postgres via Supabase, ebenfalls
kostenlos im Kleinen). Für die ersten Tests reicht SQLite völlig.

## Verbindung zum Frontend

Das Dashboard (`weg-dashboard.html`) muss danach so angepasst werden, dass es
nicht mehr die eingebauten Demo-Antworten nutzt, sondern `fetch()`-Aufrufe an
deine Server-Adresse macht (z.B. `https://dein-server.onrender.com/chat`
statt der lokalen Demo-Logik). Das ist der nächste Bau-Schritt, sobald der
Server bei dir läuft und der `/health`-Check funktioniert.

## Sicherheitshinweis

Für den Prototyp ist keine Anmeldung/kein Login eingebaut — jede Person mit
der Server-Adresse kann aktuell auf dieselben Daten zugreifen. Das ist okay
für einen ersten Test mit dir allein, aber bevor andere Personen eigene,
private Journale führen, braucht es eine einfache Nutzer-Trennung (z.B. ein
Zugriffscode pro Person). Sag Bescheid, wenn du so weit bist — das ist ein
überschaubarer nächster Schritt, kein Neubau.
