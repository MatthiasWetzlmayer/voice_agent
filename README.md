# Voice Assistant

Sprachassistent aus zwei Agenten in einem Prozess:

- **Voice-Agent** (`create_agent`, Ollama): führt das Gespräch.
- **Task-Agent** (`create_deep_agent`, Ollama): erledigt längere Aufgaben im Hintergrund
  (bisher nur Dummy-Tools `slow_work` und `dummy_action`).

Man redet einfach drauflos, alles läuft lokal und kostenlos:

- **Spracherkennung:** RealtimeSTT mit faster-whisper. Es erkennt Sprechpausen, zeigt live mit,
  was es versteht, und liefert nach der Pause den fertigen Satz.
- **Agenten:** Ollama.
- **Sprachausgabe:** Piper mit deutscher Stimme, satzweise, damit die Antwort früh beginnt.

## Aufbau

```
agent/   der Sprachassistent (Python), läuft lokal
ui/      Web-Dashboard (React), zeigt nur an, was der Agent gerade tut
```

Der Agent stellt seinen Zustand unter `http://127.0.0.1:8765/api/state` bereit (nur lesend,
nur auf diesem Rechner). Das Dashboard fragt das jede Sekunde ab. Der Agent funktioniert auch
ohne Dashboard.

## Start (macOS)

Agent:

```bash
cd agent
brew install portaudio          # für das Mikrofon (PyAudio)
ollama pull qwen3:4b
cp .env.example .env
uv sync                         # Python 3.12, RealtimeSTT unterstützt noch kein 3.13
uv run voice-assistant          # sprechen
uv run voice-assistant --no-audio   # tippen statt sprechen
uv run voice-assistant --debug      # Mikrofon-Pegel und jeden Erkennungsschritt anzeigen
uv run pytest
```

Dashboard (zweites Terminal, braucht Node.js 22.12 oder neuer):

```bash
cd ui
npm install
npm run dev                     # dann http://localhost:5173 öffnen
```

Beim ersten Start werden die Whisper-Modelle und die Piper-Stimme heruntergeladen. Das Terminal
braucht die Mikrofon-Berechtigung (Systemeinstellungen → Datenschutz & Sicherheit → Mikrofon).
Solange der Assistent spricht, ist das Mikrofon stumm, damit er sich nicht selbst hört.

Speicher auf einem 16-GB-Mac, grob: `qwen3:4b` mit 16k Kontext ca. 5 GB, Whisper `small` und
`tiny` ca. 1 GB, Piper unter 100 MB.

## Dateien

| Datei                    | Inhalt                                                                 |
|--------------------------|------------------------------------------------------------------------|
| `agent/.../agents.py`    | Prompts, Dummy-Tools, Modellauswahl, Erzeugung beider Agenten          |
| `agent/.../assistant.py` | Gespräch, Hintergrund-Tasks, Kommunikation zwischen beiden, `snapshot()` |
| `agent/.../audio.py`     | Live-Transkription (RealtimeSTT), Sprachausgabe (Piper), Satz-Splitter |
| `agent/.../dashboard.py` | Datenendpunkt `/api/state` für das Dashboard                          |
| `agent/.../main.py`      | Start                                                                  |
| `ui/src/api.ts`          | Datentypen und Abfrage des Agenten                                     |
| `ui/src/panels/`         | Ansichten des Dashboards, registriert in `panels/index.ts`             |

## Dashboard erweitern

Jede Ansicht ist ein Panel in `ui/src/panels/` und wird in `ui/src/panels/index.ts` eingetragen.
Braucht ein neues Panel Daten, die es noch nicht gibt: im Agenten in `snapshot()`
(`assistant.py`) ergänzen und in `ui/src/api.ts` den Typ `AgentState` erweitern.

## Wie die Agenten miteinander reden

**Voice → Task:** Der Voice-Agent hat die Tools `start_task`, `list_tasks`, `cancel_task` und
`answer_task`. `start_task` startet den DeepAgent mit `asyncio.create_task` und kehrt sofort
zurück. Jeder Task hat einen eigenen Thread.

**Task → Voice:** Ist ein Task fertig, fehlgeschlagen oder hat eine Rückfrage, ruft er
`notify_voice(text)` auf. Das startet einen neuen Lauf des Voice-Agenten auf dessen Thread, die
Nachricht beginnt mit `[TASK-BENACHRICHTIGUNG]`. Bei OpenAI ist das eine `SystemMessage`, bei
Ollama und anderen eine `HumanMessage`.

**Rückfragen:** `dummy_action` braucht eine Freigabe (`interrupt_on`). Der Task pausiert, der
Voice-Agent stellt die Frage, und `answer_task` setzt den Task mit
`Command(resume={"decisions": [{"type": "approve"}]})` (bzw. `"reject"`) fort.

**Reihenfolge:** Ein `asyncio.Lock` sorgt dafür, dass immer nur eine Antwort gleichzeitig
gesprochen wird. Benachrichtigungen warten, bis der Nutzer ausgeredet hat.

Beide Agenten teilen sich einen `InMemoryStore`.
