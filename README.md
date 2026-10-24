# Voice Assistant: Voice-Agent + Task-Agent

Prototyp eines Sprachassistenten aus zwei Agenten in einem Prozess (asyncio):

- **Voice-Agent**: schlanker LangChain-Agent (`langchain.agents.create_agent`) in einer
  Sandwich-Pipeline **STT -> Agent -> TTS**. Er führt das Gespräch und delegiert.
- **Task-Agent**: ein DeepAgent (`deepagents.create_deep_agent`), der längere Aufgaben im
  Hintergrund erledigt. Bisher nur Dummy-Tools (`slow_work`, `dummy_action`).

Der Schwerpunkt liegt auf der bidirektionalen Kommunikation zwischen beiden.

Geprüft mit langchain 1.4, langgraph 1.2, deepagents 0.7 (siehe `uv.lock`).

## Setup

```bash
uv sync --extra audio          # mit Mikrofon/Lautsprecher (sounddevice)
uv sync --group dev            # Tests
cp .env.example .env           # OPENAI_API_KEY eintragen
```

Für Audio wird PortAudio benötigt (Linux: `sudo apt install libportaudio2`, macOS: `brew install portaudio`).
Fehlt es, startet die App automatisch im Textmodus.

## Start

```bash
uv run voice-assistant              # Push-to-Talk + Sprachausgabe
uv run voice-assistant --no-audio   # nur Text: Eingabe tippen, Ausgabe auf der Konsole
uv run voice-assistant --fake       # Textmodus mit regelbasierten Fake-Modellen (ohne API-Key)
uv run pytest                       # Akzeptanztests mit Fake-Modellen
```

Bedienung im Terminal:

| Eingabe            | Wirkung                                                  |
|--------------------|----------------------------------------------------------|
| Enter (leer)       | Aufnahme starten, nochmal Enter stoppt (Push-to-Talk)    |
| Text + Enter       | Text direkt an den Voice-Agenten                         |
| `/tasks`           | Task-Übersicht                                           |
| `/quit`            | Beenden (laufende Tasks werden abgebrochen)              |

Beispiele für den Fake-Modus: "Hallo", "Recherchier etwas zu Wärmepumpen",
"Wie weit ist die Recherche?", "Buch mir einen Tisch" (Rückfrage, dann "Ja, mach" oder "Nein"),
"Brich die Recherche ab", "Erledige die Aufgabe zu Fehler" (simulierter Fehler).

## Module

```
src/voice_assistant/
  audio/            STT/TTS-Interfaces, OpenAI-Adapter, Konsolen-TTS, Recorder, Satz-Splitter
  voice_agent.py    Voice-Agent (create_agent), Systemprompt, Tools start/list/cancel/answer_task
  task_agent.py     Task-Agent (create_deep_agent), Dummy-Tools, interrupt_on für dummy_action
  task_manager.py   TaskManager: Tasks starten, Status, Abbruch, Rückfragen, Benachrichtigungen
  session.py        VoiceSession: Voice-Thread, Lock, Pipeline Agent-Tokens -> Sätze -> TTS
  notifications.py  make_notification(): SystemMessage (OpenAI) oder markierte HumanMessage
  persistence.py    Checkpointer-/Store-Fabriken (InMemory, später austauschbar gegen Postgres)
  app.py            Verdrahtung beider Agenten
  main.py           Terminal-Frontend (Push-to-Talk, Texteingabe, --no-audio, --fake)
  fakes.py          Regelbasierte Fake-Chatmodelle für Tests und --fake
```

## Kommunikationsfluss

```
           Nutzer (Mikrofon / Tastatur)
                     |
                     v
   STT --> VoiceSession.handle_user_text --+
                                           |  asyncio.Lock (ein Lauf zur Zeit,
   TaskManager --> notify_voice(text) -----+  frei erst nach Ende der TTS-Ausgabe)
        ^                                  v
        |                     Voice-Agent (Thread "voice-…")
        |                        | Tokens -> Sätze -> TTS
        |                        |
        |    start/list/cancel/answer_task (kehren sofort zurück)
        +------------------------+
        |
        v
   DeepAgent-Läufe als asyncio.Tasks (je Task ein Thread "task-…")
```

### Voice -> Task (Delegation)

- `start_task(name, description)` legt einen `TaskRecord` an und startet den DeepAgent per
  `asyncio.create_task`. Das Tool kehrt sofort zurück, der Voice-Agent blockiert nie.
- `list_tasks()` liefert Status, Laufzeit und aktuellen Schritt (z.B. "führt slow_work aus").
- `cancel_task(name)` ruft `task.cancel()` auf. Den Abbruch bestätigt der Voice-Agent direkt
  im Tool-Ergebnis. Eine zusätzliche Benachrichtigung gibt es nur bei *unerwarteten* Abbrüchen,
  sonst käme die Ansage doppelt.
- Tasks werden über ihren Namen angesprochen ("die Recherche"): exakter Treffer, Teilstring,
  dann unscharfer Vergleich. Bei Mehrdeutigkeit kommt eine Rückfrage.

### Task -> Voice (Rückmeldung, ohne Queue und WebSockets)

Ist ein Task fertig, schlägt er fehl, wird er unerwartet abgebrochen oder pausiert er wegen einer
Rückfrage, ruft der TaskManager `notify_voice(text)` auf. Das startet direkt einen neuen Lauf des
Voice-Agenten auf dessen bestehendem Thread:

```python
await voice_agent.astream({"messages": [make_notification(text, provider)]}, voice_config, ...)
```

Die Antwort läuft wie jede andere durch die TTS-Pipeline. `make_notification` erzeugt bei OpenAI
eine `SystemMessage`, bei Anthropic/Gemini (System-Messages nur am Anfang erlaubt) eine
`HumanMessage`. Beide tragen das Präfix `[TASK-BENACHRICHTIGUNG]`, das der Systemprompt erklärt.
Jede Benachrichtigung läuft als eigener asyncio-Task, damit der DeepAgent nie auf den Voice-Lock
wartet und ein Task-Abbruch keine laufende Ansage abschneidet.

### Synchronisation

- Ein `asyncio.Lock` schützt den Voice-Thread: Nutzerbeiträge und Benachrichtigungen laufen nie
  gleichzeitig. Wartende Läufe kommen in Ankunftsreihenfolge dran.
- Der Lock wird erst freigegeben, wenn `TTS.speak()` zurückkehrt, also die Ausgabe zu Ende ist.
  Ansagen überlagern sich dadurch nie.
- Während der Nutzer aufnimmt (inkl. Transkription), warten Benachrichtigungen
  (`recording_started()` / `recording_stopped()`).

### Rückfragen (Human-in-the-Loop)

1. `dummy_action` ist per `interrupt_on={"dummy_action": {"allowed_decisions": ["approve", "reject"]}}`
   geschützt. Der DeepAgent pausiert mit einem Interrupt vom Typ `HITLRequest`
   (`{"action_requests": [...], "review_configs": [...]}`).
2. Der TaskManager liest die offenen Interrupts per `aget_state()`, setzt den Status auf
   "wartet auf Antwort" und ruft `notify_voice` mit einer vorlesbaren Frage auf.
3. Der Voice-Agent stellt die Frage und gibt die Antwort per
   `answer_task(name, decision, message)` zurück.
4. Der TaskManager setzt den DeepAgent im Hintergrund fort:
   `Command(resume={"decisions": [{"type": "approve"}]})` bzw.
   `{"type": "reject", "message": "..."}`. Bei "Nein" wird die Aktion nicht ausgeführt, der
   Task beendet sich und meldet das.

### Gedächtnis und Persistenz

- Beide Agenten bekommen **denselben** LangGraph-Store (`InMemoryStore`). `slow_work` legt seine
  Ergebnisse dort unter `("task_results",)` ab.
- Checkpointer: je Agent ein eigener `InMemorySaver`, Threads strikt getrennt
  (Voice: `voice-…`, Task: `thread_id = task_id`). Für Postgres nur `persistence.py` anpassen.

### Fehler

Fehler im Task-Agenten werden gefangen, der Task bekommt den Status "fehlgeschlagen" und der
Voice-Agent erhält eine verständliche Benachrichtigung. Fehler im Voice-Agenten oder in TTS führen
zu einer kurzen Entschuldigung bzw. einem Log-Eintrag, das Gespräch läuft weiter.

## Akzeptanztests

`tests/test_acceptance.py` deckt mit Fake-Modellen ab:

1. Smalltalk wird direkt beantwortet, ohne Task.
2. "Recherchier mir X" startet einen Task, der Voice-Agent antwortet sofort und bleibt ansprechbar.
3. Nach Ende des Tasks meldet sich der Voice-Agent ungefragt mit dem Ergebnis.
4. Zwei Tasks laufen parallel, "Wie weit ist die Recherche?" liefert den Status.
5. `dummy_action` stellt eine Rückfrage, "Ja, mach" führt sie aus, "Nein" beendet sauber ohne Aktion.
6. `cancel_task` bricht einen laufenden Task ab, der Voice-Agent bestätigt.
7. Eine Benachrichtigung während eines laufenden Voice-Laufs wird erst danach verarbeitet
   (plus: Benachrichtigung wartet während einer Aufnahme).

Zusätzlich: Task-Fehler als Benachrichtigung, Anthropic-Variante der Benachrichtigung,
Satz-Splitter, `make_notification`.
