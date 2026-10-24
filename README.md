# Voice Assistant

Sprachassistent aus zwei Agenten in einem Prozess:

- **Voice-Agent** (`create_agent`, Ollama): führt das Gespräch.
- **Task-Agent** (`create_deep_agent`, Ollama): erledigt längere Aufgaben im Hintergrund
  (bisher nur Dummy-Tools `slow_work` und `dummy_action`).

Man redet einfach drauflos: Das Mikrofon hört dauerhaft zu, die OpenAI Realtime API
transkribiert live und erkennt Sprechpausen. Die Antwort wird satzweise per OpenAI-TTS vorgelesen.

## Start

```bash
ollama pull qwen3:8b
cp .env.example .env            # OPENAI_API_KEY eintragen
uv sync
uv run voice-assistant          # sprechen
uv run voice-assistant --no-audio   # tippen statt sprechen
uv run pytest
```

Für das Mikrofon braucht es PortAudio (`apt install libportaudio2` bzw. `brew install portaudio`).
Ohne Kopfhörer wird das Mikrofon stummgeschaltet, solange der Assistent spricht.

## Dateien

| Datei          | Inhalt                                                               |
|----------------|----------------------------------------------------------------------|
| `agents.py`    | Prompts, Dummy-Tools, Modellauswahl, Erzeugung beider Agenten        |
| `assistant.py` | Gespräch, Hintergrund-Tasks und die Kommunikation zwischen beiden    |
| `audio.py`     | Live-Transkription, TTS, Zerlegung in Sätze                          |
| `main.py`      | Start                                                                |

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
