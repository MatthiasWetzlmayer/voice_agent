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

## Start (macOS)

```bash
brew install portaudio          # für das Mikrofon (PyAudio)
ollama pull qwen3:4b
cp .env.example .env
uv sync                         # Python 3.12, RealtimeSTT unterstützt noch kein 3.13
uv run voice-assistant          # sprechen
uv run voice-assistant --no-audio   # tippen statt sprechen
uv run pytest
```

Beim ersten Start werden die Whisper-Modelle und die Piper-Stimme heruntergeladen. Das Terminal
braucht die Mikrofon-Berechtigung (Systemeinstellungen → Datenschutz & Sicherheit → Mikrofon).
Solange der Assistent spricht, ist das Mikrofon stumm, damit er sich nicht selbst hört.

Speicher auf einem 16-GB-Mac, grob: `qwen3:4b` mit 16k Kontext ca. 5 GB, Whisper `small` und
`tiny` ca. 1 GB, Piper unter 100 MB.

## Dateien

| Datei          | Inhalt                                                               |
|----------------|----------------------------------------------------------------------|
| `agents.py`    | Prompts, Dummy-Tools, Modellauswahl, Erzeugung beider Agenten        |
| `assistant.py` | Gespräch, Hintergrund-Tasks und die Kommunikation zwischen beiden    |
| `audio.py`     | Live-Transkription (RealtimeSTT), Sprachausgabe (Piper), Satz-Splitter |
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
