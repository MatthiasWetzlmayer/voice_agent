"""Audio über die OpenAI-API: Live-Transkription (Realtime) und Sprachausgabe (TTS)."""

import asyncio
import base64
import os
import re
import time
from array import array
from typing import AsyncIterator

from openai import AsyncOpenAI

SAMPLE_RATE = 24_000  # Realtime-Eingabe und TTS-PCM nutzen beide 24 kHz, 16 bit, mono


async def listen(assistant, debug: bool = False) -> None:
    """Hört dauerhaft zu, zeigt die Live-Transkription und gibt fertige Sätze an den Agenten.

    Das Mikrofon streamt laufend zur Realtime API (Transkriptions-Session). Der Server erkennt
    per VAD, wann man anfängt und aufhört zu sprechen. Solange der Assistent selbst spricht,
    wird das Mikrofon nicht gesendet, damit er sich nicht selbst hört.
    """
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    mic: asyncio.Queue[bytes] = asyncio.Queue()
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=2400,
        callback=lambda data, *_: loop.call_soon_threadsafe(mic.put_nowait, bytes(data)),
    )

    print(f"Mikrofon: {sd.query_devices(kind='input')['name']}")

    async with AsyncOpenAI().realtime.connect(extra_query={"intent": "transcription"}) as conn:
        await conn.session.update(session={
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "transcription": {"model": os.getenv("STT_MODEL", "gpt-4o-mini-transcribe"), "language": "de"},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 700},
            }},
        })

        async def send_mic():
            last_sound, last_report = time.monotonic(), 0.0
            while True:
                chunk = await mic.get()
                level = max(map(abs, array("h", chunk)), default=0)
                now = time.monotonic()
                if level > 500:
                    last_sound = now
                elif now - last_sound > 5:
                    print("Warnung: Das Mikrofon liefert seit 5 s nur Stille (Eingabegerät/Berechtigung prüfen).")
                    last_sound = now
                if debug and now - last_report > 1:
                    print(f"  [debug] Mikrofon-Pegel {level}")
                    last_report = now
                if not assistant.speaking:
                    await conn.input_audio_buffer.append(audio=base64.b64encode(chunk).decode())

        def sender_done(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                print(f"Fehler beim Senden des Mikrofons: {task.exception()!r}")

        sender = asyncio.create_task(send_mic())
        sender.add_done_callback(sender_done)
        turns = set()  # Referenzen halten, damit laufende Gesprächsrunden nicht eingesammelt werden
        with stream:
            print("Ich höre zu ... (Strg+C beendet)")
            try:
                async for event in conn:
                    if debug and not event.type.endswith(".delta"):
                        print(f"  [debug] Event: {event.type}")
                    if event.type == "input_audio_buffer.speech_started":
                        assistant.user_idle.clear()  # Benachrichtigungen warten
                        print("  (höre dich ...)")
                    elif event.type == "conversation.item.input_audio_transcription.delta":
                        print(event.delta or "", end="", flush=True)  # Live-Transkription
                    elif event.type == "conversation.item.input_audio_transcription.completed":
                        print()
                        assistant.user_idle.set()
                        if event.transcript.strip():
                            turn = asyncio.create_task(assistant.user_says(event.transcript.strip()))
                            turns.add(turn)
                            turn.add_done_callback(turns.discard)
                    elif event.type == "conversation.item.input_audio_transcription.failed":
                        assistant.user_idle.set()  # sonst warten Benachrichtigungen für immer
                        e = event.error
                        print(f"Transkription fehlgeschlagen: {e.message} (code={e.code}, param={e.param})")
                    elif event.type == "error":
                        print(f"Realtime-Fehler: {event.error.message}")
                print("Verbindung zur Realtime API wurde beendet.")
            finally:
                sender.cancel()


class OpenAITTS:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.model = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
        self.voice = os.getenv("TTS_VOICE", "alloy")

    async def speak(self, text_stream: AsyncIterator[str]) -> None:
        import sounddevice as sd

        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as out:
            async for sentence in text_stream:
                print(f"  Assistent: {sentence}")
                async with self.client.audio.speech.with_streaming_response.create(
                    model=self.model, voice=self.voice, input=sentence, response_format="pcm"
                ) as response:
                    async for chunk in response.iter_bytes(4800):
                        await asyncio.to_thread(out.write, chunk[: len(chunk) // 2 * 2])


class ConsoleTTS:
    """Für --no-audio: gibt die Antwort nur als Text aus."""

    async def speak(self, text_stream: AsyncIterator[str]) -> None:
        async for sentence in text_stream:
            print(f"  Assistent: {sentence}")


async def sentences(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Schneidet einen Token-Stream in Sätze, damit TTS früh anfangen kann."""
    buffer = ""
    async for token in tokens:
        buffer += token
        *done, buffer = re.split(r"(?<=[.!?])\s+", buffer)
        for sentence in done:
            yield sentence
    if buffer.strip():
        yield buffer.strip()
