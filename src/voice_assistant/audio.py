"""Audio über die OpenAI-API: Live-Transkription (Realtime) und Sprachausgabe (TTS)."""

import asyncio
import base64
import os
import re
from typing import AsyncIterator

from openai import AsyncOpenAI

SAMPLE_RATE = 24_000  # Realtime-Eingabe und TTS-PCM nutzen beide 24 kHz, 16 bit, mono


async def listen(assistant) -> None:
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
            while True:
                chunk = await mic.get()
                if not assistant.speaking:
                    await conn.input_audio_buffer.append(audio=base64.b64encode(chunk).decode())

        sender = asyncio.create_task(send_mic())
        turns = set()  # Referenzen halten, damit laufende Gesprächsrunden nicht eingesammelt werden
        with stream:
            print("Ich höre zu ... (Strg+C beendet)")
            try:
                async for event in conn:
                    if event.type == "input_audio_buffer.speech_started":
                        assistant.user_idle.clear()  # Benachrichtigungen warten
                    elif event.type == "conversation.item.input_audio_transcription.delta":
                        print(event.delta or "", end="", flush=True)  # Live-Transkription
                    elif event.type == "conversation.item.input_audio_transcription.completed":
                        print()
                        assistant.user_idle.set()
                        if event.transcript.strip():
                            turn = asyncio.create_task(assistant.user_says(event.transcript.strip()))
                            turns.add(turn)
                            turn.add_done_callback(turns.discard)
                    elif event.type == "error":
                        print(f"Realtime-Fehler: {event.error.message}")
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
