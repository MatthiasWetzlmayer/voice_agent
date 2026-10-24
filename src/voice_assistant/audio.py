"""Lokale Sprach-Ein- und -Ausgabe: RealtimeSTT (faster-whisper) und Piper."""

import asyncio
import os
import re
from pathlib import Path
from typing import AsyncIterator


async def listen(assistant) -> None:
    """Hört dauerhaft zu, zeigt die Live-Transkription und gibt fertige Sätze an den Agenten.

    RealtimeSTT erkennt per VAD, wann man spricht. Während des Sprechens transkribiert ein
    kleines Whisper-Modell live mit, nach der Sprechpause liefert ein größeres den fertigen Text.
    Solange der Assistent spricht, ist das Mikrofon stumm, damit er sich nicht selbst hört.
    """
    from RealtimeSTT import AudioToTextRecorder

    loop = asyncio.get_running_loop()
    recorder = AudioToTextRecorder(
        model=os.getenv("STT_MODEL", "small"),
        realtime_model_type=os.getenv("STT_REALTIME_MODEL", "tiny"),
        language="de",
        device="cpu",  # faster-whisper kann auf dem Mac die GPU nicht nutzen
        compute_type="int8",
        enable_realtime_transcription=True,
        on_realtime_transcription_update=lambda text: print(f"\r  ... {text}", end="", flush=True),
        # Callbacks kommen aus Threads von RealtimeSTT, daher threadsafe an die Loop übergeben
        on_recording_start=lambda: loop.call_soon_threadsafe(assistant.user_idle.clear),
        post_speech_silence_duration=0.7,
        spinner=False,
    )
    assistant.on_speaking = lambda speaking: recorder.set_microphone(not speaking)
    turns = set()  # Referenzen halten, damit laufende Gesprächsrunden nicht eingesammelt werden
    print("Ich höre zu ... (Strg+C beendet)")
    try:
        while True:
            text = (await asyncio.to_thread(recorder.text)).strip()
            print()
            assistant.user_idle.set()
            if text:
                turn = asyncio.create_task(assistant.user_says(text))
                turns.add(turn)
                turn.add_done_callback(turns.discard)
    finally:
        recorder.shutdown()


class PiperTTS:
    def __init__(self):
        from piper import PiperVoice
        from piper.download_voices import download_voice

        name = os.getenv("PIPER_VOICE", "de_DE-thorsten-medium")
        folder = Path("voices")
        if not (folder / f"{name}.onnx").exists():
            print(f"Lade Piper-Stimme {name} ...")
            folder.mkdir(exist_ok=True)
            download_voice(name, folder)
        self.voice = PiperVoice.load(folder / f"{name}.onnx")

    async def speak(self, text_stream: AsyncIterator[str]) -> None:
        import sounddevice as sd

        with sd.RawOutputStream(samplerate=self.voice.config.sample_rate, channels=1, dtype="int16") as out:
            async for sentence in text_stream:
                print(f"  Assistent: {sentence}")
                chunks = await asyncio.to_thread(lambda: list(self.voice.synthesize(sentence)))
                for chunk in chunks:
                    await asyncio.to_thread(out.write, chunk.audio_int16_bytes)


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
