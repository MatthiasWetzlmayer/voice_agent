"""Lokale Sprach-Ein- und -Ausgabe: RealtimeSTT (faster-whisper) und Piper."""

import asyncio
import logging
import os
import re
import time
from array import array
from pathlib import Path
from typing import AsyncIterator


async def listen(assistant, debug: bool = False) -> None:
    """Hört dauerhaft zu, zeigt die Live-Transkription und gibt fertige Sätze an den Agenten.

    RealtimeSTT erkennt per VAD, wann man spricht. Während des Sprechens transkribiert ein
    kleines Whisper-Modell live mit, nach der Sprechpause liefert ein größeres den fertigen Text.
    Solange der Assistent spricht, ist das Mikrofon stumm, damit er sich nicht selbst hört.
    Mit debug=True werden Mikrofon-Pegel und alle Erkennungsschritte angezeigt.
    """
    from RealtimeSTT import AudioToTextRecorder

    def show(message: str) -> None:
        if debug:
            print(f"  [debug] {message}", flush=True)

    if debug:
        import pyaudio

        audio = pyaudio.PyAudio()
        show(f"Mikrofon: {audio.get_default_input_device_info()['name']}")
        audio.terminate()

    last_level = [0.0]

    def on_chunk(chunk: bytes) -> None:  # nur im Debug-Modus: Pegel etwa einmal pro Sekunde
        if time.monotonic() - last_level[0] > 1:
            last_level[0] = time.monotonic()
            show(f"Mikrofon-Pegel {max(map(abs, array('h', chunk)), default=0)}")

    def on_recording_start() -> None:  # kommt aus einem Thread von RealtimeSTT
        show("Sprache erkannt, Aufnahme läuft")
        loop.call_soon_threadsafe(assistant.user_idle.clear)

    print("Lade Spracherkennung (beim ersten Start werden die Whisper-Modelle heruntergeladen) ...")
    loop = asyncio.get_running_loop()
    recorder = AudioToTextRecorder(
        model=os.getenv("STT_MODEL", "small"),
        realtime_model_type=os.getenv("STT_REALTIME_MODEL", "tiny"),
        language="de",
        device="cpu",  # faster-whisper kann auf dem Mac die GPU nicht nutzen
        compute_type="int8",
        enable_realtime_transcription=True,
        on_realtime_transcription_update=lambda text: print(f"\r  ... {text}", end="", flush=True),
        on_recording_start=on_recording_start,
        on_recording_stop=lambda: show("Sprechpause erkannt, transkribiere"),
        on_vad_start=lambda: show("VAD: Stimme"),
        on_vad_stop=lambda: show("VAD: Stille"),
        on_recorded_chunk=on_chunk if debug else None,
        post_speech_silence_duration=0.7,
        spinner=False,
        level=logging.DEBUG if debug else logging.WARNING,
        no_log_file=True,  # sonst schreibt RealtimeSTT realtimesst.log in den aktuellen Ordner
    )
    def on_speaking(speaking: bool) -> None:
        show("Mikrofon aus (Assistent spricht)" if speaking else "Mikrofon an")
        recorder.set_microphone(not speaking)

    assistant.on_speaking = on_speaking
    turns = set()  # Referenzen halten, damit laufende Gesprächsrunden nicht eingesammelt werden
    print("Ich höre zu ... (Strg+C beendet)")
    try:
        while True:
            text = (await asyncio.to_thread(recorder.text)).strip()
            print()
            show(f"Fertiger Text: {text!r}")
            assistant.user_idle.set()
            if text:
                turn = asyncio.create_task(assistant.user_says(text))
                turns.add(turn)
                turn.add_done_callback(turns.discard)
    finally:
        # RealtimeSTT loggt beim Herunterfahren einen harmlosen Fehler samt Traceback
        logging.getLogger("realtimestt").setLevel(logging.CRITICAL)
        recorder.shutdown()


class PiperTTS:
    def __init__(self):
        print("Lade Sprachausgabe ...")
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
