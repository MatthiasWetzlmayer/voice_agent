"""Mikrofonaufnahme für Push-to-Talk (sounddevice)."""

from __future__ import annotations

import io
import threading
import wave

SAMPLE_RATE = 16_000


class Recorder:
    """`start()` beginnt die Aufnahme, `stop()` liefert WAV-Bytes."""

    def __init__(self, samplerate: int = SAMPLE_RATE) -> None:
        self.samplerate = samplerate
        self._frames: list[bytes] = []
        self._lock = threading.Lock()
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._frames = []

        def callback(indata, frames, time, status) -> None:  # läuft im Audio-Thread
            with self._lock:
                self._frames.append(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self.samplerate, channels=1, dtype="int16", callback=callback
        )
        self._stream.start()

    def stop(self) -> bytes:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            pcm = b"".join(self._frames)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.samplerate)
            wav.writeframes(pcm)
        return buffer.getvalue()

    def duration(self, wav_bytes: bytes) -> float:
        return max(0, len(wav_bytes) - 44) / 2 / self.samplerate
