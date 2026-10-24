"""OpenAI-Adapter: `gpt-4o-mini-transcribe` (STT) und `gpt-4o-mini-tts` (TTS, PCM)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

log = logging.getLogger("va.audio")

PCM_SAMPLE_RATE = 24_000  # OpenAI liefert PCM als 24 kHz, 16 bit, mono


class OpenAISTT:
    def __init__(self, model: str, language: str | None = "de", client: AsyncOpenAI | None = None) -> None:
        self.model = model
        self.language = language
        self.client = client or AsyncOpenAI()

    async def transcribe(self, audio: bytes) -> str:
        kwargs = {"language": self.language} if self.language else {}
        result = await self.client.audio.transcriptions.create(
            model=self.model, file=("aufnahme.wav", audio, "audio/wav"), **kwargs
        )
        return result.text.strip()


class OpenAITTS:
    """Synthetisiert Satz für Satz und spielt die PCM-Daten gestreamt ab."""

    def __init__(
        self,
        model: str,
        voice: str,
        instructions: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.client = client or AsyncOpenAI()

    async def speak(self, text_stream: AsyncIterator[str]) -> None:
        import sounddevice as sd  # erst hier, damit --no-audio ohne PortAudio läuft

        stream = sd.RawOutputStream(samplerate=PCM_SAMPLE_RATE, channels=1, dtype="int16")
        stream.start()
        try:
            async for sentence in text_stream:
                await self._play_sentence(stream, sentence)
            # Puffer der Soundkarte ausspielen lassen, bevor der Lock frei wird.
            await asyncio.to_thread(stream.stop)
        finally:
            stream.close()

    async def _play_sentence(self, stream, sentence: str) -> None:
        kwargs = {"instructions": self.instructions} if self.instructions else {}
        leftover = b""
        async with self.client.audio.speech.with_streaming_response.create(
            model=self.model, voice=self.voice, input=sentence, response_format="pcm", **kwargs
        ) as response:
            async for chunk in response.iter_bytes(4096):
                data = leftover + chunk
                usable = len(data) - len(data) % 2  # nur ganze 16-bit-Samples schreiben
                leftover = data[usable:]
                if usable:
                    await asyncio.to_thread(stream.write, data[:usable])
