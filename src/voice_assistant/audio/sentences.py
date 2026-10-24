"""Token-Stream in Sätze schneiden, damit TTS früh beginnen kann."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

_ABBREVIATIONS = {
    "z.", "b.", "bzw.", "ca.", "dr.", "nr.", "usw.", "etc.", "evtl.", "ggf.",
    "inkl.", "vgl.", "d.", "h.", "u.", "a.", "mio.", "mrd.", "st.", "str.",
}
_BOUNDARY = re.compile(r"([.!?…]+)(\s+)|(\n+)")


class SentenceSplitter:
    """Sammelt Text-Fragmente und gibt vollständige Sätze zurück."""

    def __init__(self, min_chars: int = 2) -> None:
        self._buffer = ""
        self.min_chars = min_chars

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences: list[str] = []
        start = 0
        for match in _BOUNDARY.finditer(self._buffer):
            end = match.end(1) if match.group(1) else match.start(3)
            candidate = self._buffer[start:end].strip()
            last_word = candidate.rsplit(maxsplit=1)[-1].lower() if candidate else ""
            if match.group(1) and (last_word in _ABBREVIATIONS or re.fullmatch(r"\d+\.", last_word)):
                continue  # "z. B." oder "3. Mai" ist kein Satzende
            if len(candidate) >= self.min_chars:
                sentences.append(candidate)
                start = match.end()
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []


async def buffered(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Entkoppelt Erzeuger (Agent) und Verbraucher (TTS).

    Der Agent streamt weiter, während TTS noch den vorherigen Satz spricht. Fehler des
    Erzeugers werden beim Verbraucher erneut ausgelöst.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    error: list[BaseException] = []

    async def produce() -> None:
        try:
            async for item in source:
                await queue.put(item)
        except BaseException as exc:  # noqa: BLE001 - wird beim Verbraucher erneut ausgelöst
            error.append(exc)
        finally:
            await queue.put(None)

    producer = asyncio.create_task(produce())
    try:
        while (item := await queue.get()) is not None:
            yield item
        if error:
            raise error[0]
    finally:
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
