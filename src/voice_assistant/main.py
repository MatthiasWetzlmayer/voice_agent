"""Start: freihändiges Gespräch (Standard) oder Textmodus mit --no-audio."""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from .agents import make_model
from .assistant import Assistant
from .audio import ConsoleTTS, OpenAITTS, listen


async def main(no_audio: bool, debug: bool) -> None:
    voice_spec = os.getenv("VOICE_MODEL", "ollama:qwen3:8b")
    task_spec = os.getenv("TASK_MODEL", "ollama:qwen3:8b")
    assistant = Assistant(
        make_model(voice_spec),
        make_model(task_spec, thinking=True),
        ConsoleTTS() if no_audio else OpenAITTS(),
        provider=voice_spec.split(":")[0],
    )
    try:
        if not no_audio:
            await listen(assistant, debug)
        while True:  # Textmodus
            text = (await asyncio.to_thread(input, "> ")).strip()
            if text:
                await assistant.user_says(text)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        await assistant.shutdown()


def run() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-17s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "openai", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Sprachassistent mit Voice- und Task-Agent")
    parser.add_argument("--no-audio", action="store_true", help="Text tippen statt sprechen, Ausgabe nur auf der Konsole")
    parser.add_argument("--debug", action="store_true", help="Mikrofon-Pegel und alle Realtime-Events anzeigen")
    args = parser.parse_args()
    asyncio.run(main(args.no_audio, args.debug))


if __name__ == "__main__":
    run()
