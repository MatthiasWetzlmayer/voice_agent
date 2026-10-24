"""Start: freihändiges Gespräch (Standard) oder Textmodus mit --no-audio."""

import argparse
import asyncio
import logging
import os
import signal
import threading

from dotenv import load_dotenv

from .agents import make_model
from .assistant import Assistant
from .audio import ConsoleTTS, PiperTTS, listen


def stop_on_ctrl_c(task: asyncio.Task) -> None:
    """Strg+C bricht den Haupt-Task ab, auch während langer Ladephasen.

    Weitere Signale werden ignoriert (`uv run` reicht Strg+C doppelt weiter). Steckt das
    Programm gerade in einem langen blockierenden Aufruf oder hängt das Aufräumen, wird
    nach 5 Sekunden hart beendet.
    """
    loop = asyncio.get_running_loop()

    def handler(signum, frame) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        loop.call_soon_threadsafe(task.cancel)
        force = threading.Timer(5, lambda: (print("\nTschüss!", flush=True), os._exit(130)))
        force.daemon = True
        force.start()

    signal.signal(signal.SIGINT, handler)


async def main(no_audio: bool, debug: bool) -> None:
    stop_on_ctrl_c(asyncio.current_task())
    voice_spec = os.getenv("VOICE_MODEL", "ollama:qwen3:4b")
    task_spec = os.getenv("TASK_MODEL", "ollama:qwen3:4b")
    assistant = Assistant(
        make_model(voice_spec),
        make_model(task_spec),
        ConsoleTTS() if no_audio else PiperTTS(),
        provider=voice_spec.split(":")[0],
    )
    try:
        if not no_audio:
            await listen(assistant, debug)
        while True:  # Textmodus
            text = (await asyncio.to_thread(input, "> ")).strip()
            if text:
                await assistant.user_says(text)
    except (EOFError, asyncio.CancelledError):  # Ende der Eingabe bzw. Strg+C
        print("\nTschüss!")
    finally:
        await assistant.shutdown()


def run() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-17s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # deepagents loggt eine harmlose Warnung samt Traceback, wenn das optionale Video-Extra fehlt
    logging.getLogger("deepagents.middleware._video").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description="Sprachassistent mit Voice- und Task-Agent")
    parser.add_argument("--no-audio", action="store_true", help="Text tippen statt sprechen, Ausgabe nur auf der Konsole")
    parser.add_argument("--debug", action="store_true", help="Mikrofon-Pegel und alle Erkennungsschritte anzeigen")
    args = parser.parse_args()
    asyncio.run(main(args.no_audio, args.debug))


if __name__ == "__main__":
    run()
