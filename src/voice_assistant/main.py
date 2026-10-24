"""Terminal-Frontend: Push-to-Talk (Enter) oder getippter Text."""

from __future__ import annotations

import argparse
import asyncio
import logging

from openai import OpenAIError

from .app import build_app, init_models
from .audio import ConsoleTTS
from .config import Settings
from .logging_setup import setup_logging
from .notifications import provider_of

log = logging.getLogger("va.system")
audio_log = logging.getLogger("va.audio")

HELP = """\
Bedienung:
  Enter (leere Zeile)  Aufnahme starten, nochmal Enter zum Stoppen (Push-to-Talk)
  Text + Enter         Text direkt an den Voice-Agenten senden
  /tasks               Task-Übersicht anzeigen
  /quit                Beenden
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprachassistent mit Voice- und Task-Agent")
    parser.add_argument("--no-audio", action="store_true", help="Kein Mikrofon/Lautsprecher, Ausgabe nur auf der Konsole")
    parser.add_argument("--fake", action="store_true", help="Regelbasierte Fake-Modelle statt echter LLMs (impliziert --no-audio)")
    parser.add_argument("--fake-work-seconds", type=float, default=5.0, help="Dauer von slow_work im Fake-Modus")
    return parser.parse_args(argv)


async def amain(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)

    no_audio = args.no_audio or args.fake
    if not no_audio:
        try:
            import sounddevice  # noqa: F401
        except (ImportError, OSError) as exc:
            log.warning("Audio nicht verfügbar (%s), weiter im Textmodus (--no-audio)", exc)
            no_audio = True
    stt = recorder = None
    if no_audio:
        tts = ConsoleTTS()
    else:
        from .audio.openai_audio import OpenAISTT, OpenAITTS
        from .audio.recorder import Recorder

        tts = OpenAITTS(settings.tts_model, settings.tts_voice, settings.tts_instructions)
        stt = OpenAISTT(settings.stt_model, settings.stt_language)
        recorder = Recorder()

    if args.fake:
        from .fakes import make_fake_models

        voice_model, task_model = make_fake_models(work_seconds=args.fake_work_seconds)
        provider = "openai"
        log.info("Fake-Modelle aktiv (keine API-Aufrufe)")
    else:
        voice_model, task_model = init_models(settings)
        provider = provider_of(settings.voice_model)
        log.info("Voice-Modell: %s | Task-Modell: %s", settings.voice_model, settings.task_model)

    app = build_app(voice_model, task_model, tts, provider=provider)
    session = app.session
    log.info("Voice-Thread: %s | Audio: %s", session.thread_id, "aus" if no_audio else "an")
    print(HELP)

    try:
        while True:
            line = (await asyncio.to_thread(input, "> ")).strip()
            if line in ("/quit", "/exit"):
                break
            if line == "/tasks":
                for record in app.task_manager.all_tasks():
                    print("  " + record.summary())
                continue
            if line:
                await session.handle_user_text(line)
                continue
            if recorder is None:
                print("Kein Mikrofon im --no-audio-Modus. Bitte Text eingeben.")
                continue
            text = await record_and_transcribe(session, recorder, stt)
            if text:
                await session.handle_user_text(text)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        log.info("Beende, breche laufende Tasks ab ...")
        await app.task_manager.shutdown()


async def record_and_transcribe(session, recorder, stt) -> str:
    session.recording_started()  # Benachrichtigungen warten, solange aufgenommen wird
    try:
        recorder.start()
        audio_log.info("Aufnahme läuft ... Enter zum Stoppen")
        await asyncio.to_thread(input)
        audio = recorder.stop()
        if recorder.duration(audio) < 0.3:
            audio_log.info("Aufnahme zu kurz, ignoriert")
            return ""
        audio_log.info("transkribiere ...")
        return await stt.transcribe(audio)
    except Exception:  # noqa: BLE001
        audio_log.exception("Aufnahme oder Transkription fehlgeschlagen")
        return ""
    finally:
        # Erst nach der Transkription freigeben, damit keine Ansage dazwischenfunkt.
        session.recording_stopped()


def run() -> None:
    try:
        asyncio.run(amain(parse_args()))
    except OpenAIError as exc:
        raise SystemExit(f"OpenAI-Konfiguration fehlt oder ist ungültig: {exc}\nTipp: .env anlegen oder mit --fake testen.") from None


if __name__ == "__main__":
    run()
