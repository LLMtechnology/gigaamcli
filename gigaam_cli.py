#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time

from loguru import logger

import config
def arguments():
    parser = argparse.ArgumentParser(
        prog="gigaam",
        description="Локальное распознавание GigaAM на MPS + разделение до 4 говорящих Sortformer на MLX/Metal.",
        epilog='Пример: gigaam "запись.mp3" --output-dir ./transcripts',
    )
    parser.add_argument("audio", nargs="?", type=pathlib.Path, help="аудио/видео: mp3, wav, m4a, mp4 и т. д.")
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu"), default=config.DEVICE, help="устройство GigaAM (auto: MPS → CPU)",
    )
    parser.add_argument("--model", choices=("v3_e2e_rnnt", "v3_e2e_ctc"), default=config.MODEL_NAME, help="модель GigaAM")
    parser.add_argument("--no-diarize", action="store_true", help="только текст, без разделения говорящих")
    parser.add_argument("--speaker-names", nargs="+", default=[], metavar="ИМЯ", help="имена по порядку появления голосов")
    parser.add_argument(
        "--threshold", type=float, default=config.DIARIZATION_THRESHOLD, help="порог активности Sortformer (0–1)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=pathlib.Path, help="каталог результатов (по умолчанию рядом с исходным файлом)",
    )
    parser.add_argument(
        "--format", choices=("all", *config.OUTPUT_FORMATS), default="all", help="формат файлов (по умолчанию all)",
    )
    parser.add_argument("--force", action="store_true", help="перезаписать существующие результаты")
    parser.add_argument("--check", action="store_true", help="показать устройства и версии, не загружая модели")
    parser.add_argument("--offline", action="store_true", help="использовать скачанные модели без обращений к Hugging Face")
    parser.add_argument("--debug", action="store_true", help="полный traceback при ошибке")
    args = parser.parse_args()
    if not args.check and args.audio is None:
        parser.error("укажите имя аудиофайла")
    if not 0 < args.threshold < 1:
        parser.error("--threshold должен быть между 0 и 1")
    if len(args.speaker_names) > 4:
        parser.error("Sortformer поддерживает до 4 говорящих")
    return args


def check_installation():
    import mlx.core as mx
    import torch

    print(f"Python: {sys.version.split()[0]}")
    print(f"ffmpeg: {shutil.which('ffmpeg') or 'НЕ НАЙДЕН'}")
    print(f"GigaAM MPS: {torch.backends.mps.is_available()}")
    print(f"Sortformer MLX Metal: {mx.metal.is_available()}")
    for package in ("gigaam", "torch", "torchaudio", "mlx", "mlx-audio", "silero-vad"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"Кеш GigaAM: {config.MODEL_CACHE}")
    print(f"Диаризация: {config.DIARIZATION_MODEL}")


def write_results(paths, rendered, force):
    for file_format, path in paths.items():
        with path.open("w" if force else "x", encoding="utf-8") as output:
            output.write(rendered[file_format])
        logger.info("Сохранено: {}", path)


def run(args):
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.check:
        check_installation()
        return
    import stt_audio
    import stt_models
    import stt_output

    if args.offline:
        for filename in (f"{args.model}.ckpt", f"{args.model}_tokenizer.model"):
            if not (config.MODEL_CACHE / filename).is_file():
                raise FileNotFoundError(f"Нет модели в кеше: {filename}. Сначала запустите без --offline")
    source = args.audio.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Файл не найден: {source}")
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else source.parent)
    formats = config.OUTPUT_FORMATS if args.format == "all" else (args.format,)
    paths = {file_format: output_dir / f"{source.stem}.{file_format}" for file_format in formats}
    for path in paths.values():
        if path == source:
            raise ValueError("Путь результата совпал с исходным файлом")
        if path.exists() and not args.force:
            raise FileExistsError(f"Результат уже существует: {path}. Укажите --force или другой --output-dir")
    device = stt_models.resolve_device(args.device)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="gigaam-") as scratch:
        logger.info("Подготовка аудио: {}", source.name)
        audio = stt_audio.decode_audio(source, pathlib.Path(scratch) / "audio.wav")
        duration = len(audio) / config.SAMPLE_RATE
        logger.info("Длительность: {}", stt_output.format_timestamp(duration))
        chunks = stt_audio.speech_chunks(audio)
        logger.info("Найдено речевых фрагментов: {}", len(chunks))
        words = stt_models.transcribe(audio, chunks, scratch, args.model, device) if chunks else []
        turns = stt_models.diarize(audio, args.threshold) if words and not args.no_diarize else []
        if words and not args.no_diarize and not turns:
            raise RuntimeError("Диаризация не обнаружила голоса. Попробуйте меньший --threshold или --no-diarize")
        if not args.no_diarize:
            words = stt_output.assign_speakers(words, turns)
        segments = stt_output.segments_from_words(words, diarize=not args.no_diarize)
    plain_text = stt_output.render_txt(segments, diarize=not args.no_diarize, speaker_names=args.speaker_names)
    elapsed = time.monotonic() - started
    payload = {
        "source": str(source), "duration": round(duration, 3), "elapsed_seconds": round(elapsed, 2),
        "model": args.model, "device": device,
        "diarization_model": None if args.no_diarize else config.DIARIZATION_MODEL,
        "diarization_device": None if args.no_diarize else "mlx-metal",
        "speaker_names": args.speaker_names, "segments": segments, "words": words, "speaker_turns": turns,
    }
    rendered = {
        "txt": plain_text,
        "srt": stt_output.render_srt(segments, diarize=not args.no_diarize, speaker_names=args.speaker_names),
        "json": json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results(paths, rendered, args.force)
    print(plain_text, end="")
    logger.info("Готово за {:.1f} с; аудио {:.1f} с", elapsed, duration)


def main():
    args = arguments()
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="DEBUG" if args.debug else "INFO")
    try:
        run(args)
    except KeyboardInterrupt:
        logger.error("Остановлено пользователем")
        raise SystemExit(130)
    except Exception as error:
        if args.debug:
            logger.exception("Ошибка: {}", error)
        else:
            logger.error("Ошибка: {}", error)
        raise SystemExit(1)
