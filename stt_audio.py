import math
import pathlib
import shutil
import subprocess

import numpy as np
import silero_vad
import soundfile
import torch

import config


def decode_audio(input_path, wav_path):
    input_path = pathlib.Path(input_path).expanduser().resolve()
    wav_path = pathlib.Path(wav_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {input_path}")
    if input_path == wav_path:
        raise ValueError("The decoded WAV path must differ from the input file")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found. Install it with: brew install ffmpeg")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(input_path), "-vn", "-ac", "1", "-ar", str(config.SAMPLE_RATE),
            "-c:a", "pcm_s16le", "-f", "wav", str(wav_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"ffmpeg could not decode {input_path.name}: {detail}")
    audio, sample_rate = soundfile.read(wav_path, dtype="float32")
    if sample_rate != config.SAMPLE_RATE or audio.ndim != 1:
        raise RuntimeError(f"Decoded audio must be mono at {config.SAMPLE_RATE} Hz")
    if not audio.size:
        raise ValueError(f"Audio file contains no samples: {input_path}")
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)


def speech_chunks(audio):
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("Speech detection expects mono audio")
    if not audio.size:
        return []
    sample_rate = config.SAMPLE_RATE
    padding = config.CHUNK_PADDING_SECONDS
    core_limit = int((config.MAX_CHUNK_SECONDS - 2 * padding) * sample_rate)
    if padding < 0 or core_limit <= 0:
        raise ValueError("Chunk duration must exceed twice the nonnegative padding")
    vad = silero_vad.load_silero_vad(onnx=True)
    speech = silero_vad.get_speech_timestamps(
        torch.from_numpy(audio),
        vad,
        sampling_rate=sample_rate,
        max_speech_duration_s=config.MAX_CHUNK_SECONDS - 2 * padding,
        min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
        speech_pad_ms=0,
        return_seconds=False,
    )
    spans = []
    previous_end = 0
    for item in sorted(speech, key=lambda item: item["start"]):
        start = max(previous_end, min(audio.size, int(item["start"])))
        end = min(audio.size, int(item["end"]))
        if end <= start:
            continue
        count = math.ceil((end - start) / core_limit)
        for index in range(count):
            left = start + (end - start) * index // count
            right = start + (end - start) * (index + 1) // count
            spans.append((left, right))
        previous_end = end
    merged = []
    max_gap = sample_rate
    for start, end in spans:
        if merged and start - merged[-1][1] < max_gap and end - merged[-1][0] <= core_limit:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    chunks = []
    pad_samples = int(padding * sample_rate)
    for index, (start, end) in enumerate(merged):
        previous_end = merged[index - 1][1] if index else 0
        next_start = merged[index + 1][0] if index + 1 < len(merged) else audio.size
        start -= min(pad_samples, (start - previous_end) // 2)
        end += min(pad_samples, (next_start - end) // 2)
        chunks.append((start / sample_rate, end / sample_rate))
    return chunks
