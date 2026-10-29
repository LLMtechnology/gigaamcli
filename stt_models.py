import gc
import importlib
import pathlib

import soundfile
import torch
from loguru import logger
from tqdm import tqdm

import config


def resolve_device(requested):
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен в установленном PyTorch. Используйте --device cpu")
    return requested


def check_diarization():
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Sortformer MLX требует Apple Silicon с доступным Metal GPU")
    importlib.import_module("mlx_audio.vad.models.sortformer")


def transcribe(audio, chunks, scratch, model_name, device):
    import gigaam

    torch.set_num_threads(config.TORCH_THREADS)
    logger.info("GigaAM {}: {}", model_name, device)
    model = gigaam.load_model(
        model_name,
        device=device,
        fp16_encoder=device == "mps",
        use_flash=False,
        download_root=str(config.MODEL_CACHE),
    )
    words = []
    chunk_path = pathlib.Path(scratch) / "chunk.wav"
    for start, end in tqdm(chunks, desc="Распознавание", unit="фрагм"):
        left = round(start * config.SAMPLE_RATE)
        right = round(end * config.SAMPLE_RATE)
        soundfile.write(chunk_path, audio[left:right], config.SAMPLE_RATE, subtype="PCM_16")
        with torch.inference_mode():
            result = model.transcribe(str(chunk_path), word_timestamps=True)
        if result.text.strip() and not result.words:
            raise RuntimeError("GigaAM не вернула таймкоды слов для непустого текста")
        for word in result.words or []:
            word_start = min(end, max(start, start + float(word.start)))
            word_end = min(end, max(word_start, start + float(word.end)))
            if word.text.strip():
                words.append({"text": word.text.strip(), "start": round(word_start, 3), "end": round(word_end, 3)})
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return words


def diarize(audio, threshold):
    import mlx.core as mx
    import mlx_audio.vad

    if not mx.metal.is_available():
        raise RuntimeError("Sortformer MLX требует Apple Silicon с доступным Metal GPU")
    mx.set_default_device(mx.gpu)
    logger.info("Диаризация: Sortformer v2.1 / MLX Metal (до 4 говорящих)")
    model = mlx_audio.vad.load(config.DIARIZATION_MODEL)
    turns = []
    state = model.init_streaming_state()
    frame_seconds = model.config.processor_config.hop_length * model.config.fc_encoder_config.subsampling_factor
    frame_seconds /= model.config.processor_config.sampling_rate
    chunk_samples = round(config.DIARIZATION_CHUNK_SECONDS * config.SAMPLE_RATE)
    duration = len(audio) / config.SAMPLE_RATE
    for offset in tqdm(range(0, len(audio), chunk_samples), desc="Говорящие", unit="фрагм"):
        chunk = audio[offset:offset + chunk_samples]
        chunk_start = offset / config.SAMPLE_RATE
        chunk_end = min(duration, (offset + len(chunk)) / config.SAMPLE_RATE)
        library_offset = state.frames_processed * frame_seconds
        result, state = model.feed(
            chunk, state, sample_rate=config.SAMPLE_RATE, threshold=threshold, min_duration=0.0, merge_gap=0.2,
        )
        for segment in result.segments:
            start = max(chunk_start, chunk_start + float(segment.start) - library_offset)
            end = min(chunk_end, chunk_start + float(segment.end) - library_offset)
            if end > start:
                turns.append({"start": start, "end": end, "speaker": int(segment.speaker)})
    del model, state, result
    gc.collect()
    mx.clear_cache()
    turns.sort(key=lambda turn: (turn["speaker"], turn["start"]))
    merged = []
    for turn in turns:
        if merged and merged[-1]["speaker"] == turn["speaker"] and turn["start"] <= merged[-1]["end"] + 0.2:
            merged[-1]["end"] = max(merged[-1]["end"], turn["end"])
        else:
            merged.append(dict(turn))
    return sorted(
        [turn for turn in merged if turn["end"] - turn["start"] >= 0.15],
        key=lambda turn: (turn["start"], turn["speaker"]),
    )

