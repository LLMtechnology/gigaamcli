import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "v3_e2e_rnnt"
DEVICE = "auto"
SAMPLE_RATE = 16000
MAX_CHUNK_SECONDS = 24.0
CHUNK_PADDING_SECONDS = 0.2
VAD_MIN_SPEECH_MS = 150
VAD_MIN_SILENCE_MS = 200
MODEL_CACHE = Path.home() / ".cache" / "gigaam"
DIARIZATION_MODEL = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32"
DIARIZATION_CHUNK_SECONDS = 5.0
DIARIZATION_THRESHOLD = 0.5
PHRASE_PAUSE_SECONDS = 0.6
MAX_PHRASE_SECONDS = 15.0
OUTPUT_FORMATS = ("txt", "srt", "json")
TORCH_THREADS = 4
