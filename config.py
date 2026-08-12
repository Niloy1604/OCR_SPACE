"""
Configuration settings for Real-Time Optical Character Recognition (OCR).
Automatically loads environment variables from .env file.
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Camera Source Configuration
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))

# OCR.space API Settings (Read from .env file or environment variable)
# Get a free/paid key at https://ocr.space/ocrapi
OCRSPACE_API_KEY = os.environ.get("OCRSPACE_API_KEY", "").strip()
OCRSPACE_ENDPOINT = os.environ.get("OCRSPACE_ENDPOINT", "https://api.ocr.space/parse/image").strip()

# Throttling & Trigger Settings
DETECTION_INTERVAL_SECONDS = float(os.environ.get("DETECTION_INTERVAL_SECONDS", 2.0))
MOTION_STABILITY_ENABLED = True  # Enable frame stability checking before sending to API
MOTION_THRESHOLD = 12.0  # Max average pixel intensity change to consider frame "stable"
FORCE_TRIGGER_ON_STABLE_ONLY = False  # If True, wait until stable; if False, fallback to interval

# OCR Settings
# OCR.space Engine 3's supported Indian languages are Hindi, Bengali,
# Gujarati, Kannada, Malayalam, Marathi, Nepali, and Urdu (Tamil, Telugu,
# Punjabi, Odia, Assamese, and Sanskrit are not in its supported list).
# With more than one hint configured, vision_ocr.py automatically requests
# language=auto so OCR.space can auto-detect across all of them per frame.
LANG_ENV = os.environ.get("LANGUAGE_HINTS", "en,hi,bn,gu,kn,ml,mr,ne,ur")
LANGUAGE_HINTS = [lang.strip() for lang in LANG_ENV.split(",") if lang.strip()]
USE_DOCUMENT_TEXT_DETECTION = False  # Set True for dense text/documents, False for sparse real-world text
JPEG_QUALITY = 85  # JPEG compression quality (1-100) for API payload & network efficiency

# Local Offline Fallback Settings
FALLBACK_ENV = os.environ.get("ENABLE_LOCAL_OFFLINE_FALLBACK", "True").lower()
ENABLE_LOCAL_OFFLINE_FALLBACK = FALLBACK_ENV in ("true", "1", "yes")

# UI Overlay & Visualization Settings
WINDOW_TITLE = "Real-Time OCR.space Vision OCR"
OCR_WINDOW_TITLE = "OCR Progress"
SHOW_BOUNDING_BOXES = True
SHOW_TEXT_LABELS = True
BOX_COLOR_BGR = (0, 255, 0)  # Green bounding box
TEXT_COLOR_BGR = (255, 255, 255)  # White text
TEXT_BG_COLOR_BGR = (0, 0, 0)  # Black text background box
HEADER_BG_COLOR_BGR = (40, 40, 40)  # Dark gray header background bar

# Deduplication Settings
DEDUPLICATE_CONSOLE_OUTPUT = True  # Avoid re-printing identical consecutive text in console

# ------------------------------------------------------------------ #
# Google TTS (Algieba) & Cloud TTS (Chirp 3) Settings
# ------------------------------------------------------------------ #
GOOGLE_API_KEY = (
    os.environ.get("GOOGLE_API_KEY", "").strip()
    or os.environ.get("GEMINI_API_KEY", "").strip()
    or os.environ.get("GOOGLE_TTS_API_KEY", "").strip()
)
GOOGLE_TTS_VOICE = os.environ.get("GOOGLE_TTS_VOICE", "Algieba").strip()
CHIRP3_BACKUP_VOICE = os.environ.get("CHIRP3_BACKUP_VOICE", "en-US-Chirp3-HD-Charon").strip()
GOOGLE_TTS_MODEL = os.environ.get("GOOGLE_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip()
GOOGLE_TTS_TIMEOUT_SECONDS = float(os.environ.get("GOOGLE_TTS_TIMEOUT_SECONDS", "30").strip() or "30")
GOOGLE_TTS_RETRIES = max(1, int(os.environ.get("GOOGLE_TTS_RETRIES", "2").strip() or "2"))

# Text similarity threshold for deduplicating noisy OCR frames (0.0 to 1.0)
TTS_SIMILARITY_THRESHOLD = float(os.environ.get("TTS_SIMILARITY_THRESHOLD", 0.85))

# Enable/disable TTS output
ENABLE_TTS = os.environ.get("ENABLE_TTS", "True").lower() in ("true", "1", "yes")



# Where downloaded Piper voice (.onnx/.onnx.json) files are cached.
# Defaults to a "piper_voices" folder next to this file.
PIPER_VOICE_DIR = os.environ.get("PIPER_VOICE_DIR", "").strip() or str(Path(__file__).parent / "piper_voices")

# Maps a language code (from LANGUAGE_HINTS / OCR script detection) to a
# specific Piper voice name. Only languages with an OFFICIAL Piper voice as
# of Aug 2026 are pre-filled below -- check
# https://github.com/rhasspy/piper/blob/master/VOICES.md for the current
# full list and add more here as they become available (e.g. bn, gu, kn,
# mr, ne, ur do not have official Piper voices yet at time of writing).
# Any language not listed here falls back to PIPER_DEFAULT_VOICE.
PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "hi": "hi_IN-pratham-medium",
    "ml": "ml_IN-arjun-medium",
}

# Used for any language in LANGUAGE_HINTS that isn't in PIPER_VOICES above,
# so TTS never silently fails for an unmapped language -- it just speaks
# in the default voice instead.
PIPER_DEFAULT_VOICE = os.environ.get("PIPER_DEFAULT_VOICE", "en_US-lessac-medium")

# Language to use when a detected/requested language has no mapped Piper
# voice yet (kept for reference; PIPER_DEFAULT_VOICE above is what's
# actually used as the fallback voice).
TTS_FALLBACK_LANGUAGE = os.environ.get("TTS_FALLBACK_LANGUAGE", "en")

# Don't re-speak the same OCR text more than once within this window (avoids
# nagging the user every time a stable frame re-triggers OCR on a label
# that hasn't changed).
TTS_MIN_REPEAT_INTERVAL_SECONDS = float(os.environ.get("TTS_MIN_REPEAT_INTERVAL_SECONDS", 4.0))

# Piper is fast enough that a fairly generous cap is fine -- this mainly
# exists to avoid a pathologically long single OCR block (e.g. a dense
# document) blocking the queue for too long.
TTS_MAX_TEXT_CHARS = int(os.environ.get("TTS_MAX_TEXT_CHARS", 400))

# Piper's raw output can sound quiet on some systems/speakers. This
# multiplies the generated waveform's amplitude before playback (clipped
# to avoid distortion). 1.0 = unchanged, 2.0 = roughly twice as loud.
TTS_VOLUME_GAIN = float(os.environ.get("TTS_VOLUME_GAIN", 2.5))