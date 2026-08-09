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
# AI4Bharat Indic Parler-TTS Settings
# ------------------------------------------------------------------ #
# Speaks OCR-detected text aloud in the language/script it was written in.
# Model card: https://huggingface.co/ai4bharat/indic-parler-tts
ENABLE_TTS = os.environ.get("ENABLE_TTS", "True").lower() in ("true", "1", "yes")
TTS_MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "ai4bharat/indic-parler-tts")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "auto")  # "auto" | "cpu" | "cuda"

# When False (default), the real Indic Parler-TTS model is ALWAYS used and
# the robotic pyttsx3/SAPI voice is never invoked automatically. Set this to
# True only if you explicitly want a fast/low-quality fallback voice to be
# used as a last resort if the real model fails to load.
TTS_ALLOW_ROBOTIC_FALLBACK = os.environ.get("TTS_ALLOW_ROBOTIC_FALLBACK", "False").lower() in ("true", "1", "yes")

# Don't re-speak the same OCR text more than once within this window (avoids
# nagging the user every time a stable frame re-triggers OCR on a label
# that hasn't changed).
TTS_MIN_REPEAT_INTERVAL_SECONDS = float(os.environ.get("TTS_MIN_REPEAT_INTERVAL_SECONDS", 4.0))

# Natural-language voice-description prompt template passed to Indic
# Parler-TTS. {speaker} and {language} are filled in per detected script.
TTS_VOICE_DESCRIPTION_TEMPLATE = os.environ.get(
    "TTS_VOICE_DESCRIPTION_TEMPLATE",
    "{speaker}'s voice speaks {language} clearly at a natural, moderate pace, "
    "with minimal background noise, suitable for a wearable accessibility assistant.",
)

# Parler-TTS is autoregressive -- generation time scales with text length, and
# on CPU (no CUDA) a long multi-line OCR block (e.g. a full ID card) can take
# well over a minute to speak. Cap how much of any single OCR result gets
# sent to TTS so responses stay usable in real time; the full text is still
# shown on-screen regardless.
TTS_MAX_TEXT_CHARS = int(os.environ.get("TTS_MAX_TEXT_CHARS", 200))