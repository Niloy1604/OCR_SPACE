r"""
Text-to-Speech via AI4Bharat's Indic Parler-TTS.

Speaks OCR-detected text out loud in the language/script it was written in.
Runs entirely on a background worker thread with a request queue, so calling
`speak()` from the webcam loop never blocks frame capture or rendering --
matching the pattern main_webcam.py already uses for OCR (ThreadPoolExecutor
submit + poll for `.done()`).

Model: https://huggingface.co/ai4bharat/indic-parler-tts
Supports 21 Indic languages + English via natural-language voice-description
prompts (no explicit language= argument -- language is inferred from the
description text and the words being spoken).

--------------------------------------------------------------------------
FIXED IN THIS VERSION
--------------------------------------------------------------------------
1. ROOT CAUSE BUG: `_ensure_loaded()` used to try the robotic Windows
   pyttsx3/SAPI fallback FIRST, and returned immediately if it succeeded --
   silently skipping the real Indic Parler-TTS model with no way to opt
   out. That's now reversed: the real model is always attempted first,
   and the robotic fallback is only used if the model genuinely fails to
   load AND `config.TTS_ALLOW_ROBOTIC_FALLBACK` is explicitly set to True.

2. `pyttsx3` reuse bug: the old code reused a single `pyttsx3.init()`
   engine instance across every `speak()` call. This is a well-known
   pyttsx3 issue on Windows where the engine goes silent after the first
   utterance because the underlying SAPI proxy doesn't reset cleanly.
   The fallback path (if ever enabled) now creates a fresh engine per call.

3. Added clear, periodic "still downloading/loading, do not close the
   app" progress logging during the first-run model load, since a silent
   multi-minute wait was what caused the app to be closed mid-download
   last time -- which corrupts the local Hugging Face cache and causes
   subsequent "model not found" style errors. See also the cache-clearing
   note below if you've already hit that.
--------------------------------------------------------------------------
If you previously interrupted a download, clear the corrupted cache once
before re-running:

    Windows (PowerShell):
        Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--ai4bharat--indic-parler-tts"

    Linux/Mac:
        rm -rf ~/.cache/huggingface/hub/models--ai4bharat--indic-parler-tts
--------------------------------------------------------------------------
"""

import os
import time
import queue
import logging
import threading
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IndicParlerTTS")


# ------------------------------------------------------------------ #
# Script-based language detection
# ------------------------------------------------------------------ #
# OCR.space already narrows recognition to config.LANGUAGE_HINTS. This just
# figures out *which* of those configured languages a given piece of
# recognized text is actually written in (by Unicode script), so the TTS
# description prompt matches the text instead of always defaulting to
# English or requiring a separate language-ID model/API call.

_SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)],                      # Devanagari (Hindi/Marathi/Nepali/Sanskrit)
    "bn": [(0x0980, 0x09FF)],                       # Bengali/Assamese
    "gu": [(0x0A80, 0x0AFF)],                       # Gujarati
    "kn": [(0x0C80, 0x0CFF)],                       # Kannada
    "ml": [(0x0D00, 0x0D7F)],                       # Malayalam
    "ta": [(0x0B80, 0x0BFF)],                       # Tamil
    "te": [(0x0C00, 0x0C7F)],                       # Telugu
    "or": [(0x0B00, 0x0B7F)],                       # Odia
    "pa": [(0x0A00, 0x0A7F)],                       # Gurmukhi (Punjabi)
    "ur": [(0x0600, 0x06FF), (0x0750, 0x077F)],      # Arabic script (Urdu)
    "en": [(0x0041, 0x005A), (0x0061, 0x007A)],      # Latin
}

# language code -> (name Indic Parler-TTS expects in the description prompt,
# a reference speaker name known to work well for that language). Speaker
# names come from the model card's list of recommended per-language voices;
# any of these can be swapped freely.
_LANGUAGE_PROMPTS = {
    "en": ("English", "Thoma"),
    "hi": ("Hindi", "Rohit"),
    "bn": ("Bengali", "Arjun"),
    "gu": ("Gujarati", "Yash"),
    "kn": ("Kannada", "Suresh"),
    "ml": ("Malayalam", "Anjali"),
    "mr": ("Marathi", "Sanjay"),
    "ne": ("Nepali", "Amrita"),
    "ur": ("Urdu", "Ahmed"),
    "ta": ("Tamil", "Jaya"),
    "te": ("Telugu", "Prakash"),
    "or": ("Odia", "Manas"),
    "pa": ("Punjabi", "Divjot"),
}

DEFAULT_VOICE_DESCRIPTION = (
    "{speaker}'s voice speaks {language} clearly at a natural, moderate pace, "
    "with minimal background noise, suitable for a wearable accessibility assistant."
)


def detect_script_language(text: str, allowed_hints: Optional[List[str]] = None) -> str:
    """
    Guess which configured language `text` is written in, by counting
    characters per Unicode script block. Falls back to English if nothing
    recognizable is found or the majority script isn't in the allowed set.
    """
    if not text or not text.strip():
        return "en"

    allowed = set(allowed_hints or getattr(config, "LANGUAGE_HINTS", None) or ["en"])
    counts = {code: 0 for code in _SCRIPT_RANGES}

    for ch in text:
        cp = ord(ch)
        for code, ranges in _SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[code] += 1
                break

    counts = {code: n for code, n in counts.items() if code in allowed and n > 0}
    if not counts:
        return "en" if "en" in allowed else next(iter(allowed), "en")

    return max(counts, key=counts.get)


@dataclass
class TTSRequest:
    text: str
    language: Optional[str] = None


class IndicParlerTTSEngine:
    """
    Wraps ai4bharat/indic-parler-tts behind a background worker thread with
    a request queue. Callers fire-and-forget via `speak(text)`; the model is
    lazy-loaded on the first request (it's a ~0.9B model, so this can take
    a while) rather than blocking application startup.

    By default this ALWAYS uses the real Indic Parler-TTS model. The
    robotic pyttsx3/SAPI voice is never used automatically -- set
    config.TTS_ALLOW_ROBOTIC_FALLBACK = True if you explicitly want a
    fast/low-quality fallback voice while the real model loads or if the
    real model fails to load for some reason.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        enabled: Optional[bool] = None,
        sample_rate: Optional[int] = None,
    ):
        self.enabled = getattr(config, "ENABLE_TTS", True) if enabled is None else enabled
        self.model_name = model_name or getattr(config, "TTS_MODEL_NAME", "ai4bharat/indic-parler-tts")
        self._requested_device = device or getattr(config, "TTS_DEVICE", "auto")
        self._sample_rate_override = sample_rate

        # Opt-in only. Defaults to False so the real model is never
        # silently bypassed.
        self._allow_robotic_fallback = getattr(config, "TTS_ALLOW_ROBOTIC_FALLBACK", False)

        self._model = None
        self._tokenizer = None
        self._description_tokenizer = None
        self._torch = None
        self._device = None
        self._loaded = False
        self._load_failed = False
        self._load_lock = threading.Lock()
        self._tts_backend = None

        self._queue: "queue.Queue[Optional[TTSRequest]]" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._busy = False

        self._last_spoken_text = ""
        self._last_spoken_time = 0.0

        if self.enabled:
            self._start_worker()
        else:
            logger.warning("TTS disabled via config.ENABLE_TTS=False.")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _start_worker(self):
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="IndicParlerTTS-Worker", daemon=True
        )
        self._worker_thread.start()

    def _ensure_loaded(self):
        if self._loaded or self._load_failed:
            return

        with self._load_lock:
            if self._loaded or self._load_failed:
                return

            logger.info(
                f"Loading Indic Parler-TTS model '{self.model_name}' (first call). "
                f"This is a ~0.9B parameter model -- on first run it must download "
                f"~2GB+ of weights, which can take several minutes depending on your "
                f"connection. DO NOT close the app during this step: closing it "
                f"mid-download can corrupt the local Hugging Face cache and cause "
                f"'model not found' errors on the next run."
            )

            t0 = time.perf_counter()
            heartbeat_stop = threading.Event()
            self._start_load_heartbeat(t0, heartbeat_stop)

            try:
                import torch
                from transformers import AutoTokenizer
                import importlib

                parler_tts_module = importlib.import_module("parler_tts")
                ParlerTTSForConditionalGeneration = getattr(
                    parler_tts_module, "ParlerTTSForConditionalGeneration"
                )

                self._torch = torch

                if self._requested_device == "auto":
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self._device = self._requested_device

                dtype = torch.float16 if self._device == "cuda" else torch.float32

                self._model = ParlerTTSForConditionalGeneration.from_pretrained(
                    self.model_name, torch_dtype=dtype
                ).to(self._device)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._description_tokenizer = AutoTokenizer.from_pretrained(
                    self._model.config.text_encoder._name_or_path
                )

                self._tts_backend = "parler"
                self._loaded = True
                logger.info(
                    f"Indic Parler-TTS ready on '{self._device}' "
                    f"in {time.perf_counter() - t0:.1f}s."
                )
                return

            except Exception:
                logger.exception(
                    "Failed to load Indic Parler-TTS. If this mentions a missing "
                    "file, corrupted shard, or 'not found' error, your local cache "
                    "is likely corrupted from an interrupted previous download. "
                    "Clear it and try again:\n"
                    "  Windows: Remove-Item -Recurse -Force "
                    "\"$env:USERPROFILE\\.cache\\huggingface\\hub\\models--ai4bharat--indic-parler-tts\"\n"
                    "  Linux/Mac: rm -rf ~/.cache/huggingface/hub/models--ai4bharat--indic-parler-tts"
                )
            finally:
                heartbeat_stop.set()

            # Real model failed to load. Only now consider the robotic
            # fallback, and only if the user explicitly opted in.
            if self._allow_robotic_fallback:
                try:
                    self._setup_fallback_speaker()
                    logger.warning(
                        "Falling back to robotic local speech synthesis "
                        "(config.TTS_ALLOW_ROBOTIC_FALLBACK=True)."
                    )
                    return
                except Exception:
                    logger.warning("Robotic fallback speaker is also unavailable.")

            self._load_failed = True
            logger.error("TTS will be skipped for this session -- no working backend.")

    def _start_load_heartbeat(self, start_time: float, stop_event: threading.Event):
        """Logs a 'still working' line every 20s so a slow first-run
        download never again looks like it has silently frozen."""

        def _beat():
            while not stop_event.wait(20.0):
                elapsed = time.perf_counter() - start_time
                logger.info(f"...still loading Indic Parler-TTS ({elapsed:.0f}s elapsed, please wait)...")

        threading.Thread(target=_beat, daemon=True).start()

    def _setup_fallback_speaker(self):
        if os.name != "nt":
            raise RuntimeError("Windows speech fallback is only available on Windows.")

        import pyttsx3  # raises ImportError if not installed -- that's fine, it's caught by the caller

        # Just verify it can initialize; a FRESH engine is created per
        # utterance in _speak_with_fallback to avoid the known pyttsx3
        # "goes silent after first call" reuse bug.
        probe = pyttsx3.init()
        probe.stop()
        self._tts_backend = "pyttsx3"
        self._loaded = True

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def speak(self, text: str, language: Optional[str] = None, dedupe: bool = True):
        """
        Queue `text` for background synthesis + playback. Non-blocking --
        safe to call every time new OCR text arrives. Near-duplicate text
        spoken within TTS_MIN_REPEAT_INTERVAL_SECONDS is dropped so the
        device doesn't re-read a static label on every stable frame.
        """
        if not self.enabled or not text or not text.strip():
            return

        clean = " ".join(text.strip().split())

        max_chars = getattr(config, "TTS_MAX_TEXT_CHARS", 200)
        if max_chars and len(clean) > max_chars:
            truncated = clean[:max_chars].rsplit(" ", 1)[0].strip()
            logger.info(
                f"OCR text is {len(clean)} chars -- truncating to {len(truncated)} "
                f"chars for TTS (see TTS_MAX_TEXT_CHARS in config.py) so speech "
                f"doesn't take minutes to generate on CPU."
            )
            clean = truncated

        min_interval = getattr(config, "TTS_MIN_REPEAT_INTERVAL_SECONDS", 4.0)

        if (
            dedupe
            and clean == self._last_spoken_text
            and (time.time() - self._last_spoken_time) < min_interval
        ):
            return

        self._last_spoken_text = clean
        self._last_spoken_time = time.time()
        self._queue.put(TTSRequest(text=clean, language=language))

    def stop(self):
        """Signal the worker thread to finish and wait briefly for it."""
        self._stop_event.set()
        self._queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

    def wait_until_idle(self, poll_interval: float = 0.2, timeout: Optional[float] = None) -> bool:
        """
        Block until the request queue is empty AND the worker isn't mid-synthesis.
        Useful for standalone scripts/tests where the caller needs to know
        speech has actually finished (loading the model + generating +
        playing audio) before the process exits -- unlike `stop()`, which is
        meant for a UI shutdown path and doesn't wait for slow first-time
        model loads. Returns False if `timeout` elapses first.
        """
        start = time.time()
        while True:
            if self._queue.empty() and not self._busy:
                return True
            if timeout is not None and (time.time() - start) > timeout:
                return False
            time.sleep(poll_interval)

    # ------------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------------ #

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                request = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if request is None:
                break

            self._busy = True
            try:
                self._ensure_loaded()
                if not self._load_failed:
                    preview = request.text[:60] + ("..." if len(request.text) > 60 else "")
                    logger.info(f"Synthesizing speech ({len(request.text)} chars): '{preview}'")
                    t0 = time.perf_counter()

                    if self._tts_backend == "pyttsx3":
                        self._speak_with_fallback(request.text)
                        logger.info(f"Fallback playback finished in {time.perf_counter() - t0:.1f}s.")
                    else:
                        audio, sample_rate = self._synthesize(request.text, request.language)
                        logger.info(
                            f"Synthesis done in {time.perf_counter() - t0:.1f}s "
                            f"({len(audio) / sample_rate:.1f}s of audio). Playing..."
                        )

                        self._play(audio, sample_rate)
                        logger.info("Playback finished.")
            except Exception:
                logger.exception("TTS synthesis/playback failed for this request:")
            finally:
                self._busy = False

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    def _synthesize(self, text: str, language: Optional[str]):
        lang_code = language or detect_script_language(text)
        language_name, speaker = _LANGUAGE_PROMPTS.get(lang_code, _LANGUAGE_PROMPTS["en"])

        template = getattr(config, "TTS_VOICE_DESCRIPTION_TEMPLATE", DEFAULT_VOICE_DESCRIPTION)
        description = template.format(speaker=speaker, language=language_name)

        description_ids = self._description_tokenizer(
            description, return_tensors="pt"
        ).input_ids.to(self._device)
        prompt_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)

        with self._torch.no_grad():
            generation = self._model.generate(
                input_ids=description_ids, prompt_input_ids=prompt_ids
            )

        audio = generation.to(self._torch.float32).cpu().numpy().squeeze()
        sample_rate = self._sample_rate_override or self._model.config.sampling_rate
        return audio, sample_rate

    def _speak_with_fallback(self, text: str):
        """Creates a FRESH pyttsx3 engine per call -- reusing one instance
        across calls is a known pyttsx3 bug that goes silent after the
        first utterance."""
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    @staticmethod
    def _play(audio: np.ndarray, sample_rate: int):
        import sounddevice as sd

        sd.play(audio, samplerate=sample_rate)
        sd.wait()


def save_wav(audio: np.ndarray, sample_rate: int, path: str):
    """Utility for debugging: dump a generated waveform to a .wav file."""
    import soundfile as sf

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, sample_rate)


if __name__ == "__main__":
    # Quick standalone smoke test:
    #   python tts.py "आपकी दवाई की पर्ची तैयार है"
    import sys

    sample_text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test of the Indic Parler TTS engine."
    engine = IndicParlerTTSEngine()
    engine.speak(sample_text)
    logger.info(
        "Queued. Waiting for model load + synthesis + playback to finish "
        "(first run can take several minutes while the ~0.9B model downloads)..."
    )
    finished = engine.wait_until_idle(timeout=900)  # generous timeout for first-time model download
    if not finished:
        logger.error("Timed out after 15 minutes waiting for TTS to finish -- check your connection/logs above.")
    engine.stop()