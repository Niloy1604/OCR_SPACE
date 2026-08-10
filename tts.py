r"""
Text-to-Speech via Piper (https://github.com/OHF-Voice/piper1-gpl).

Speaks OCR-detected text out loud in the language/script it was written in.
Runs entirely on a background worker thread with a request queue, so calling
`speak()` from the webcam loop never blocks frame capture or rendering.

--------------------------------------------------------------------------
WHY THIS REPLACES THE PREVIOUS (Indic Parler-TTS) VERSION
--------------------------------------------------------------------------
The previous version used ai4bharat/indic-parler-tts, a ~0.9B parameter
AUTOREGRESSIVE model: it generates audio one token at a time, which on CPU
took ~80+ seconds to speak a single short sentence -- unusable for a
real-time wearable device.

Piper is NON-AUTOREGRESSIVE (VITS architecture, exported to ONNX, ~15-60M
parameters per voice): it generates the entire waveform in one parallel
pass. It runs comfortably in real time on CPU alone -- including on
Raspberry-Pi-class hardware, which matters if this is meant to eventually
run on constrained wearable/embedded hardware rather than a full PC.

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
    pip install piper-tts

Voices are downloaded automatically on first use and cached under
config.PIPER_VOICE_DIR. You can also pre-download explicitly:
    python -m piper.download_voices en_US-lessac-medium
    python -m piper.download_voices hi_IN-pratham-medium

See config.py's PIPER_VOICES dict to map each language code you use in
LANGUAGE_HINTS to a specific Piper voice name. Not every language in
LANGUAGE_HINTS necessarily has an official Piper voice yet -- check
https://github.com/rhasspy/piper/blob/master/VOICES.md for the current
list, and add/adjust entries in PIPER_VOICES as more become available.
Any language without a configured voice falls back to PIPER_DEFAULT_VOICE
(English by default) rather than failing.
--------------------------------------------------------------------------
"""

import io
import os
import wave
import time
import queue
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PiperTTS")


# ------------------------------------------------------------------ #
# Script-based language detection
# ------------------------------------------------------------------ #
# OCR.space already narrows recognition to config.LANGUAGE_HINTS. This just
# figures out *which* of those configured languages a given piece of
# recognized text is actually written in (by Unicode script), so we pick
# a matching Piper voice instead of always defaulting to English.

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


import re

def clean_text_for_speech(text: str) -> str:
    """
    Strip Markdown-style artifacts OCR.space sometimes adds (heading `#`
    markers, `*`/`_` emphasis, backticks) before handing text to TTS --
    otherwise Piper tries to literally pronounce them (e.g. reads "#" as
    "hash" mid-sentence).
    """
    cleaned = re.sub(r"#+\s*", " ", text)
    cleaned = re.sub(r"[*_`]+", "", cleaned)
    return " ".join(cleaned.split())


def segment_by_script(text: str, allowed_hints: Optional[List[str]] = None) -> List[tuple]:
    """
    Split `text` into contiguous runs grouped by script (e.g. Devanagari
    vs Latin), so each run can be spoken with the correct language's voice
    instead of forcing one voice to read a mixed-language sentence.
    Digits/punctuation/whitespace attach to whichever run they're inside
    and never trigger a split on their own.

    Returns a list of (language_code, text_segment) tuples in order.
    """
    allowed = set(allowed_hints or getattr(config, "LANGUAGE_HINTS", None) or ["en"])
    segments: List[tuple] = []
    current_lang: Optional[str] = None
    current_chars: List[str] = []

    for ch in text:
        cp = ord(ch)
        detected = None
        for code, ranges in _SCRIPT_RANGES.items():
            if code in allowed and any(lo <= cp <= hi for lo, hi in ranges):
                detected = code
                break

        if detected is None:
            # Punctuation/digit/space/unrecognized script -- stays part of
            # whatever run is currently open rather than forcing a split.
            current_chars.append(ch)
            continue

        if current_lang is None:
            current_lang = detected
            current_chars.append(ch)
        elif detected == current_lang:
            current_chars.append(ch)
        else:
            segment_text = "".join(current_chars).strip()
            if segment_text:
                segments.append((current_lang, segment_text))
            current_lang = detected
            current_chars = [ch]

    if current_chars:
        segment_text = "".join(current_chars).strip()
        if segment_text:
            segments.append((current_lang or "en", segment_text))

    return segments if segments else [("en", text.strip())]


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


class PiperTTSEngine:
    """
    Wraps Piper TTS behind a background worker thread with a request queue.
    Callers fire-and-forget via `speak(text)`. Piper voices are small
    (tens of MB) and non-autoregressive, so unlike the old Parler-TTS
    engine, loading and synthesis are both fast enough to happen inline
    without a long "please wait" warm-up.
    """

    def __init__(
        self,
        voice_dir: Optional[str] = None,
        enabled: Optional[bool] = None,
        voices: Optional[Dict[str, str]] = None,
        default_voice: Optional[str] = None,
    ):
        self.enabled = getattr(config, "ENABLE_TTS", True) if enabled is None else enabled
        self.voice_dir = voice_dir or getattr(config, "PIPER_VOICE_DIR", "./piper_voices")
        self.voices_map = voices or getattr(config, "PIPER_VOICES", {"en": "en_US-lessac-medium"})
        self.default_voice = default_voice or getattr(config, "PIPER_DEFAULT_VOICE", "en_US-lessac-medium")

        os.makedirs(self.voice_dir, exist_ok=True)

        self._loaded_voices: Dict[str, "object"] = {}  # voice_name -> PiperVoice instance
        self._load_lock = threading.Lock()
        self._piper_module = None
        self._load_failed = False

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
            target=self._worker_loop, name="PiperTTS-Worker", daemon=True
        )
        self._worker_thread.start()

    def _get_voice(self, voice_name: str):
        """Load (and cache) a Piper voice by name, downloading it first if
        it isn't already present in self.voice_dir. Piper voices are small
        (tens of MB), so this is fast even on first use -- nothing like the
        multi-minute wait the old autoregressive model needed."""
        if voice_name in self._loaded_voices:
            return self._loaded_voices[voice_name]

        with self._load_lock:
            if voice_name in self._loaded_voices:
                return self._loaded_voices[voice_name]

            if self._piper_module is None:
                import piper
                from piper import PiperVoice
                from piper.download_voices import download_voice
                self._piper_module = piper
                self._PiperVoice = PiperVoice
                self._download_voice = download_voice

            onnx_path = os.path.join(self.voice_dir, f"{voice_name}.onnx")
            if not os.path.exists(onnx_path):
                logger.info(f"Downloading Piper voice '{voice_name}' (~15-60MB, one-time)...")
                self._download_voice(voice_name, Path(self.voice_dir))
                logger.info(f"Voice '{voice_name}' downloaded.")

            voice = self._PiperVoice.load(onnx_path)
            self._loaded_voices[voice_name] = voice
            logger.info(f"Piper voice '{voice_name}' ready.")
            return voice

    def _resolve_voice_name(self, language: str) -> str:
        return self.voices_map.get(language, self.default_voice)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def busy(self) -> bool:
        """True if the worker is currently synthesizing/playing speech, or
        has requests still waiting."""
        return self._busy or not self._queue.empty()

    def speak(self, text: str, language: Optional[str] = None, dedupe: bool = True):
        """
        Queue `text` for background synthesis + playback. Non-blocking --
        safe to call every time new OCR text arrives. Near-duplicate text
        spoken within TTS_MIN_REPEAT_INTERVAL_SECONDS is dropped so the
        device doesn't re-read a static label on every stable frame.
        """
        if not self.enabled or self._load_failed or not text or not text.strip():
            return

        clean = " ".join(text.strip().split())
        clean = clean_text_for_speech(clean)

        max_chars = getattr(config, "TTS_MAX_TEXT_CHARS", 400)
        if max_chars and len(clean) > max_chars:
            truncated = clean[:max_chars].rsplit(" ", 1)[0].strip()
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
        """Block until the request queue is empty AND the worker isn't
        mid-synthesis. Returns False if `timeout` elapses first."""
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
                if request.language:
                    segments = [(request.language, request.text)]
                else:
                    segments = segment_by_script(request.text)

                preview = request.text[:60] + ("..." if len(request.text) > 60 else "")
                voice_summary = " -> ".join(self._resolve_voice_name(lang) for lang, _ in segments)
                logger.info(f"Synthesizing ({voice_summary}, {len(request.text)} chars): '{preview}'")
                t0 = time.perf_counter()

                audio, sample_rate = self._synthesize_segments(segments)

                logger.info(
                    f"Synthesis done in {time.perf_counter() - t0:.2f}s "
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

    def _synthesize_segments(self, segments: List[tuple]):
        """Synthesize each (language, text) segment with its own voice and
        stitch the results into one waveform at a common sample rate, with
        a brief pause between segments spoken in different languages."""
        target_sr = None
        clips: List[np.ndarray] = []
        gap_seconds = 0.15

        for lang, seg_text in segments:
            if not seg_text.strip():
                continue
            voice_name = self._resolve_voice_name(lang)
            voice = self._get_voice(voice_name)
            audio, sr = self._synthesize_one(voice, seg_text)

            if target_sr is None:
                target_sr = sr
            elif sr != target_sr:
                audio = self._resample(audio, sr, target_sr)

            if clips:
                clips.append(np.zeros(int(target_sr * gap_seconds), dtype=np.float32))
            clips.append(audio)

        if not clips:
            return np.zeros(1, dtype=np.float32), target_sr or 22050

        return np.concatenate(clips), target_sr

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr or len(audio) == 0:
            return audio
        duration = len(audio) / orig_sr
        target_len = max(1, int(duration * target_sr))
        orig_idx = np.arange(len(audio))
        target_idx = np.linspace(0, len(audio) - 1, num=target_len)
        return np.interp(target_idx, orig_idx, audio).astype(np.float32)

    @staticmethod
    def _synthesize_one(voice, text: str):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            raw = wav_file.readframes(n_frames)
            sample_width = wav_file.getsampwidth()

        if sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            # Fallback for any voice/config that isn't 16-bit PCM.
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0

        return audio, sample_rate

    @staticmethod
    def _play(audio: np.ndarray, sample_rate: int):
        import sounddevice as sd

        gain = getattr(config, "TTS_VOLUME_GAIN", 1.0)
        if gain != 1.0:
            audio = np.clip(audio * gain, -1.0, 1.0)

        sd.play(audio, samplerate=sample_rate)
        sd.wait()


def save_wav(audio: np.ndarray, sample_rate: int, path: str):
    """Utility for debugging: dump a generated waveform to a .wav file."""
    import soundfile as sf

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, sample_rate)


if __name__ == "__main__":
    # Quick standalone smoke test:
    #   python tts.py "Hello, this is a real time test."
    #   python tts.py "आपकी दवाई की पर्ची तैयार है"
    import sys

    sample_text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test of Piper text to speech."
    engine = PiperTTSEngine()
    engine.speak(sample_text)
    logger.info("Queued. Waiting for synthesis + playback to finish...")
    finished = engine.wait_until_idle(timeout=60)
    if not finished:
        logger.error("Timed out after 60s -- check the logs above.")
    engine.stop()