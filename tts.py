r"""
Multi-Level Text-to-Speech Architecture and Text Stabilization.

Architecture:
    TTSManager
        ├── GeminiTTSEngine (Primary: Gemini TTS with Algieba voice)
        ├── CloudTTSChirp3Engine (Secondary: Google Cloud TTS with Chirp 3 HD voice)
        └── PiperTTSEngine (Offline Fallback: Local Piper ONNX synthesis)

Includes TextStabilizer for fuzzy text normalization & deduplication to prevent
re-reading repeated or slightly noisy OCR output.

Runs on a non-blocking background worker thread so calls to `speak()` never block
camera feed or OCR loops.
"""

import base64
import difflib
import io
import os
import re
import wave
import time
import queue
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TTSManager")


# ------------------------------------------------------------------ #
# Text Stabilization & Similarity Deduplication
# ------------------------------------------------------------------ #

class TextStabilizer:
    """
    Cleans OCR artifacts and calculates text similarity to prevent re-speaking
    identical or slightly noisy consecutive OCR detections (e.g. "Welcome to Kolkata"
    vs "Welcome to Kolkatta").
    """

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """Strip Markdown-style artifacts (heading `#`, `*`/`_` emphasis, backticks)."""
        cleaned = re.sub(r"#+\s*", " ", text)
        cleaned = re.sub(r"[*_`]+", "", cleaned)
        return " ".join(cleaned.split())

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize case and whitespace for similarity comparison."""
        cleaned = TextStabilizer.clean_text_for_speech(text)
        return cleaned.strip().lower()

    @staticmethod
    def is_similar_text(text1: str, text2: str, threshold: float = 0.85) -> bool:
        """
        Returns True if text1 and text2 are fuzzy matches above the similarity threshold.
        """
        norm1 = TextStabilizer.normalize_text(text1)
        norm2 = TextStabilizer.normalize_text(text2)

        if not norm1 and not norm2:
            return True
        if not norm1 or not norm2:
            return False
        if norm1 == norm2:
            return True

        ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        return ratio >= threshold


def decode_audio_bytes(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decodes audio bytes (WAV RIFF format or raw 16-bit PCM) into a float32 numpy array
    and sample rate.
    """
    if not raw_bytes:
        return np.zeros(1, dtype=np.float32), 24000

    # Check for RIFF header (standard WAV)
    if raw_bytes[:4] in (b"RIFF", b"RIFX"):
        try:
            buf = io.BytesIO(raw_bytes)
            with wave.open(buf, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(n_frames)

            if sample_width == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 255.0

            return audio, sample_rate
        except Exception as e:
            logger.debug(f"Failed WAV header decode, falling back to raw PCM: {e}")

    # Raw 16-bit signed PCM mono, 24kHz default
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, 24000


# ------------------------------------------------------------------ #
# Common Audio Playback Layer
# ------------------------------------------------------------------ #

class AudioPlayback:
    """Handles audio playback using sounddevice with stdlib winsound fallback."""

    @staticmethod
    def play(audio: np.ndarray, sample_rate: int):
        gain = getattr(config, "TTS_VOLUME_GAIN", 1.0)
        if gain != 1.0:
            audio = np.clip(audio * gain, -1.0, 1.0)

        try:
            import sounddevice as sd
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except ImportError:
            try:
                import winsound
                import tempfile
                pcm_data = (audio * 32767).astype(np.int16).tobytes()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_name = f.name
                    with wave.open(tmp_name, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(pcm_data)
                winsound.PlaySound(tmp_name, winsound.SND_FILENAME)
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Audio playback unavailable: {e}")
        except Exception as e:
            logger.warning(f"Sounddevice playback error: {e}")


# ------------------------------------------------------------------ #
# Engine 1: Google Gemini TTS (Primary - Algieba Voice)
# ------------------------------------------------------------------ #

class GeminiTTSEngine:
    """Primary TTS engine using Gemini TTS API with the Algieba voice."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(config, "GOOGLE_API_KEY", "")
        self.voice = getattr(config, "GOOGLE_TTS_VOICE", "Algieba")
        self.model = getattr(config, "GOOGLE_TTS_MODEL", "gemini-2.5-flash-preview-tts") or "gemini-2.5-flash-preview-tts"
        self.request_timeout = float(getattr(config, "GOOGLE_TTS_TIMEOUT_SECONDS", 30.0) or 30.0)
        self.max_retries = max(1, int(getattr(config, "GOOGLE_TTS_RETRIES", 2) or 2))

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        if not self.api_key or self.api_key == "your_google_api_key_here":
            raise ValueError("Gemini API key is not configured.")

        # Try google-genai SDK first if available
        try:
            try:
                import google.genai as genai
                from google.genai import types  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                genai = None
                types = None
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.voice
                            )
                        )
                    ),
                ),
            )
            raw_b64 = response.candidates[0].content.parts[0].inline_data.data
            raw_bytes = base64.b64decode(raw_b64) if isinstance(raw_b64, str) else raw_b64
            return decode_audio_bytes(raw_bytes)
        except Exception as sdk_err:
            logger.debug(f"google-genai SDK synthesis unavailable/failed ({sdk_err}). Trying REST API...")

        # Fallback to direct REST API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self.voice
                        }
                    }
                }
            }
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                res = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.request_timeout,
                )
                res.raise_for_status()
                data = res.json()

                part = data["candidates"][0]["content"]["parts"][0]
                inline_data = part.get("inlineData") or part.get("inline_data")
                raw_b64 = inline_data["data"]
                raw_bytes = base64.b64decode(raw_b64)
                return decode_audio_bytes(raw_bytes)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Gemini TTS REST request failed on attempt {attempt}/{self.max_retries}: {e}. Retrying..."
                    )
                    time.sleep(min(2 * attempt, 5))
                    continue
                raise last_error

        raise last_error


# ------------------------------------------------------------------ #
# Engine 2: Google Cloud TTS (Secondary Backup - Chirp 3 HD Voice)
# ------------------------------------------------------------------ #

class CloudTTSChirp3Engine:
    """Secondary TTS engine using Google Cloud Text-to-Speech with Chirp 3 HD voice."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(config, "GOOGLE_API_KEY", "")
        self.voice = getattr(config, "CHIRP3_BACKUP_VOICE", "en-US-Chirp3-HD-Charon")

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        # Try google-cloud-texttospeech SDK first if available
        try:
            from google.cloud import texttospeech

            client = texttospeech.TextToSpeechClient(
                client_options={"api_key": self.api_key} if self.api_key else None
            )
            input_text = texttospeech.SynthesisInput(text=text)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=self.voice,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )
            response = client.synthesize_speech(
                input=input_text, voice=voice_params, audio_config=audio_config
            )
            return decode_audio_bytes(response.audio_content)
        except Exception as sdk_err:
            logger.debug(f"Cloud TTS SDK synthesis unavailable/failed ({sdk_err}). Trying REST API...")

        if not self.api_key or self.api_key == "your_google_api_key_here":
            raise ValueError("Google Cloud API key is not configured.")

        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_key}"
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": self.voice,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16"
            }
        }
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        res.raise_for_status()
        data = res.json()

        raw_b64 = data["audioContent"]
        raw_bytes = base64.b64decode(raw_b64)
        return decode_audio_bytes(raw_bytes)


# ------------------------------------------------------------------ #
# Engine 3: Piper TTS (Offline Local Fallback)
# ------------------------------------------------------------------ #

_SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)],
    "bn": [(0x0980, 0x09FF)],
    "gu": [(0x0A80, 0x0AFF)],
    "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)],
    "ta": [(0x0B80, 0x0BFF)],
    "te": [(0x0C00, 0x0C7F)],
    "or": [(0x0B00, 0x0B7F)],
    "pa": [(0x0A00, 0x0A7F)],
    "ur": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "en": [(0x0041, 0x005A), (0x0061, 0x007A)],
}


class PiperTTSEngine:
    """Offline local TTS fallback engine using Piper ONNX models."""

    def __init__(self):
        self.voice_dir = getattr(config, "PIPER_VOICE_DIR", "./piper_voices")
        self.voices_map = getattr(config, "PIPER_VOICES", {"en": "en_US-lessac-medium"})
        self.default_voice = getattr(config, "PIPER_DEFAULT_VOICE", "en_US-lessac-medium")
        self._loaded_voices: Dict[str, object] = {}
        self._load_lock = threading.Lock()

    def _get_voice(self, voice_name: str):
        if voice_name in self._loaded_voices:
            return self._loaded_voices[voice_name]

        with self._load_lock:
            if voice_name in self._loaded_voices:
                return self._loaded_voices[voice_name]

            import piper
            from piper import PiperVoice
            from piper.download_voices import download_voice

            onnx_path = os.path.join(self.voice_dir, f"{voice_name}.onnx")
            if not os.path.exists(onnx_path):
                os.makedirs(self.voice_dir, exist_ok=True)
                download_voice(voice_name, Path(self.voice_dir))

            voice = PiperVoice.load(onnx_path)
            self._loaded_voices[voice_name] = voice
            return voice

    def synthesize(self, text: str, language: Optional[str] = None) -> Tuple[np.ndarray, int]:
        import piper
        voice_name = self.voices_map.get(language or "en", self.default_voice)
        voice = self._get_voice(voice_name)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        buf.seek(0)
        return decode_audio_bytes(buf.read())


# ------------------------------------------------------------------ #
# High-Level TTS Manager
# ------------------------------------------------------------------ #

@dataclass
class TTSRequest:
    text: str
    language: Optional[str] = None


class TTSManager:
    """
    Central Manager for the Multi-Level TTS Architecture.
    Orchestrates fallback: Gemini TTS (Algieba) -> Cloud TTS (Chirp 3) -> Piper (Offline).
    """

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = getattr(config, "ENABLE_TTS", True) if enabled is None else enabled

        # Initialize engines
        self.primary_engine = GeminiTTSEngine()
        self.secondary_engine = CloudTTSChirp3Engine()
        self.offline_engine = PiperTTSEngine()
        self.stabilizer = TextStabilizer()

        self._queue: "queue.Queue[Optional[TTSRequest]]" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._busy = False

        self._last_spoken_text = ""
        self._last_spoken_time = 0.0

        if self.enabled:
            logger.info("TTS Manager initialized.")
            logger.info(f"  Primary:  Gemini TTS ({config.GOOGLE_TTS_MODEL}, Voice: {config.GOOGLE_TTS_VOICE})")
            logger.info(f"  Backup:   Google Cloud TTS (Voice: {config.CHIRP3_BACKUP_VOICE})")
            logger.info(f"  Offline:  Piper TTS ({config.PIPER_DEFAULT_VOICE})")
            self._start_worker()
        else:
            logger.warning("TTS disabled via config.ENABLE_TTS=False.")

    def _start_worker(self):
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="TTSManager-Worker", daemon=True
        )
        self._worker_thread.start()

    @property
    def busy(self) -> bool:
        return self._busy or not self._queue.empty()

    def speak(self, text: str, language: Optional[str] = None, dedupe: bool = True):
        """
        Queue OCR text for synthesis and playback.
        Filters duplicates & near-similar text using TextStabilizer.
        """
        if not self.enabled or not text or not text.strip():
            return

        clean = TextStabilizer.clean_text_for_speech(text.strip())

        max_chars = getattr(config, "TTS_MAX_TEXT_CHARS", 400)
        if max_chars and len(clean) > max_chars:
            clean = clean[:max_chars].rsplit(" ", 1)[0].strip()

        min_interval = getattr(config, "TTS_MIN_REPEAT_INTERVAL_SECONDS", 4.0)
        similarity_threshold = getattr(config, "TTS_SIMILARITY_THRESHOLD", 0.85)

        if dedupe:
            time_since_last = time.time() - self._last_spoken_time
            if (
                time_since_last < min_interval
                and TextStabilizer.is_similar_text(clean, self._last_spoken_text, threshold=similarity_threshold)
            ):
                logger.debug(f"Skipping duplicate/similar OCR speech request: '{clean[:40]}...'")
                return

        self._last_spoken_text = clean
        self._last_spoken_time = time.time()
        self._queue.put(TTSRequest(text=clean, language=language))

    def stop(self):
        """Signals the worker thread to stop synthesis and clears remaining queue."""
        self._stop_event.set()
        with self._queue.mutex:
            self._queue.queue.clear()
        self._queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

    def wait_until_idle(self, poll_interval: float = 0.2, timeout: Optional[float] = None) -> bool:
        start = time.time()
        while True:
            if self._queue.empty() and not self._busy:
                return True
            if timeout is not None and (time.time() - start) > timeout:
                return False
            time.sleep(poll_interval)

    # ------------------------------------------------------------------ #
    # Worker Loop & Fallback Orchestration
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
                text = request.text
                preview = text[:60] + ("..." if len(text) > 60 else "")
                logger.info(f"Synthesizing OCR text ({len(text)} chars): '{preview}'")
                t0 = time.perf_counter()

                audio, sr, engine_used = self._synthesize_with_fallback(text, request.language)

                logger.info(
                    f"Speech synthesis completed via [{engine_used}] in {time.perf_counter() - t0:.2f}s "
                    f"({len(audio) / sr:.1f}s audio). Playing..."
                )
                AudioPlayback.play(audio, sr)
                logger.info("Playback finished.")
            except Exception:
                logger.exception("TTS synthesis/playback error:")
            finally:
                self._busy = False

    def _synthesize_with_fallback(self, text: str, language: Optional[str]) -> Tuple[np.ndarray, int, str]:
        # 1. Primary: Gemini TTS (Algieba voice)
        try:
            audio, sr = self.primary_engine.synthesize(text)
            if audio is not None and len(audio) > 0:
                return audio, sr, f"Gemini TTS ({self.primary_engine.voice})"
        except Exception as e:
            logger.warning(f"Primary Gemini TTS failed: {e}. Trying secondary Cloud TTS (Chirp 3)...")

        # 2. Secondary Backup: Google Cloud TTS (Chirp 3 HD voice)
        try:
            audio, sr = self.secondary_engine.synthesize(text)
            if audio is not None and len(audio) > 0:
                return audio, sr, f"Cloud TTS Backup ({self.secondary_engine.voice})"
        except Exception as e:
            logger.warning(f"Secondary Cloud TTS (Chirp 3) failed: {e}. Trying Piper offline fallback...")

        # 3. Offline Fallback: Piper TTS
        try:
            audio, sr = self.offline_engine.synthesize(text, language)
            return audio, sr, "Piper Offline TTS"
        except Exception as e:
            logger.error(f"Offline Piper TTS failed: {e}")

        return np.zeros(1, dtype=np.float32), 24000, "Silent Fallback"


# Backwards compatibility alias
TTSEngine = TTSManager


if __name__ == "__main__":
    import sys

    sample_text = sys.argv[1] if len(sys.argv) > 1 else "Testing Multi-Level TTS Manager with Gemini Algieba voice."
    manager = TTSManager()
    logger.info(f"Queuing sample text: '{sample_text}'...")
    manager.speak(sample_text)
    finished = manager.wait_until_idle(timeout=30)
    if not finished:
        logger.error("TTS test timed out after 30s.")
    manager.stop()