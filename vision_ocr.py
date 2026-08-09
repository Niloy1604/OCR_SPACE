import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw, ImageFont

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VisionOCR")

DEFAULT_OCRSPACE_ENDPOINT = "https://api.ocr.space/parse/image"


@dataclass
class VisionOCRItem:
    """Individual recognized text block / word with bounding polygon."""
    text: str
    confidence: float
    bounding_poly: List[Tuple[int, int]]


@dataclass
class VisionOCRResult:
    """Complete result container for an OCR request."""
    success: bool
    full_text: str = ""
    items: List[VisionOCRItem] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None
    engine_name: str = "OCR.space (Engine 3)"
    frame_width: int = 0
    frame_height: int = 0
    timestamp: float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        return not self.full_text or not self.full_text.strip()


class VisionOCREngine:
    """
    OCR engine backed by the OCR.space cloud API using OCREngine=3 (their most
    accurate recognition engine). If no API key is configured, or a request
    fails/times out, this falls back to a purely local OpenCV text-region
    detector so the app can still report approximate text locations offline.
    """

    # OCR.space Engine 3 (primary) with automatic Engine 2 fallback.
    OCR_ENGINE = "3"
    FALLBACK_OCR_ENGINE = "2"

    _LANGUAGE_MAP = {
        "en": "eng", "eng": "eng",
        "hi": "hin", "hin": "hin",
        "bn": "ben", "ben": "ben",
        "gu": "guj", "guj": "guj",
        "kn": "kan", "kan": "kan",
        "ml": "mal", "mal": "mal",
        "mr": "mar", "mar": "mar",
        "ne": "nep", "nep": "nep",
        "ur": "urd", "urd": "urd",
        "ar": "ara", "ara": "ara",
        "zh": "chs", "zh-cn": "chs", "zh-tw": "cht",
        "ja": "jpn", "jpn": "jpn",
        "ko": "kor", "kor": "kor",
        "fr": "fre", "fre": "fre",
        "de": "ger", "ger": "ger",
        "es": "spa", "spa": "spa",
        "it": "ita", "ita": "ita",
        "pt": "por", "por": "por",
        "ru": "rus", "rus": "rus",
        "tr": "tur", "tur": "tur",
        "nl": "dut", "dut": "dut",
        "pl": "pol", "pol": "pol",
        "vi": "vnm", "vnm": "vnm",
        "uk": "ukr", "ukr": "ukr",
    }

    
    AUTO_LANGUAGE = "auto"

    def __init__(self, use_document_text: bool = False):
        # Retained for backward-compatible call sites; OCR.space engine 3
        # already performs full-document style parsing, so this flag does not
        # change request behaviour.
        self.use_document_text = use_document_text

        self.api_key = self._resolve_api_key()
        self.endpoint = (
            getattr(config, "OCRSPACE_ENDPOINT", None)
            or os.environ.get("OCRSPACE_ENDPOINT", "").strip()
            or DEFAULT_OCRSPACE_ENDPOINT
        )

        if self.api_key:
            self.mode = "OCRSPACE_ENGINE3"
            logger.info("OCR.space engine initialized (OCREngine=3).")
        else:
            self.mode = "LOCAL_OPENCV_CONTOURS"
            logger.warning(
                "No OCR.space API key found (config.OCRSPACE_API_KEY / "
                "OCRSPACE_API_KEY env var). Falling back to local OpenCV "
                "text-region detector."
            )

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        key = getattr(config, "OCRSPACE_API_KEY", "") or os.environ.get("OCRSPACE_API_KEY", "")
        key = (key or "").strip()
        return key or None

    @classmethod
    def _map_language(cls, hint: str) -> str:
        return cls._LANGUAGE_MAP.get((hint or "").strip().lower(), "eng")

    @staticmethod
    def _normalize_text(text: Optional[str]) -> str:
        if not text:
            return ""

        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return ""

        placeholders = {
            "*[no text detected]*",
            "[no text detected]",
            "no text detected",
            "no text detected yet",
        }
        if cleaned.lower() in placeholders:
            return ""

        if cleaned.lower().startswith("*[no text detected") or cleaned.lower().startswith("[no text detected"):
            return ""

        return cleaned

    @classmethod
    def _resolve_request_language(cls, language_hints: Optional[List[str]]) -> str:
        """
        Decide which single `language` value to send to OCR.space.

        - No hints, "auto", or more than one hint -> "auto", so Engine 3 can
          auto-detect across its full language list (this is what lets a
          single deployment handle English plus any of the Indian scripts
          OCR.space supports without per-request guessing).
        - Exactly one recognized hint -> map it to OCR.space's specific code
          for a slightly faster/more targeted single-language pass.
        """
        hints = [h.strip().lower() for h in (language_hints or []) if h and h.strip()]

        if not hints or len(hints) > 1 or "auto" in hints:
            return cls.AUTO_LANGUAGE

        return cls._map_language(hints[0])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect_text_from_bytes(
        self,
        image_bytes: bytes,
        language_hints: Optional[List[str]] = None,
        image_width: int = 0,
        image_height: int = 0,
    ) -> VisionOCRResult:
        if not image_bytes:
            return VisionOCRResult(success=False, error="Empty image bytes.")

        effective_language_hints = language_hints or config.LANGUAGE_HINTS or ["en"]

        if self.api_key:
            result = self._detect_ocrspace(
                image_bytes, effective_language_hints, image_width, image_height
            )
            if result.success:
                return result
            logger.warning(
                f"OCR.space request failed ({result.error}). "
                "Falling back to local offline text-region detector."
            )

        frame = self._bytes_to_frame(image_bytes)
        return self._detect_local_frame(frame)

    def detect_text_from_cv2_frame(
        self,
        frame: np.ndarray,
        language_hints: Optional[List[str]] = None,
        jpeg_quality: int = 85,
    ) -> VisionOCRResult:
        if frame is None or frame.size == 0:
            return VisionOCRResult(success=False, error="Invalid camera frame.")

        effective_language_hints = language_hints or config.LANGUAGE_HINTS or ["en"]
        height, width = frame.shape[:2]

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        success, encoded_img = cv2.imencode(".jpg", frame, encode_param)
        if not success:
            return VisionOCRResult(success=False, error="JPEG encoding failed.")

        return self.detect_text_from_bytes(
            image_bytes=encoded_img.tobytes(),
            language_hints=effective_language_hints,
            image_width=width,
            image_height=height,
        )

    # ------------------------------------------------------------------ #
    # OCR.space
    # ------------------------------------------------------------------ #

    def _detect_ocrspace(
        self,
        image_bytes: bytes,
        language_hints: Optional[List[str]],
        image_width: int,
        image_height: int,
    ) -> VisionOCRResult:
        start_time = time.perf_counter()
        lang = self._resolve_request_language(language_hints)

        engines_to_try = [self.OCR_ENGINE, self.FALLBACK_OCR_ENGINE]
        last_error: Optional[str] = None
        last_engine_name = "OCR.space (Engine 3)"

        for engine_number in engines_to_try:
            payload = {
                "apikey": self.api_key,
                "language": lang,
                "OCREngine": engine_number,
                "isOverlayRequired": "true",
                "detectOrientation": "true",
            }
            files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}

            try:
                response = requests.post(self.endpoint, data=payload, files=files, timeout=25)
                response.raise_for_status()
                result_json = response.json()
            except Exception as e:
                last_error = f"OCR.space request error: {e}"
                if engine_number == self.OCR_ENGINE:
                    logger.warning(f"OCR.space Engine 3 failed ({last_error}); retrying with Engine 2.")
                    last_engine_name = "OCR.space (Engine 2)"
                    continue
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return VisionOCRResult(
                    success=False,
                    error=last_error,
                    latency_ms=latency_ms,
                    engine_name=last_engine_name,
                )

            if result_json.get("IsErroredOnProcessing"):
                error_msg = result_json.get("ErrorMessage") or result_json.get("ErrorDetails") or "Unknown OCR.space error"
                if isinstance(error_msg, list):
                    error_msg = "; ".join(str(m) for m in error_msg)
                last_error = str(error_msg)
                if engine_number == self.OCR_ENGINE:
                    logger.warning(f"OCR.space Engine 3 returned an error ({last_error}); retrying with Engine 2.")
                    last_engine_name = "OCR.space (Engine 2)"
                    continue
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return VisionOCRResult(
                    success=False,
                    error=last_error,
                    latency_ms=latency_ms,
                    engine_name=last_engine_name,
                )

            parsed_results = result_json.get("ParsedResults") or []
            if not parsed_results:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return VisionOCRResult(
                    success=True,
                    full_text="",
                    items=[],
                    latency_ms=latency_ms,
                    engine_name="OCR.space (Engine 3)" if engine_number == self.OCR_ENGINE else "OCR.space (Engine 2)",
                    frame_width=image_width,
                    frame_height=image_height,
                )

            full_text_parts: List[str] = []
            items: List[VisionOCRItem] = []

            for parsed in parsed_results:
                if parsed.get("FileParseExitCode", 1) not in (1,):
                    # Non-success parse for this particular result; skip its text
                    # but keep looking at any other parsed results.
                    continue

                text = (parsed.get("ParsedText") or "").strip()
                if text:
                    full_text_parts.append(text)

                overlay = parsed.get("TextOverlay") or {}
                for line in overlay.get("Lines", []) or []:
                    words = line.get("Words", []) or []
                    if not words:
                        continue

                    # Build ONE bounding box that spans the whole line (rather
                    # than a separate box per word) so we draw a single readable
                    # label per line instead of one black label stacked above
                    # every individual word -- the latter is what produced solid
                    # black bars over dense text.
                    lefts, tops, rights, bottoms = [], [], [], []
                    for word in words:
                        left = int(word.get("Left", 0))
                        top = int(word.get("Top", 0))
                        width = int(word.get("Width", 0))
                        height = int(word.get("Height", 0))
                        lefts.append(left)
                        tops.append(top)
                        rights.append(left + width)
                        bottoms.append(top + height)

                    x_min, y_min = min(lefts), min(tops)
                    x_max, y_max = max(rights), max(bottoms)

                    line_text = (line.get("LineText") or "").strip()
                    if not line_text:
                        line_text = " ".join(w.get("WordText", "") for w in words).strip()
                    if not line_text:
                        continue

                    poly = [
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max),
                    ]
                    items.append(VisionOCRItem(text=line_text, confidence=1.0, bounding_poly=poly))

            full_text = "\n".join(full_text_parts)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return VisionOCRResult(
                success=True,
                full_text=full_text,
                items=items,
                latency_ms=latency_ms,
                engine_name="OCR.space (Engine 3)" if engine_number == self.OCR_ENGINE else "OCR.space (Engine 2)",
                frame_width=image_width,
                frame_height=image_height,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return VisionOCRResult(
            success=False,
            error=last_error or "OCR.space request failed.",
            latency_ms=latency_ms,
            engine_name=last_engine_name,
        )

    # ------------------------------------------------------------------ #
    # Offline fallback (used only when OCR.space is unconfigured/unreachable)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bytes_to_frame(image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _detect_local_frame(self, frame: Optional[np.ndarray]) -> VisionOCRResult:
        start_time = time.perf_counter()
        if frame is None or frame.size == 0:
            return VisionOCRResult(success=False, error="Failed to decode image / invalid frame.")

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > 25 and bh > 12 and bw < w * 0.9 and bh < h * 0.9:
                poly = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
                items.append(VisionOCRItem(text="[Text Region]", confidence=0.7, bounding_poly=poly))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return VisionOCRResult(
            success=True,
            full_text="Running Offline Text Region Detection (OCR.space unavailable)",
            items=items,
            latency_ms=latency_ms,
            engine_name="OpenCV Text Detector (Offline Fallback)",
            frame_width=w,
            frame_height=h,
        )


def draw_unicode_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_size: int = 20,
) -> np.ndarray:
    if not text:
        return frame

    try:
        font_paths = [
            "C:/Windows/Fonts/Nirmala.ttc",
            "C:/Windows/Fonts/Nirmala.ttf",
            "C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",
            "C:/Windows/Fonts/NotoSansDevanagariUI-Regular.ttf",
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break

        if font is None:
            raise OSError("No suitable Unicode font found")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_image)
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return frame


def draw_ocr_overlay(
    frame: np.ndarray,
    ocr_result: VisionOCRResult,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    show_boxes: bool = True,
    show_labels: bool = True,
    label_opacity: float = 0.0,
) -> np.ndarray:
    if ocr_result is None or not ocr_result.success or not ocr_result.items:
        return frame

    annotated = frame.copy()
    frame_h, frame_w = annotated.shape[:2]

    for item in ocr_result.items:
        poly = np.array(item.bounding_poly, dtype=np.int32)
        if len(poly) == 0:
            continue

        poly_reshaped = poly.reshape((-1, 1, 2))

        if show_boxes:
            cv2.polylines(annotated, [poly_reshaped], isClosed=True, color=box_color, thickness=2)

        if show_labels and item.text:
            x_min = max(0, min(pt[0] for pt in item.bounding_poly))
            y_min = max(0, min(pt[1] for pt in item.bounding_poly))
            y_max = min(frame_h, max(pt[1] for pt in item.bounding_poly))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(item.text, font, font_scale, thickness)
            label_w = min(text_w + 8, frame_w - x_min)

            # Prefer placing the label just above the box; if there's no
            # room (box is near the top edge), place it just below instead
            # so it never gets clipped off-screen or squashed to zero height.
            if y_min - text_h - 8 >= 0:
                tag_y1 = y_min - text_h - 8
                tag_y2 = y_min
            else:
                tag_y1 = y_max
                tag_y2 = min(frame_h, y_max + text_h + 8)

            tag_x1 = x_min
            tag_x2 = min(frame_w, x_min + label_w)

            # Keep the text label itself lightweight and transparent so it no
            # longer blocks nearby words or creates dark rectangular artifacts.
            if label_opacity > 0.0:
                label_region = annotated[tag_y1:tag_y2, tag_x1:tag_x2]
                if label_region.size > 0:
                    overlay_box = np.full_like(label_region, bg_color, dtype=np.uint8)
                    blended = cv2.addWeighted(overlay_box, label_opacity, label_region, 1 - label_opacity, 0)
                    annotated[tag_y1:tag_y2, tag_x1:tag_x2] = blended

            annotated = draw_unicode_text(
                annotated,
                item.text,
                (tag_x1 + 4, max(text_h, tag_y2 - 6)),
                text_color,
                font_size=max(16, int(font_scale * 24)),
            )

    return annotated