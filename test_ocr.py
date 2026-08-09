import sys
from unittest.mock import patch

import numpy as np
import cv2
import requests

import config
from vision_ocr import VisionOCREngine, VisionOCRResult, VisionOCRItem, draw_ocr_overlay


def create_synthetic_test_image() -> np.ndarray:
    """Generates a synthetic 640x480 test image with text rendered on it."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background

    # Draw test text onto image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "GOOGLE VISION OCR TEST", (50, 150), font, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, "REALTIME CAM FEED 2026", (50, 250), font, 0.9, (180, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (40, 90), (550, 300), (0, 0, 200), 2)

    return img


def test_overlay_drawing():
    print("[TEST] Testing overlay drawing functions...")
    synthetic_img = create_synthetic_test_image()

    fake_result = VisionOCRResult(
        success=True,
        full_text="GOOGLE VISION OCR TEST\nREALTIME CAM FEED 2026",
        items=[
            VisionOCRItem(
                text="GOOGLE",
                confidence=0.99,
                bounding_poly=[(50, 120), (200, 120), (200, 155), (50, 155)],
            ),
            VisionOCRItem(
                text="VISION",
                confidence=0.98,
                bounding_poly=[(210, 120), (350, 120), (350, 155), (210, 155)],
            ),
        ],
        latency_ms=123.45,
    )

    annotated = draw_ocr_overlay(synthetic_img, fake_result)
    assert annotated is not None
    assert annotated.shape == synthetic_img.shape
    print("  -> Overlay rendering test passed successfully!")


def test_overlay_does_not_draw_a_label_background():
    print("[TEST] Testing that label backgrounds stay transparent...")
    img = np.full((120, 220, 3), 120, dtype=np.uint8)
    fake_result = VisionOCRResult(
        success=True,
        full_text="HELLO",
        items=[
            VisionOCRItem(
                text="HELLO",
                confidence=1.0,
                bounding_poly=[(10, 10), (80, 10), (80, 30), (10, 30)],
            )
        ],
        latency_ms=1.0,
    )

    annotated = draw_ocr_overlay(
        img,
        fake_result,
        show_boxes=False,
        text_color=(120, 120, 120),
        bg_color=(0, 0, 0),
    )

    roi = annotated[30:60, 10:80]
    assert np.all(roi == 120)
    print("  -> Label background stays transparent.")


def test_frame_encoding():
    print("[TEST] Testing in-memory JPEG frame encoding...")
    engine = VisionOCREngine()
    synthetic_img = create_synthetic_test_image()

    # Test encoding utility
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
    success, encoded_img = cv2.imencode(".jpg", synthetic_img, encode_param)
    assert success
    jpg_bytes = encoded_img.tobytes()
    assert len(jpg_bytes) > 0
    print(f"  -> Frame successfully encoded to JPEG ({len(jpg_bytes)} bytes) in memory!")


def test_engine_graceful_auth_error():
    print("[TEST] Testing GCP authentication error handling when key is unconfigured...")
    engine = VisionOCREngine()
    synthetic_img = create_synthetic_test_image()

    res = engine.detect_text_from_cv2_frame(synthetic_img)
    # Result should return a structured result object without crashing
    assert isinstance(res, VisionOCRResult)
    print(f"  -> Engine returned structured result gracefully (Success: {res.success}).")
    if not res.success:
        print(f"  -> Expected Auth/API Error message captured: '{res.error}'")


def test_placeholder_text_is_treated_as_empty():
    print("[TEST] Testing placeholder OCR text normalization...")
    normalized = VisionOCREngine._normalize_text("*[No text detected]*")
    assert normalized == ""
    assert VisionOCREngine._normalize_text("  Hello world  ") == "Hello world"
    print("  -> Placeholder OCR text is normalized correctly.")


def test_engine_falls_back_to_engine_2_when_engine_3_fails():
    print("[TEST] Testing OCR.space Engine 3 -> Engine 2 fallback...")

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, data=None, files=None, timeout=25):
        if data.get("OCREngine") == "3":
            raise requests.exceptions.HTTPError("Engine 3 token exhausted")
        return FakeResponse(
            {
                "IsErroredOnProcessing": False,
                "ParsedResults": [
                    {
                        "FileParseExitCode": 1,
                        "ParsedText": "fallback text",
                        "TextOverlay": {"Lines": []},
                    }
                ],
            }
        )

    engine = VisionOCREngine()
    engine.api_key = "dummy-key"
    engine.endpoint = "https://example.invalid"
    engine.mode = "OCRSPACE_ENGINE3"

    with patch("vision_ocr.requests.post", side_effect=fake_post):
        res = engine._detect_ocrspace(b"fake-image", ["en"], 640, 480)

    assert res.success is True
    assert res.full_text == "fallback text"
    assert res.engine_name == "OCR.space (Engine 2)"
    print("  -> Fallback to Engine 2 worked successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print("  Vision OCR Pipeline Diagnostic Tests")
    print("=" * 60)
    test_overlay_drawing()
    test_frame_encoding()
    test_engine_graceful_auth_error()
    test_engine_falls_back_to_engine_2_when_engine_3_fails()
    print("=" * 60)
    print("[SUCCESS] All local diagnostic tests passed!")
    print("=" * 60)
