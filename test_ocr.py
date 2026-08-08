import sys
import numpy as np
import cv2

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


if __name__ == "__main__":
    print("=" * 60)
    print("  Vision OCR Pipeline Diagnostic Tests")
    print("=" * 60)
    test_overlay_drawing()
    test_frame_encoding()
    test_engine_graceful_auth_error()
    print("=" * 60)
    print("[SUCCESS] All local diagnostic tests passed!")
    print("=" * 60)
