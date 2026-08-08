"""
Real-Time Webcam Text Detection with OCR.space API & Offline Fallback Support.

Features:
- Live OpenCV camera feed window
- Automatically detects if an OCR.space API key is configured
- Seamless offline local fallback (OpenCV text-region detector) if no key is provided
- Motion stability detection & configurable time throttling
- Non-blocking background worker thread (maintains 60 FPS video preview)
- Console deduplication & real-time latency benchmark
"""

import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
import cv2

import config
from vision_ocr import VisionOCREngine, VisionOCRResult, draw_ocr_overlay, draw_unicode_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WebcamOCR")


def compute_frame_motion(prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> float:
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return 999.0

    prev_small = cv2.resize(prev_gray, (160, 120))
    curr_small = cv2.resize(curr_gray, (160, 120))

    diff = cv2.absdiff(prev_small, curr_small)
    return float(np.mean(diff))


def is_text_substantially_different(new_text: str, old_text: str) -> bool:
    new_norm = " ".join(new_text.strip().split())
    old_norm = " ".join(old_text.strip().split())
    return new_norm != old_norm


def draw_status_header(
    frame: np.ndarray,
    status_text: str,
    fps: float,
    latency_ms: float,
    engine_name: str,
    last_text: str,
    error_msg: Optional[str] = None,
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    banner_height = 70
    cv2.rectangle(annotated, (0, 0), (w, banner_height), config.HEADER_BG_COLOR_BGR, cv2.FILLED)
    cv2.line(annotated, (0, banner_height), (w, banner_height), (100, 100, 100), 1)

    # Line 1: Status, Engine, FPS, Latency
    info_str = f"Status: {status_text} | Engine: {engine_name} | FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms"
    cv2.putText(
        annotated,
        info_str,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255) if "Processing" in status_text else (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    # Line 2: Detected Text or Error
    if error_msg:
        display_str = f"Notice: {error_msg[:80]}"
        color = (0, 165, 255)
    elif last_text:
        display_str = f"OCR Text: '{last_text[:75]}{'...' if len(last_text) > 75 else ''}'"
        color = (255, 255, 255)
    else:
        display_str = "OCR Text: (No text detected)"
        color = (180, 180, 180)

    annotated = draw_unicode_text(
        annotated,
        display_str,
        (12, 42),
        color,
        font_size=20,
    )

    # Bottom helper bar
    help_str = "Press 'q': Quit | 's': Force OCR | 'c': Clear"
    cv2.putText(
        annotated,
        help_str,
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return annotated


def draw_ocr_progress_window(
    status_text: str,
    engine_name: str,
    latency_ms: float,
    last_text: str,
    error_msg: Optional[str] = None,
) -> np.ndarray:
    progress = np.full((360, 640, 3), (25, 25, 25), dtype=np.uint8)

    cv2.putText(progress, "OCR Progress", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(progress, f"Status: {status_text}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(progress, f"Engine: {engine_name}", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(progress, f"Latency: {latency_ms:.1f} ms", (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(progress, "Detected Text:", (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    message = error_msg or last_text or "No text detected yet."
    overlay_color = (0, 165, 255) if error_msg else (255, 255, 255)
    progress = draw_unicode_text(progress, message[:180], (20, 208), overlay_color, font_size=18)
    cv2.putText(progress, "Press 'q' to stop the webcam", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return progress


def main():
    engine = VisionOCREngine(use_document_text=config.USE_DOCUMENT_TEXT_DETECTION)

    if engine.mode == "OCRSPACE_ENGINE3":
        logger.info("OCR backend: OCR.space (Engine 3, cloud).")
    else:
        logger.warning(
            "OCR backend: local offline fallback (OpenCV text-region detector). "
            "Set OCRSPACE_API_KEY in your .env to use OCR.space."
        )

    logger.info(f"Opening camera source (index={config.CAMERA_INDEX})...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        logger.error("Unable to access camera.")
        logger.error(f"Camera index '{config.CAMERA_INDEX}' could not be opened.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(config.OCR_WINDOW_TITLE, cv2.WINDOW_NORMAL)

    executor = ThreadPoolExecutor(max_workers=1)
    ocr_future = None

    prev_gray: Optional[np.ndarray] = None
    last_detection_time = 0.0
    current_ocr_result: Optional[VisionOCRResult] = None
    status_text = "Idle"
    engine_name = engine.mode
    is_processing = False
    force_trigger = False

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            current_time = time.time()

            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = compute_frame_motion(prev_gray, curr_gray)
            prev_gray = curr_gray.copy()

            if ocr_future is not None and ocr_future.done():
                try:
                    result: VisionOCRResult = ocr_future.result()
                    current_ocr_result = result
                    is_processing = False

                    engine_name = result.engine_name

                    if result.success:
                        status_text = f"Ready ({result.latency_ms:.0f}ms)"
                    else:
                        status_text = "Error"
                        logger.error(f"OCR Error: {result.error}")

                except Exception as e:
                    logger.error(f"OCR thread exception: {e}")
                    is_processing = False
                    status_text = "Worker Exception"

                ocr_future = None

            time_elapsed = current_time - last_detection_time
            is_time_due = time_elapsed >= config.DETECTION_INTERVAL_SECONDS
            is_stable = motion_score < config.MOTION_THRESHOLD

            should_trigger = False

            if force_trigger:
                should_trigger = True
                force_trigger = False
            elif not is_processing and is_time_due:
                if config.MOTION_STABILITY_ENABLED:
                    if is_stable or time_elapsed >= config.DETECTION_INTERVAL_SECONDS * 2.0:
                        should_trigger = True
                else:
                    should_trigger = True

            if should_trigger and not is_processing:
                is_processing = True
                status_text = "Processing..."
                last_detection_time = current_time

                frame_to_process = frame.copy()
                ocr_future = executor.submit(
                    engine.detect_text_from_cv2_frame,
                    frame_to_process,
                    config.LANGUAGE_HINTS,
                    config.JPEG_QUALITY,
                )

            annotated_frame = frame.copy()

            if config.SHOW_BOUNDING_BOXES and current_ocr_result and current_ocr_result.success:
                annotated_frame = draw_ocr_overlay(
                    annotated_frame,
                    current_ocr_result,
                    box_color=config.BOX_COLOR_BGR,
                    text_color=config.TEXT_COLOR_BGR,
                    bg_color=config.TEXT_BG_COLOR_BGR,
                    show_boxes=config.SHOW_BOUNDING_BOXES,
                    show_labels=config.SHOW_TEXT_LABELS,
                )

            last_text_str = current_ocr_result.full_text if current_ocr_result else ""
            latency_ms = current_ocr_result.latency_ms if current_ocr_result else 0.0
            error_msg = current_ocr_result.error if (current_ocr_result and current_ocr_result.error) else None

            final_frame = draw_status_header(
                annotated_frame,
                status_text=status_text,
                fps=current_fps,
                latency_ms=latency_ms,
                engine_name=engine_name,
                last_text=last_text_str,
                error_msg=error_msg,
            )
            progress_frame = draw_ocr_progress_window(
                status_text=status_text,
                engine_name=engine_name,
                latency_ms=latency_ms,
                last_text=last_text_str,
                error_msg=error_msg,
            )

            cv2.imshow(config.WINDOW_TITLE, final_frame)
            cv2.imshow(config.OCR_WINDOW_TITLE, progress_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                force_trigger = True
            elif key == ord("c"):
                current_ocr_result = None
                status_text = "Cleared"

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(wait=False)
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam application closed.")


if __name__ == "__main__":
    main()