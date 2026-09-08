"""
Real-Time Webcam Text Detection with OCR.space API & Offline Fallback Support.

Features:
- Live OpenCV camera feed window
- Supports both laptop built-in webcams and any external USB webcams
- DirectShow (cv2.CAP_DSHOW) support on Windows for robust external camera compatibility
- Automatic camera discovery and runtime camera switching (Press 'w' or number keys)
- Graceful auto-fallback if a camera is disconnected or fails to open
- Automatically detects if an OCR.space API key is configured
- Seamless offline local fallback (OpenCV text-region detector) if no key is provided
- Motion stability detection & configurable time throttling
- Non-blocking background worker thread (maintains 60 FPS video preview)
- Console deduplication & real-time latency benchmark
"""

import sys
import os
import time
import logging
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2

import config
from vision_ocr import VisionOCREngine, VisionOCRResult, draw_ocr_overlay, draw_unicode_text
from tts import TTSManager


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WebcamOCR")


def get_backend_candidates(backend_pref: str = "AUTO") -> List[int]:
    """Return an ordered list of OpenCV VideoCapture backend flags to attempt."""
    pref = backend_pref.upper()
    if pref == "DSHOW":
        return [cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif pref == "MSMF":
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif pref == "AUTO":
        if os.name == "nt":
            # On Windows, DirectShow is vastly more reliable for external USB webcams
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        return [cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera_capture(
    index: int,
    backend_pref: str = "AUTO",
    width: int = 1280,
    height: int = 720,
) -> Optional[cv2.VideoCapture]:
    """
    Attempt to open camera with preferred backend, falling back if necessary.
    Configures width and height with error protection.
    """
    candidates = get_backend_candidates(backend_pref)

    for backend in candidates:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Attempt to set desired resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                # Verify actual frame read to ensure camera isn't a ghost/busy device
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    return cap

                cap.release()
        except Exception as e:
            logger.debug(f"Error trying backend {backend} on camera {index}: {e}")
            continue

    return None


def scan_available_cameras(max_indices: int = 5, backend_pref: str = "AUTO") -> List[Dict]:
    """
    Probe camera indices (0 to max_indices - 1) to identify all connected and working cameras.
    Silences OpenCV C++ log output during probing.
    """
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

    available = []
    for idx in range(max_indices):
        cap = open_camera_capture(idx, backend_pref)
        if cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            label = "Laptop / Built-in Webcam" if idx == 0 else f"External Webcam {idx}"
            available.append({
                "index": idx,
                "label": label,
                "width": w,
                "height": h,
            })
            cap.release()

    return available


def compute_frame_motion(prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> float:
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return 999.0

    prev_small = cv2.resize(prev_gray, (160, 120))
    curr_small = cv2.resize(curr_gray, (160, 120))

    diff = cv2.absdiff(prev_small, curr_small)
    return float(np.mean(diff))


def draw_status_header(
    frame: np.ndarray,
    status_text: str,
    fps: float,
    latency_ms: float,
    engine_name: str,
    last_text: str,
    camera_index: int,
    camera_label: str,
    error_msg: Optional[str] = None,
    tts_muted: bool = False,
    camera_notice: Optional[str] = None,
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    banner_height = 70
    cv2.rectangle(annotated, (0, 0), (w, banner_height), config.HEADER_BG_COLOR_BGR, cv2.FILLED)
    cv2.line(annotated, (0, banner_height), (w, banner_height), (100, 100, 100), 1)

    # Line 1: Status, Engine, Camera, FPS, Latency, TTS
    tts_str = "Muted" if tts_muted else "On"
    cam_short = f"Cam[{camera_index}]: {camera_label.split('/')[0].strip()}"
    info_str = (
        f"Status: {status_text} | {cam_short} | "
        f"FPS: {fps:.1f} | Lat: {latency_ms:.0f}ms | TTS: {tts_str}"
    )
    cv2.putText(
        annotated,
        info_str,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255) if "Processing" in status_text else (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    # Line 2: Camera Notice, Error, or Detected Text
    if camera_notice:
        display_str = f">> {camera_notice}"
        color = (0, 255, 255)  # Bright yellow
    elif error_msg:
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
    help_str = "Press 'q': Quit | 's': Force OCR | 'w': Switch Camera | 'c': Clear | 'm': TTS"
    cv2.putText(
        annotated,
        help_str,
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return annotated


def draw_ocr_progress_window(
    status_text: str,
    engine_name: str,
    latency_ms: float,
    last_text: str,
    camera_index: int,
    camera_label: str,
    error_msg: Optional[str] = None,
) -> np.ndarray:
    progress = np.full((360, 640, 3), (25, 25, 25), dtype=np.uint8)

    cv2.putText(progress, "OCR Progress", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(progress, f"Status: {status_text}", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(progress, f"Active Camera: [{camera_index}] {camera_label}", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)
    cv2.putText(progress, f"Engine: {engine_name}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(progress, f"Latency: {latency_ms:.1f} ms", (20, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(progress, "Detected Text:", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    message = error_msg or last_text or "No text detected yet."
    overlay_color = (0, 165, 255) if error_msg else (255, 255, 255)
    progress = draw_unicode_text(progress, message[:180], (20, 218), overlay_color, font_size=18)
    cv2.putText(progress, "Keys: 'w' (Switch Camera) | 's' (Force OCR) | 'q' (Quit)", (20, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return progress


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Webcam OCR with Multi-Camera & External Webcam Support"
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=None,
        help="Camera device index (e.g. 0 for laptop webcam, 1 for external USB webcam)",
    )
    parser.add_argument(
        "-l", "--list-cameras",
        action="store_true",
        help="Scan and list all detected camera devices, then exit",
    )
    parser.add_argument(
        "-e", "--prefer-external",
        action="store_true",
        default=getattr(config, "PREFER_EXTERNAL_WEBCAM", False),
        help="Automatically prioritize external webcam if one is connected",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=getattr(config, "CAMERA_BACKEND", "AUTO"),
        choices=["AUTO", "DSHOW", "MSMF"],
        help="OpenCV capture backend to use (default: AUTO, uses DirectShow on Windows)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture frame width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture frame height (default: 720)",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=getattr(config, "MAX_CAMERA_SCAN_INDEX", 5),
        help="Maximum camera indices to scan (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If user requests listing cameras, scan and print
    if args.list_cameras:
        print("\n[*] Scanning for connected webcams (laptop and external)...")
        detected = scan_available_cameras(max_indices=args.max_scan, backend_pref=args.backend)
        if not detected:
            print("[-] No active webcams detected. Please verify your camera is plugged in.")
            sys.exit(1)

        print(f"\n[+] Found {len(detected)} camera device(s):")
        print("-" * 55)
        for cam in detected:
            print(f"  Camera Index [{cam['index']}]: {cam['label']} (Native: {cam['width']}x{cam['height']})")
        print("-" * 55)
        print("To launch with a specific camera:")
        print("  python main_webcam.py --camera 0  (Laptop Webcam)")
        if len(detected) > 1:
            print(f"  python main_webcam.py --camera {detected[1]['index']}  (External Webcam)")
        print()
        sys.exit(0)

    engine = VisionOCREngine(use_document_text=config.USE_DOCUMENT_TEXT_DETECTION)

    if engine.mode == "OCRSPACE_ENGINE3":
        logger.info("OCR backend: OCR.space (Engine 3, cloud).")
    else:
        logger.warning(
            "OCR backend: local offline fallback (OpenCV text-region detector). "
            "Set OCRSPACE_API_KEY in your .env to use OCR.space."
        )

    tts = TTSManager()
    tts_muted = not config.ENABLE_TTS

    # Discover cameras
    logger.info("Scanning for connected webcams...")
    available_cameras = scan_available_cameras(max_indices=args.max_scan, backend_pref=args.backend)
    if available_cameras:
        logger.info(f"Detected {len(available_cameras)} camera(s):")
        for cam in available_cameras:
            logger.info(f"  [{cam['index']}] {cam['label']} ({cam['width']}x{cam['height']})")
    else:
        logger.warning("Auto-scan found no cameras. Attempting direct opening...")

    # Determine desired initial camera index
    target_cam_idx = config.CAMERA_INDEX
    if args.camera is not None:
        target_cam_idx = args.camera
    elif args.prefer_external:
        # Pick the first external camera (index > 0) if available
        external_cams = [c for c in available_cameras if c["index"] > 0]
        if external_cams:
            target_cam_idx = external_cams[0]["index"]
            logger.info(f"--prefer-external active: Selected Camera [{target_cam_idx}] ({external_cams[0]['label']}).")

    # Helper to get friendly camera label
    def get_camera_label(idx: int) -> str:
        for c in available_cameras:
            if c["index"] == idx:
                return c["label"]
        return "Laptop Webcam" if idx == 0 else f"External Webcam {idx}"

    current_cam_idx = target_cam_idx
    logger.info(f"Opening camera source [index={current_cam_idx}] ({get_camera_label(current_cam_idx)})...")
    cap = open_camera_capture(current_cam_idx, args.backend, args.width, args.height)

    # Fallback to any available camera if selected camera failed
    if cap is None and available_cameras:
        for fallback_cam in available_cameras:
            if fallback_cam["index"] != current_cam_idx:
                logger.warning(
                    f"Camera [{current_cam_idx}] failed to open. "
                    f"Falling back to Camera [{fallback_cam['index']}] ({fallback_cam['label']})..."
                )
                current_cam_idx = fallback_cam["index"]
                cap = open_camera_capture(current_cam_idx, args.backend, args.width, args.height)
                if cap is not None:
                    break

    if cap is None or not cap.isOpened():
        logger.error("Unable to access camera.")
        logger.error(
            f"Neither camera index '{target_cam_idx}' nor any fallback cameras could be opened.\n"
            "Tip: Run 'python main_webcam.py --list-cameras' to view all connected camera devices."
        )
        sys.exit(1)

    cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(config.OCR_WINDOW_TITLE, cv2.WINDOW_NORMAL)

    executor = ThreadPoolExecutor(max_workers=1)
    ocr_future = None

    prev_gray: Optional[np.ndarray] = None
    last_detection_time = 0.0
    current_ocr_result: Optional[VisionOCRResult] = None
    last_real_text = ""
    status_text = "Idle"
    engine_name = engine.mode
    is_processing = False
    force_trigger = False
    was_forced_snap = False

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    camera_notice_msg: Optional[str] = None
    camera_notice_expiry = 0.0
    consecutive_read_failures = 0

    logger.info(
        f"Webcam OCR running with Camera [{current_cam_idx}] ({get_camera_label(current_cam_idx)}). "
        "Press 'w' to switch webcams at any time."
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 15:
                    logger.warning(f"Camera [{current_cam_idx}] stopped sending frames.")
                    # Try to switch to another camera if available
                    if len(available_cameras) > 1:
                        next_cams = [c for c in available_cameras if c["index"] != current_cam_idx]
                        if next_cams:
                            new_idx = next_cams[0]["index"]
                            logger.info(f"Auto-recovering feed: switching to Camera [{new_idx}]...")
                            cap.release()
                            new_cap = open_camera_capture(new_idx, args.backend, args.width, args.height)
                            if new_cap is not None:
                                cap = new_cap
                                current_cam_idx = new_idx
                                consecutive_read_failures = 0
                                camera_notice_msg = f"Reconnected to Camera [{new_idx}] {get_camera_label(new_idx)}"
                                camera_notice_expiry = time.time() + 3.0
                                prev_gray = None
                                continue
                time.sleep(0.05)
                continue

            consecutive_read_failures = 0
            current_time = time.time()

            # Clear camera switch banner after expiry
            if camera_notice_msg and current_time > camera_notice_expiry:
                camera_notice_msg = None

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

                        normalized_text = engine._normalize_text(result.full_text)
                        if normalized_text:
                            last_real_text = normalized_text

                        logger.info(
                            f"OCR result via [{result.engine_name}] "
                            f"({result.latency_ms:.0f}ms): "
                            f"'{normalized_text[:80]}{'...' if len(normalized_text) > 80 else ''}'"
                            if normalized_text else
                            f"OCR result via [{result.engine_name}] ({result.latency_ms:.0f}ms): (no text detected)"
                        )

                        if (
                            not tts_muted
                            and normalized_text
                        ):
                            if tts.busy:
                                logger.info(
                                    "TTS still speaking a previous result -- "
                                    "skipping this one to avoid a backlog."
                                )
                            else:
                                tts.speak(normalized_text, dedupe=not was_forced_snap)
                                was_forced_snap = False
                    else:
                        status_text = "Error"
                        logger.error(f"OCR Error via [{engine_name}]: {result.error}")

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

            last_text_str = last_real_text if last_real_text else (current_ocr_result.full_text if current_ocr_result else "")
            latency_ms = current_ocr_result.latency_ms if current_ocr_result else 0.0
            error_msg = current_ocr_result.error if (current_ocr_result and current_ocr_result.error) else None

            final_frame = draw_status_header(
                annotated_frame,
                status_text=status_text,
                fps=current_fps,
                latency_ms=latency_ms,
                engine_name=engine_name,
                last_text=last_text_str,
                camera_index=current_cam_idx,
                camera_label=get_camera_label(current_cam_idx),
                error_msg=error_msg,
                tts_muted=tts_muted,
                camera_notice=camera_notice_msg,
            )
            progress_frame = draw_ocr_progress_window(
                status_text=status_text,
                engine_name=engine_name,
                latency_ms=latency_ms,
                last_text=last_text_str,
                camera_index=current_cam_idx,
                camera_label=get_camera_label(current_cam_idx),
                error_msg=error_msg,
            )

            cv2.imshow(config.WINDOW_TITLE, final_frame)
            cv2.imshow(config.OCR_WINDOW_TITLE, progress_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                force_trigger = True
                was_forced_snap = True
            elif key == ord("c"):
                current_ocr_result = None
                last_real_text = ""
                tts._last_spoken_text = ""
                status_text = "Cleared"
            elif key == ord("m"):
                tts_muted = not tts_muted
                logger.info(f"TTS {'muted' if tts_muted else 'unmuted'}.")

            # Camera Switching: 'w' key or TAB key (9) cycles to next camera
            elif key == ord("w") or key == 9:
                # If only 1 camera was initially found, re-scan in case external was plugged in
                if len(available_cameras) <= 1:
                    logger.info("Scanning for newly attached webcams...")
                    refreshed = scan_available_cameras(max_indices=args.max_scan, backend_pref=args.backend)
                    if refreshed:
                        available_cameras = refreshed

                if len(available_cameras) > 1:
                    indices = [c["index"] for c in available_cameras]
                    try:
                        curr_pos = indices.index(current_cam_idx)
                        next_idx = indices[(curr_pos + 1) % len(indices)]
                    except ValueError:
                        next_idx = indices[0]

                    logger.info(f"Switching camera from [{current_cam_idx}] to [{next_idx}]...")
                    new_cap = open_camera_capture(next_idx, args.backend, args.width, args.height)
                    if new_cap is not None:
                        cap.release()
                        cap = new_cap
                        current_cam_idx = next_idx
                        prev_gray = None
                        current_ocr_result = None
                        camera_notice_msg = f"Switched to [{current_cam_idx}] {get_camera_label(current_cam_idx)}"
                        camera_notice_expiry = time.time() + 3.0
                        logger.info(f"Active camera is now: [{current_cam_idx}] {get_camera_label(current_cam_idx)}")
                    else:
                        logger.error(f"Failed to open Camera [{next_idx}]. Staying on [{current_cam_idx}].")
                else:
                    logger.info("Only 1 camera detected. Plug in an external webcam and press 'w' again.")
                    camera_notice_msg = "Only 1 camera detected. Plug in USB webcam & press 'w'"
                    camera_notice_expiry = time.time() + 3.0

            # Direct Camera Selection: keys '0' through '9'
            elif ord("0") <= key <= ord("9"):
                target_num = key - ord("0")
                if target_num != current_cam_idx:
                    logger.info(f"Direct switch request to Camera [{target_num}]...")
                    new_cap = open_camera_capture(target_num, args.backend, args.width, args.height)
                    if new_cap is not None:
                        cap.release()
                        cap = new_cap
                        current_cam_idx = target_num
                        prev_gray = None
                        current_ocr_result = None
                        # Update available list if needed
                        if not any(c["index"] == target_num for c in available_cameras):
                            available_cameras.append({
                                "index": target_num,
                                "label": f"External Webcam {target_num}",
                                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            })
                        camera_notice_msg = f"Switched to [{current_cam_idx}] {get_camera_label(current_cam_idx)}"
                        camera_notice_expiry = time.time() + 3.0
                        logger.info(f"Switched to Camera [{current_cam_idx}] {get_camera_label(current_cam_idx)}")
                    else:
                        logger.warning(f"Camera [{target_num}] is not available.")
                        camera_notice_msg = f"Camera [{target_num}] is not available"
                        camera_notice_expiry = time.time() + 2.5

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(wait=False)
        tts.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam application closed.")


if __name__ == "__main__":
    main()