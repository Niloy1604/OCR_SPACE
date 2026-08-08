"""
Mobile Backend REST API Server (FastAPI).

Enables mobile apps (iOS/Android) or HTTP clients to upload images captured on mobile devices
and route them through the exact same Google Cloud Vision OCR engine (`vision_ocr.py`).

Run with:
    uvicorn app_server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
import cv2

try:
    from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    raise ImportError("FastAPI is required for app_server.py. Run: pip install fastapi uvicorn python-multipart pydantic")

import config
from vision_ocr import VisionOCREngine, VisionOCRResult, draw_ocr_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MobileOCRServer")

app = FastAPI(
    title="Mobile Vision OCR API Backend",
    description="Real-Time Google Cloud Vision OCR service for mobile devices and remote camera clients.",
    version="1.0.0",
)

# Enable CORS for mobile app frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
ocr_engine = VisionOCREngine(use_document_text=config.USE_DOCUMENT_TEXT_DETECTION)


# Pydantic Response Models
class BoundingVertex(BaseModel):
    x: int
    y: int


class OCRItemResponse(BaseModel):
    text: str
    confidence: float
    bounding_poly: List[List[int]]


class OCRResponse(BaseModel):
    success: bool
    full_text: str
    latency_ms: float
    items: List[OCRItemResponse]
    error: Optional[str] = None


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "service": "Google Cloud Vision Mobile OCR API",
        "status": "online",
        "gcp_client_ready": ocr_engine.client is not None,
        "docs_url": "/docs",
    }


@app.post("/api/v1/ocr", response_model=OCRResponse)
async def process_ocr_image(
    file: UploadFile = File(..., description="JPEG or PNG image binary file from mobile camera"),
    language_hints: Optional[List[str]] = Query(default=config.LANGUAGE_HINTS, description="Language hint codes"),
):
    """
    Primary endpoint for mobile devices:
    Accepts an uploaded image file (JPEG/PNG), processes it using Google Cloud Vision API,
    and returns detected text, bounding polygons, and execution latency.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG, WEBP, etc.)")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Re-use core OCR function
        result: VisionOCRResult = ocr_engine.detect_text_from_bytes(
            image_bytes=contents,
            language_hints=language_hints,
        )

        formatted_items = [
            OCRItemResponse(
                text=item.text,
                confidence=item.confidence,
                bounding_poly=[list(pt) for pt in item.bounding_poly],
            )
            for item in result.items
        ]

        return OCRResponse(
            success=result.success,
            full_text=result.full_text,
            latency_ms=result.latency_ms,
            items=formatted_items,
            error=result.error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing mobile image upload: {e}")
        raise HTTPException(status_code=500, detail=f"Server error processing OCR: {str(e)}")


@app.post("/api/v1/ocr/annotate")
async def annotate_ocr_image(
    file: UploadFile = File(..., description="JPEG or PNG image file"),
    language_hints: Optional[List[str]] = Query(default=config.LANGUAGE_HINTS),
):
    """
    Utility endpoint for mobile clients wanting an annotated JPEG image
    with bounding boxes and text labels pre-drawn on the image.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Decode image bytes to OpenCV numpy array
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode image file.")

    # Perform OCR
    result: VisionOCRResult = ocr_engine.detect_text_from_bytes(
        image_bytes=contents,
        language_hints=language_hints,
    )

    # Draw overlays
    annotated = draw_ocr_overlay(
        frame=frame,
        ocr_result=result,
        box_color=config.BOX_COLOR_BGR,
        text_color=config.TEXT_COLOR_BGR,
        bg_color=config.TEXT_BG_COLOR_BGR,
    )

    # Encode annotated frame back to JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    success, encoded_img = cv2.imencode(".jpg", annotated, encode_param)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_server:app", host="0.0.0.0", port=8000, reload=True)
