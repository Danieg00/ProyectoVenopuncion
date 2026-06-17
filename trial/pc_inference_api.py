#!/usr/bin/env python3
"""Trial v1 PC API: receive frames from Pi, return JSON, forward to Unity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from utils.unity_export import UnityExporter


app = FastAPI(title="Venipuncture Trial API", version="1.0")


UNITY_EXPORTER: Optional[UnityExporter] = None


def detect_arm_bbox(frame_bgr: np.ndarray) -> tuple[bool, Optional[dict]]:
    """Simple HSV-based arm proxy detector for transport trials."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 40], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < 2500:
        return False, None

    x, y, w, h = cv2.boundingRect(largest)
    return True, {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    frame: int = Form(...),
    timestamp: str = Form(...),
    camera_id: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    jpeg_quality: Optional[int] = Form(None),
    request_id: Optional[str] = Form(None),
) -> dict:
    start = time.perf_counter()

    if image.content_type not in ("image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_content_type", "message": "image must be JPEG"},
        )

    payload = await image.read()
    np_buf = np.frombuffer(payload, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "decode_failed", "message": "could not decode JPEG"},
        )

    h, w = frame_bgr.shape[:2]
    if width != w or height != h:
        width, height = w, h

    arm_detected, arm_bbox = detect_arm_bbox(frame_bgr)

    response = {
        "detections": [
            {
                "frame": int(frame),
                "timestamp": timestamp,
                "board": {
                    "detected": False,
                    "corners": None,
                    "ids": None,
                    "pose": None,
                },
                "arm": {
                    "detected": arm_detected,
                    "bbox": arm_bbox,
                },
                "vein": {
                    "detected": False,
                    "contour": [],
                    "confidence": 0.0,
                },
            }
        ]
    }

    udp_ok = False
    if UNITY_EXPORTER is not None:
        udp_ok = UNITY_EXPORTER.send_to_unity(response)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(
        (
            f"frame={frame} cam={camera_id} size={width}x{height} "
            f"jpeg_q={jpeg_quality} request_id={request_id} "
            f"arm={arm_detected} udp_sent={udp_ok} t_ms={elapsed_ms:.1f}"
        ),
        flush=True,
    )

    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trial API (Pi -> PC -> Unity)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API bind host")
    parser.add_argument("--port", type=int, default=8000, help="API bind port")
    parser.add_argument(
        "--unity-host",
        type=str,
        default=None,
        help="Unity IP to forward JSON via UDP (optional)",
    )
    parser.add_argument("--unity-port", type=int, default=5000, help="Unity UDP port")
    return parser.parse_args()


def main() -> None:
    global UNITY_EXPORTER

    args = parse_args()
    if args.unity_host:
        UNITY_EXPORTER = UnityExporter(udp_host=args.unity_host, udp_port=args.unity_port)
        print(f"Unity UDP bridge enabled: {args.unity_host}:{args.unity_port}", flush=True)
    else:
        print("Unity UDP bridge disabled (no --unity-host)", flush=True)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
