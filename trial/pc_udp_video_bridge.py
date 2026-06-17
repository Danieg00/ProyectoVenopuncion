#!/usr/bin/env python3
"""Receive an rpicam-vid UDP H.264 stream and forward Unity-ready JSON.

This bridge is the Pi 3B+ path for the current project slice:
Pi camera -> UDP H.264 stream -> PC decode -> lightweight frame analysis -> Unity JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.unity_export import UnityExporter


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must look like 1536x864")
    return int(parts[0]), int(parts[1])


def detect_arm_bbox(frame_bgr: np.ndarray) -> tuple[bool, Optional[dict]]:
    """Lightweight skin-tone proxy detector for transport validation."""

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
    if cv2.contourArea(largest) < 2500:
        return False, None

    x, y, w, h = cv2.boundingRect(largest)
    return True, {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
    }


def build_unity_payload(frame_index: int, timestamp: str, frame_bgr: np.ndarray) -> dict:
    arm_detected, arm_bbox = detect_arm_bbox(frame_bgr)

    return {
        "detections": [
            {
                "frame": int(frame_index),
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


def open_video_capture(input_uri: str, backend: str, retry_seconds: float) -> cv2.VideoCapture:
    backend_map = {
        "auto": [None, cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER],
        "ffmpeg": [cv2.CAP_FFMPEG],
        "gstreamer": [cv2.CAP_GSTREAMER],
    }
    candidate_backends = backend_map[backend]

    deadline = None if retry_seconds <= 0 else time.perf_counter() + retry_seconds

    while True:
        for candidate in candidate_backends:
            if candidate is None:
                capture = cv2.VideoCapture(input_uri)
            else:
                capture = cv2.VideoCapture(input_uri, candidate)

            if capture.isOpened():
                return capture

            capture.release()

        if deadline is not None and time.perf_counter() >= deadline:
            break

        print(f"Waiting for stream: {input_uri}", flush=True)
        time.sleep(1.0)

    raise SystemExit(
        "Could not open the UDP video stream. Check the Pi command, the laptop IP, and OpenCV backend support."
    )


def save_payload(output_dir: Path, frame_index: int, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"frame_{frame_index:06d}_unity.json"
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi UDP video -> Unity JSON bridge")
    parser.add_argument(
        "--input-uri",
        default="udp://@:5000?fifo_size=5000000&overrun_nonfatal=1",
        help="Input stream URI for the PC receiver",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "gstreamer"],
        default="auto",
        help="OpenCV backend used to decode the stream",
    )
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=15.0,
        help="How long to wait for the stream to appear before exiting",
    )
    parser.add_argument("--unity-host", required=True, help="Unity machine IP for UDP JSON output")
    parser.add_argument("--unity-port", type=int, default=5000, help="Unity UDP port")
    parser.add_argument("--camera-id", default="picamera3_noir", help="Camera label for logs")
    parser.add_argument("--resolution", type=parse_resolution, default=(1536, 864), help="Expected input size")
    parser.add_argument("--no-display", action="store_true", help="Disable local preview window")
    parser.add_argument("--max-frames", type=int, default=-1, help="Stop after N frames (-1 for unlimited)")
    parser.add_argument(
        "--save-json-dir",
        default="output/data",
        help="Optional directory for one JSON file per frame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_width, expected_height = args.resolution

    print("=" * 72)
    print("Pi UDP Video -> Unity JSON Bridge")
    print("=" * 72)
    print(f"Input stream: {args.input_uri}")
    print(f"Unity target: {args.unity_host}:{args.unity_port}")
    print(f"Expected input: {expected_width}x{expected_height} | backend={args.backend}")

    capture = open_video_capture(args.input_uri, args.backend, args.stream_timeout)
    unity_exporter = UnityExporter(udp_host=args.unity_host, udp_port=args.unity_port)
    json_output_dir = Path(args.save_json_dir) if args.save_json_dir else None

    frame_index = 0
    sent_ok = 0
    dropped = 0
    start_time = time.perf_counter()

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                dropped += 1
                if dropped % 30 == 0:
                    print(f"frame={frame_index} read_failed drops={dropped}", flush=True)
                time.sleep(0.01)
                continue

            dropped = 0
            height, width = frame_bgr.shape[:2]
            timestamp = now_iso()
            payload = build_unity_payload(frame_index, timestamp, frame_bgr)

            if unity_exporter.send_to_unity(payload):
                sent_ok += 1

            if json_output_dir is not None:
                save_payload(json_output_dir, frame_index, payload)

            if not args.no_display:
                display = frame_bgr.copy()
                arm = payload["detections"][0]["arm"]
                if arm["detected"] and arm["bbox"]:
                    bbox = arm["bbox"]
                    x = int(bbox["x"])
                    y = int(bbox["y"])
                    w = int(bbox["width"])
                    h = int(bbox["height"])
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        display,
                        "arm",
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    display,
                    f"frame={frame_index} {width}x{height}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Pi UDP Video Bridge", display)

            if frame_index % 30 == 0:
                elapsed = time.perf_counter() - start_time
                fps = frame_index / elapsed if elapsed > 0 else 0.0
                arm_detected = payload["detections"][0]["arm"]["detected"]
                print(
                    f"frame={frame_index} size={width}x{height} arm={arm_detected} sent_ok={sent_ok} fps={fps:.1f}",
                    flush=True,
                )

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("Interrupted by user", flush=True)
    finally:
        capture.release()
        unity_exporter.close()
        if not args.no_display:
            cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start_time
        avg_fps = frame_index / elapsed if elapsed > 0 else 0.0
        print("Done.")
        print(f"  frames={frame_index}")
        print(f"  sent_ok={sent_ok}")
        print(f"  avg_fps={avg_fps:.1f}")
        if json_output_dir is not None:
            print(f"  json_dir={json_output_dir}")


if __name__ == "__main__":
    main()