#!/usr/bin/env python3
"""Simple smoke test for trial API /health and /infer endpoints."""

import argparse
from datetime import datetime, timezone

import cv2
import requests


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for trial inference API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--image-path",
        default="output/frames/demo_board_with_arm.png",
        help="Path to input image for /infer",
    )
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    health_url = f"{args.base_url}/health"
    infer_url = f"{args.base_url}/infer"

    print(f"Checking {health_url}")
    health = requests.get(health_url, timeout=args.timeout)
    health.raise_for_status()
    print(f"health={health.json()}")

    image = cv2.imread(args.image_path)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image_path}")

    h, w = image.shape[:2]
    ok, jpg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        raise SystemExit("Could not encode image as JPEG")

    data = {
        "frame": "0",
        "timestamp": now_iso(),
        "camera_id": "smoke_test",
        "width": str(w),
        "height": str(h),
        "jpeg_quality": "75",
        "request_id": "smoke-test-001",
    }
    files = {"image": ("smoke.jpg", jpg.tobytes(), "image/jpeg")}

    print(f"Posting {infer_url}")
    response = requests.post(infer_url, data=data, files=files, timeout=args.timeout)
    response.raise_for_status()

    payload = response.json()
    detections = payload.get("detections", [])
    if not detections:
        raise SystemExit("No detections in API response")

    arm = detections[0].get("arm", {})
    print(f"infer_ok frame={detections[0].get('frame')} arm_detected={arm.get('detected')} bbox={arm.get('bbox')}")


if __name__ == "__main__":
    main()
