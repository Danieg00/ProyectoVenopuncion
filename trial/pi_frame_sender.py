#!/usr/bin/env python3
"""Trial v1 Pi sender: capture with Picamera2 and POST frames to PC API."""

import argparse
import time
from datetime import datetime, timezone

import cv2
import requests

try:
    from picamera2 import Picamera2
except ImportError as exc:
    raise SystemExit(
        "picamera2 is required on Raspberry Pi. Install with: sudo apt install python3-picamera2"
    ) from exc


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must look like 640x480")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi Camera -> PC Trial Sender")
    parser.add_argument("--api-url", required=True, help="Inference URL, e.g. http://192.168.1.20:8000/infer")
    parser.add_argument("--camera-id", default="pi3b_cam3", help="Camera id label sent to API")
    parser.add_argument("--resolution", type=parse_resolution, default=(640, 480), help="Capture resolution WxH")
    parser.add_argument("--fps", type=float, default=7.0, help="Target capture FPS")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality (0-100)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Stop after N frames (-1 infinite)")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width, height = args.resolution

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.4)

    frame_idx = 0
    sent_ok = 0
    start = time.perf_counter()
    frame_period = 1.0 / max(args.fps, 0.1)

    print(
        f"Starting sender -> {args.api_url} | res={width}x{height} fps={args.fps} q={args.jpeg_quality}",
        flush=True,
    )

    try:
        while True:
            loop_start = time.perf_counter()

            rgb = picam2.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            ok, jpg = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(max(0, min(100, args.jpeg_quality)))],
            )
            if not ok:
                print(f"frame={frame_idx} encode_failed", flush=True)
                continue

            ts = now_iso()
            data = {
                "frame": str(frame_idx),
                "timestamp": ts,
                "camera_id": args.camera_id,
                "width": str(width),
                "height": str(height),
                "jpeg_quality": str(args.jpeg_quality),
            }
            files = {"image": (f"frame_{frame_idx:06d}.jpg", jpg.tobytes(), "image/jpeg")}

            try:
                response = requests.post(args.api_url, data=data, files=files, timeout=args.timeout)
                response.raise_for_status()
                _ = response.json()
                sent_ok += 1
                print(f"frame={frame_idx} status=ok code={response.status_code}", flush=True)
            except Exception as exc:
                print(f"frame={frame_idx} status=error err={exc}", flush=True)
                time.sleep(0.25)

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            elapsed = time.perf_counter() - loop_start
            sleep_s = frame_period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("Interrupted by user", flush=True)
    finally:
        picam2.stop()
        total = time.perf_counter() - start
        avg_fps = frame_idx / total if total > 0 else 0.0
        print(
            f"Done. frames={frame_idx} sent_ok={sent_ok} total_s={total:.1f} avg_fps={avg_fps:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
