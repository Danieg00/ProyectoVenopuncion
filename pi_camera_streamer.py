#!/usr/bin/env python3
"""
Raspberry Pi Camera Streamer for Unity Integration.
Streams detection data via UDP to a networked Unity application.

Usage:
    python pi_camera_streamer.py --unity-host <IP> --camera-id 0
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path

from cameras.webcam_reader import WebcamReader
from models.charuco_detector import CharucoDetector
from models.arm_segmenter import ArmSegmenter
from utils.visualization import FrameAnnotator
from utils.unity_export import UnityExporter
from utils.config import load_camera_calibration


def main():
    parser = argparse.ArgumentParser(
        description="Stream Venipuncture detection data to Unity on local network"
    )
    parser.add_argument("--unity-host", required=True, 
                       help="IP address of Unity machine (e.g., 192.168.1.100)")
    parser.add_argument("--camera-id", type=int, default=0, 
                       help="Camera ID (default: 0)")
    parser.add_argument("--resolution", type=str, default="640x480", 
                       help="Camera resolution (default: 640x480)")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable local display (useful for headless Pi)")
    parser.add_argument("--max-frames", type=int, default=-1,
                       help="Max frames to process (-1 for unlimited)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Pi Camera → Unity Streamer")
    print("="*60 + "\n")
    
    # Parse resolution
    res_parts = args.resolution.split('x')
    width, height = int(res_parts[0]), int(res_parts[1])
    
    # Initialize components
    print(f"[1/4] Initializing camera {args.camera_id}...")
    camera = WebcamReader(camera_id=args.camera_id, resolution=(width, height), fps=15)
    if not camera.open():
        print("ERROR: Could not open camera")
        return
    
    print(f"[2/4] Initializing detector...")
    calib = load_camera_calibration("webcam")
    K = np.array([
        [calib['matrix']['fx'], 0, calib['matrix']['cx']],
        [0, calib['matrix']['fy'], calib['matrix']['cy']],
        [0, 0, 1]
    ], dtype=np.float32)
    dist = np.array([
        calib['distortion']['k1'],
        calib['distortion']['k2'],
        calib['distortion']['p1'],
        calib['distortion']['p2'],
        calib['distortion']['k3']
    ], dtype=np.float32)
    detector = CharucoDetector(camera_matrix=K, dist_coeffs=dist)
    
    print(f"[3/4] Initializing segmenter...")
    segmenter = ArmSegmenter(use_depth=False, use_ml=False)  # Minimal for Pi
    
    print(f"[4/4] Initializing Unity exporter...")
    unity_exporter = UnityExporter(udp_host=args.unity_host, udp_port=5000)
    
    annotator = FrameAnnotator()
    frame_count = 0
    start_time = time.time()
    
    print(f"\nStreaming to {args.unity_host}:5000")
    print("Press Q or ESC to stop\n")
    
    try:
        while True:
            # Read frame
            success, rgb_frame, depth_frame = camera.read()
            if not success or rgb_frame is None:
                print("ERROR: Could not read frame")
                break
            
            # Detect
            board_detection, board_pose = detector.detect_and_estimate_pose(rgb_frame)
            arm_segmentation = segmenter.segment_arm(rgb_frame, depth_frame)
            
            # Prepare data
            frame_data = unity_exporter.build_frame_data(
                frame_count, board_detection, board_pose, arm_segmentation
            )
            
            # Send to Unity
            unity_exporter.send_to_unity(frame_data)
            
            # Local visualization (optional)
            if not args.no_display:
                annotated = rgb_frame.copy()
                if board_detection.success:
                    annotated = detector.draw_detection(annotated, board_detection, board_pose)
                if arm_segmentation.success:
                    annotated = segmenter.draw_segmentation(annotated, arm_segmentation)
                annotated = annotator.annotate_frame(annotated, board_detection, board_pose, arm_segmentation)
                
                cv2.imshow("Streaming to Unity", annotated)
            
            # Status
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frame {frame_count} | FPS: {fps:.1f} | Board: {board_detection.success} | Arm: {arm_segmentation.success}")
            
            # Check exit
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            
            # Check max frames
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print("\nCleaning up...")
        camera.release()
        unity_exporter.close()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\nStreaming complete:")
        print(f"  Frames: {frame_count}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Avg FPS: {frame_count / elapsed:.1f}")


if __name__ == "__main__":
    main()
