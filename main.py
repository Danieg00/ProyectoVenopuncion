"""
Main pipeline for Venipuncture AR Training System.
Integrates camera input, ChArUco detection, arm segmentation, and visualization.
"""

import cv2
import argparse
import time
import numpy as np
from pathlib import Path

from cameras.webcam_reader import WebcamReader
from models.charuco_detector import CharucoDetector
from models.arm_segmenter import ArmSegmenter
from utils.visualization import FrameAnnotator, VideoWriter
from utils.data_export import DetectionDataExporter
from utils.unity_export import UnityExporter
from utils.config import load_camera_calibration, RUNTIME_CONFIG


class VenopunctureARPipeline:
    """Main pipeline orchestrating detection and visualization."""
    
    def __init__(self, camera_source: str = "webcam", camera_id: int = 0,
                 output_video: bool = False, export_data: bool = True,
                 draw_visualization: bool = True, unity_host: str = None):
        """
        Initialize pipeline.
        
        Args:
            camera_source: Camera type ('webcam', 'kinect', 'picamera')
            camera_id: Camera index/ID
            output_video: Save output to video file
            export_data: Export frame data to JSON/CSV
            draw_visualization: Display real-time visualization
            unity_host: IP address to stream detection data to Unity (optional)
        """
        
        self.camera_source = camera_source
        self.camera_id = camera_id
        self.output_video = output_video
        self.export_data = export_data
        self.draw_visualization = draw_visualization
        self.unity_host = unity_host
        
        # Initialize components
        self.camera = None
        self.detector = None
        self.segmenter = None
        self.annotator = FrameAnnotator()
        self.video_writer = None
        self.data_exporter = None
        self.unity_exporter = None
        
        # Stats
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0
        self.last_frame_time = None
    
    def initialize(self) -> bool:
        """Initialize all components. Return True if successful."""
        
        print("\n" + "="*60)
        print("Venipuncture AR Training System - ChArUco Detection Pipeline")
        print("="*60 + "\n")
        
        # Initialize camera
        print(f"[1/4] Initializing camera: {self.camera_source}")
        if not self._init_camera():
            print("ERROR: Could not initialize camera")
            return False
        
        # Load camera calibration
        print(f"[2/4] Loading camera calibration")
        self._init_detector()
        
        # Initialize arm segmenter
        print(f"[3/4] Initializing arm segmenter")
        self._init_segmenter()
        
        # Initialize exporters
        print(f"[4/4] Initializing output modules")
        self._init_exporters()
        
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        print("\n✓ All components initialized successfully\n")
        return True
    
    def _init_camera(self) -> bool:
        """Initialize camera based on source."""
        
        if self.camera_source == "webcam":
            self.camera = WebcamReader(
                camera_id=self.camera_id,
                resolution=(1280, 720),
                fps=30
            )
        else:
            print(f"ERROR: Camera source '{self.camera_source}' not yet implemented")
            return False
        
        if not self.camera.open():
            return False
        
        print(f"  ✓ Camera initialized: {self.camera.get_resolution()}")
        return True
    
    def _init_detector(self):
        """Initialize ChArUco detector with calibration."""
        
        calib = load_camera_calibration(self.camera_source)
        
        # Convert dict to numpy array
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
        
        self.detector = CharucoDetector(camera_matrix=K, dist_coeffs=dist)
        print(f"  ✓ ChArUco detector initialized")
    
    def _init_segmenter(self):
        """Initialize arm segmenter."""
        
        has_depth = self.camera.has_depth() if self.camera else False
        self.segmenter = ArmSegmenter(use_depth=has_depth, use_ml=True)
        print(f"  ✓ Arm segmenter initialized (depth: {has_depth})")
    
    def _init_exporters(self):
        """Initialize video writer and data exporter."""
        
        if self.output_video:
            width, height = self.camera.get_resolution()
            output_path = "output/videos/venopuncture_output.mp4"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.video_writer = VideoWriter(output_path, fps=30,
                                           frame_size=(width, height))
            print(f"  ✓ Video writer initialized")
        
        if self.export_data:
            self.data_exporter = DetectionDataExporter()
            print(f"  ✓ Data exporter initialized")
        
        if self.unity_host:
            self.unity_exporter = UnityExporter(udp_host=self.unity_host, udp_port=5000)
            print(f"  ✓ Unity exporter initialized ({self.unity_host}:5000)")
    
    def run(self, max_frames: int = -1, verbose: bool = True):
        """
        Run main detection loop.
        
        Args:
            max_frames: Maximum frames to process (-1 for unlimited)
            verbose: Print frame stats
        """
        
        if not self._check_initialized():
            return False
        
        print("Starting detection loop (Press Q to quit, S to save frame)\n")
        
        try:
            while True:
                # Read frame
                success, rgb_frame, depth_frame = self.camera.read()
                
                if not success or rgb_frame is None:
                    print("\nERROR: Could not read frame from camera")
                    break
                
                # Detect board
                board_detection, board_pose = self.detector.detect_and_estimate_pose(
                    rgb_frame
                )
                
                # Segment arm
                arm_segmentation = self.segmenter.segment_arm(rgb_frame, depth_frame)
                
                # Annotate frame
                annotated = rgb_frame.copy()
                
                if board_detection.success:
                    annotated = self.detector.draw_detection(annotated, board_detection,
                                                            board_pose)
                
                if arm_segmentation.success:
                    annotated = self.segmenter.draw_segmentation(annotated,
                                                                arm_segmentation)
                
                # Add info and FPS
                annotated = self.annotator.annotate_frame(
                    annotated,
                    board_detection=board_detection,
                    board_pose=board_pose,
                    arm_segmentation=arm_segmentation,
                    fps=self.fps
                )
                
                # Export data
                if self.data_exporter:
                    self.data_exporter.export_frame(board_detection, board_pose,
                                                   arm_segmentation)
                
                # Export to Unity
                if self.unity_exporter:
                    frame_data = self.unity_exporter.build_frame_data(
                        self.frame_count, board_detection, board_pose, arm_segmentation
                    )
                    self.unity_exporter.send_to_unity(frame_data)
                
                # Write to video
                if self.video_writer:
                    self.video_writer.write(annotated)
                
                # Display
                if self.draw_visualization:
                    cv2.imshow("Venipuncture AR - Arm + Board Detection", annotated)
                
                # Update stats
                self.frame_count += 1
                self._update_fps()
                
                if verbose and self.frame_count % 30 == 0:
                    print(f"Frame {self.frame_count} | FPS: {self.fps:.1f} | "
                          f"Board: {board_detection.success} | Arm: {arm_segmentation.success}")
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n\nQuitting...")
                    break
                elif key == ord('s'):  # S for save
                    self._save_frame(annotated)
                
                # Check max frames
                if max_frames > 0 and self.frame_count >= max_frames:
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
        
        return True
    
    def _update_fps(self):
        """Update FPS calculation."""
        
        current_time = time.time()
        
        if self.last_frame_time:
            frame_time = current_time - self.last_frame_time
            if frame_time > 0:
                self.fps = 1.0 / frame_time
        
        self.last_frame_time = current_time
    
    def _save_frame(self, frame: np.ndarray):
        """Save current frame to file."""
        
        save_dir = Path("output/frames")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"frame_{self.frame_count:06d}.png"
        filepath = save_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"Frame saved: {filepath}")
    
    def _check_initialized(self) -> bool:
        """Check if all components are initialized."""
        
        if not self.camera or not self.detector or not self.segmenter:
            print("ERROR: Pipeline not properly initialized")
            return False
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        
        print("\nCleaning up...")
        
        if self.camera:
            self.camera.release()
            print("  ✓ Camera released")
        
        if self.video_writer:
            self.video_writer.release()
            print("  ✓ Video writer closed")
        
        if self.data_exporter:
            self.data_exporter.close()
            print("  ✓ Data exporter closed")
        
        if self.unity_exporter:
            self.unity_exporter.close()
            print("  ✓ Unity exporter closed")
        
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\nSession Summary:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"  Average FPS: {self.frame_count / elapsed:.1f}" if elapsed > 0 else "")
        print()


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Venipuncture AR Training System - ChArUco Detection"
    )
    
    parser.add_argument(
        "--camera",
        type=str,
        default="webcam",
        choices=["webcam", "kinect", "picamera"],
        help="Camera source"
    )
    
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera index (for webcam)"
    )
    
    parser.add_argument(
        "--output-video",
        action="store_true",
        help="Save output to video file"
    )
    
    parser.add_argument(
        "--export-data",
        action="store_true",
        default=True,
        help="Export detection data to JSON/CSV"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum frames to process (-1 for unlimited)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable real-time display"
    )
    
    parser.add_argument(
        "--unity-host",
        type=str,
        default=None,
        help="IP address to stream detection data to Unity (e.g., 192.168.1.100)"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = VenopunctureARPipeline(
        camera_source=args.camera,
        camera_id=args.camera_id,
        output_video=args.output_video,
        export_data=args.export_data,
        draw_visualization=not args.no_display,
        unity_host=args.unity_host
    )
    
    if pipeline.initialize():
        pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
