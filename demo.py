"""
Demo script to test ChArUco detection without a physical camera.
Generates a virtual board image and tests detection on it.
Supports custom arm images and Unity export.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from models.charuco_detector import CharucoDetector
from models.arm_segmenter import ArmSegmenter
from utils.visualization import FrameAnnotator
from utils.config import BOARD_CONFIG
from utils.data_export import DetectionDataExporter


def create_demo_board_image(scale: float = 1.0) -> np.ndarray:
    """
    Create a virtual ChArUco board image for testing.
    
    Args:
        scale: Scaling factor (1.0 = 1400x1000px)
    
    Returns:
        Board image as numpy array
    """
    
    image_size = (int(BOARD_CONFIG["board_image_size"][0] * scale),
                  int(BOARD_CONFIG["board_image_size"][1] * scale))
    
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, BOARD_CONFIG["dictionary"])
    )
    
    board = cv2.aruco.CharucoBoard(
        size=(BOARD_CONFIG["squares_x"], BOARD_CONFIG["squares_y"]),
        squareLength=BOARD_CONFIG["square_length"],
        markerLength=BOARD_CONFIG["marker_length"],
        dictionary=dictionary
    )
    
    board_image = board.generateImage(
        outSize=image_size,
        marginSize=int(10 * scale),
        borderBits=1
    )
    
    # Add white background to simulate real conditions
    final_image = np.ones((int(1200 * scale), int(1600 * scale), 3), dtype=np.uint8) * 255
    
    # Place board in center
    y_offset = int((final_image.shape[0] - board_image.shape[0]) / 2)
    x_offset = int((final_image.shape[1] - board_image.shape[1]) / 2)
    
    final_image[y_offset:y_offset + board_image.shape[0],
                x_offset:x_offset + board_image.shape[1]] = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
    
    return final_image


def add_arm_to_image(image: np.ndarray, arm_image_path: str = None) -> np.ndarray:
    """
    Add an arm to the image for testing arm segmentation.
    
    Args:
        image: Background image
        arm_image_path: Optional path to custom arm image (PNG with transparency)
    
    Returns:
        Image with arm overlay
    """
    
    h, w = image.shape[:2]
    output = image.copy()
    
    if arm_image_path and Path(arm_image_path).exists():
        # Load custom arm image
        arm_img = cv2.imread(arm_image_path, cv2.IMREAD_UNCHANGED)
        if arm_img is None:
            print(f"WARNING: Could not load arm image from {arm_image_path}, using synthetic arm")
            return add_synthetic_arm(output)
        
        # Resize arm image to fit on board
        arm_h, arm_w = arm_img.shape[:2]
        target_w = int(w * 0.6)  # 60% of image width
        scale = target_w / arm_w
        arm_h_resized = int(arm_h * scale)
        arm_img = cv2.resize(arm_img, (target_w, arm_h_resized))
        
        # Position arm on lower half of image
        y_pos = h - arm_h_resized - int(h * 0.1)
        x_pos = int((w - target_w) / 2)
        
        # Apply alpha blending if image has transparency
        if arm_img.shape[2] == 4:
            arm_bgr = arm_img[:, :, :3]
            arm_alpha = arm_img[:, :, 3].astype(float) / 255.0
            
            for c in range(3):
                output[y_pos:y_pos + arm_h_resized, x_pos:x_pos + target_w, c] = \
                    (arm_bgr[:, :, c] * arm_alpha + 
                     output[y_pos:y_pos + arm_h_resized, x_pos:x_pos + target_w, c] * (1 - arm_alpha)).astype(np.uint8)
        else:
            output[y_pos:y_pos + arm_h_resized, x_pos:x_pos + target_w] = arm_img[:, :, :3]
        
        return output
    
    return add_synthetic_arm(output)


def add_synthetic_arm(image: np.ndarray) -> np.ndarray:
    """Add a simulated arm to the image (synthetic fallback)."""
    
    h, w = image.shape[:2]
    output = image.copy()
    
    # Draw a simulated arm (skin-colored rectangle)
    arm_color = (155, 120, 100)  # BGR skin tone
    cv2.rectangle(output, (w//4, h//2), (3*w//4, 4*h//5), arm_color, -1)
    
    # Add some arm-like shape (ellipse for vein region)
    cv2.ellipse(output, (w//2, 3*h//5), (w//8, h//6), 0, 0, 360, (100, 80, 60), 5)
    
    return output


def export_for_unity(detection_data: dict, output_path: str = "output/data/unity_detection.json"):
    """
    Export detection results in a format optimized for Unity visualization.
    
    Args:
        detection_data: Detection results dictionary
        output_path: Path to save Unity-compatible JSON
    """
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(detection_data, f, indent=2)
    
    print(f"✓ Unity-compatible export: {output_path}")


def demo_detection(arm_image_path: str = None, export_unity: bool = True):
    """
    Run detection demo on generated board image.
    
    Args:
        arm_image_path: Optional path to custom arm image
        export_unity: Export results for Unity visualization
    """
    
    print("\n" + "="*60)
    print("ChArUco Detection Demo (No Camera Required)")
    print("="*60 + "\n")
    
    # Create detector and segmenter
    print("[1/5] Initializing detector...")
    detector = CharucoDetector()
    
    print("[2/5] Initializing arm segmenter...")
    segmenter = ArmSegmenter(use_depth=False, use_ml=False)  # No ML needed for demo
    
    print("[3/5] Generating virtual board image...")
    board_image = create_demo_board_image()
    
    print("[4/5] Testing detection...\n")
    
    # Test 1: Plain board
    print("--- Test 1: Plain Board ---")
    detection1, pose1 = detector.detect_and_estimate_pose(board_image)
    annotated1 = detector.draw_detection(board_image, detection1, pose1)
    
    result = "✓ DETECTED" if detection1.success else "✗ NOT DETECTED"
    print(f"Board detection: {result}")
    if pose1.success:
        print(f"  Distance: {pose1.distance:.2f}m")
        print(f"  Rotation vector: {pose1.rvec}")
        print(f"  Translation vector: {pose1.tvec}")
    
    cv2.imwrite("output/frames/demo_board_only.png", annotated1)
    print("  Saved: output/frames/demo_board_only.png\n")
    
    # Test 2: Board with simulated arm
    print("--- Test 2: Board with Simulated Arm ---")
    board_with_arm = add_arm_to_image(board_image)
    detection2, pose2 = detector.detect_and_estimate_pose(board_with_arm)
    annotated2 = detector.draw_detection(board_with_arm, detection2, pose2)
    
    result = "✓ DETECTED" if detection2.success else "✗ NOT DETECTED"
    print(f"Board detection: {result}")
    
    arm_result = segmenter.segment_arm(board_with_arm)
    result = "✓ DETECTED" if arm_result.success else "✗ NOT DETECTED"
    print(f"Arm segmentation: {result}")
    
    if arm_result.success and arm_result.bbox:
        x, y, w, h = arm_result.bbox
        print(f"  Arm bounding box: ({x}, {y}) {w}x{h}")
    
    # Draw arm segmentation
    if arm_result.success:
        annotated2 = segmenter.draw_segmentation(annotated2, arm_result, alpha=0.3)
    
    cv2.imwrite("output/frames/demo_board_with_arm.png", annotated2)
    print("  Saved: output/frames/demo_board_with_arm.png\n")
    
    # Test 3: Custom arm image (if provided)
    if arm_image_path:
        print("--- Test 3: Board with Custom Arm Image ---")
        board_with_custom_arm = add_arm_to_image(board_image, arm_image_path)
        detection3, pose3 = detector.detect_and_estimate_pose(board_with_custom_arm)
        annotated3 = detector.draw_detection(board_with_custom_arm, detection3, pose3)
        
        result = "✓ DETECTED" if detection3.success else "✗ NOT DETECTED"
        print(f"Board detection: {result}")
        
        arm_custom_result = segmenter.segment_arm(board_with_custom_arm)
        result = "✓ DETECTED" if arm_custom_result.success else "✗ NOT DETECTED"
        print(f"Arm segmentation: {result}")
        
        if arm_custom_result.success:
            annotated3 = segmenter.draw_segmentation(annotated3, arm_custom_result, alpha=0.3)
        
        cv2.imwrite("output/frames/demo_board_with_custom_arm.png", annotated3)
        print("  Saved: output/frames/demo_board_with_custom_arm.png\n")
    
    # Test 4: Export for Unity (if enabled)
    if export_unity:
        print("[5/5] Exporting for Unity visualization...")
        
        # Prepare Unity-compatible data from Test 2
        frame_h, frame_w = board_with_arm.shape[:2]
        
        unity_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "image_resolution": {
                    "width": frame_w,
                    "height": frame_h
                },
                "demo": True,
                "note": "This is synthetic demo data. Replace with real camera data for production use."
            },
            "detections": [
                {
                    "frame": 0,
                    "board": {
                        "detected": detection2.success,
                        "corners": detection2.corners.tolist() if detection2.corners is not None else None,
                        "ids": detection2.ids.tolist() if detection2.ids is not None else None,
                        "pose": {
                            "rvec": pose2.rvec.tolist() if pose2.rvec is not None else None,
                            "tvec": pose2.tvec.tolist() if pose2.tvec is not None else None,
                            "distance_m": float(pose2.distance) if pose2.distance else None
                        } if pose2.success else None
                    },
                    "arm": {
                        "detected": arm_result.success,
                        "bbox": {
                            "x": int(arm_result.bbox[0]),
                            "y": int(arm_result.bbox[1]),
                            "width": int(arm_result.bbox[2]),
                            "height": int(arm_result.bbox[3])
                        } if arm_result.bbox else None
                    }
                }
            ]
        }
        
        export_for_unity(unity_data)
    
    # Interactive display
    print("\n--- Display Results ---")
    print("Displaying images. Press any key to continue or ESC to exit early.\n")
    
    cv2.imshow("Demo 1: Plain Board", annotated1)
    cv2.waitKey(2000)
    
    cv2.imshow("Demo 2: Board with Arm", annotated2)
    cv2.waitKey(2000)
    
    cv2.destroyAllWindows()
    
    print("✓ Demo complete!")
    print("\nGenerated files:")
    print("  • output/frames/demo_board_only.png")
    print("  • output/frames/demo_board_with_arm.png")
    if export_unity:
        print("  • output/data/unity_detection.json (for Unity integration)")
    print("\nNext steps:")
    print("  1. Review the output images in output/frames/")
    print("  2. For Unity integration, see unity/README.md")
    print("  3. When ready with a camera, run: python main.py --camera webcam")
    print("  4. Use charuco_board_generator.py to print the board\n")


if __name__ == "__main__":
    import argparse
    
    # Create output directory
    Path("output/frames").mkdir(parents=True, exist_ok=True)
    Path("output/data").mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="ChArUco Detection Demo")
    parser.add_argument(
        "--arm-image",
        type=str,
        default=None,
        help="Path to custom arm image (PNG with transparency recommended)"
    )
    parser.add_argument(
        "--export-unity",
        action="store_true",
        default=True,
        help="Export data in Unity-compatible JSON format"
    )
    parser.add_argument(
        "--no-unity-export",
        action="store_true",
        help="Disable Unity export"
    )
    
    args = parser.parse_args()
    export_unity = not args.no_unity_export
    
    demo_detection(arm_image_path=args.arm_image, export_unity=export_unity)
