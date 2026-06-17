"""
Configuration module for Venipuncture AR Training System
Stores board specifications, camera parameters, and detection thresholds
"""

import yaml
import os
from pathlib import Path

# ============================================================================
# CHARUCO BOARD SPECIFICATIONS
# ============================================================================

BOARD_CONFIG = {
    "squares_x": 7,
    "squares_y": 5,
    "square_length": 0.04,  # 4 cm
    "marker_length": 0.02,  # 2 cm (50% of square)
    "dictionary": "DICT_4X4_50",  # Balance between speed and robustness
    "board_image_size": (1400, 1000),  # Pixels for printing
}

# ============================================================================
# CAMERA CALIBRATION (Placeholder - Load from yaml or compute once)
# ============================================================================

# Default camera matrix (approximate for 1080p, ~60° FOV)
# Should be replaced with actual calibration per camera
DEFAULT_CAMERA_MATRIX = {
    "fx": 800.0,  # Focal length x
    "fy": 800.0,  # Focal length y
    "cx": 640.0,  # Principal point x (image width / 2)
    "cy": 360.0,  # Principal point y (image height / 2)
}

DEFAULT_DISTORTION = {
    "k1": 0.0,
    "k2": 0.0,
    "p1": 0.0,
    "p2": 0.0,
    "k3": 0.0,
}

# ============================================================================
# ARM SEGMENTATION PARAMETERS
# ============================================================================

ARM_DETECTION_CONFIG = {
    # Depth-based segmentation (if using RGB-D camera)
    "depth_min_mm": 300,  # Minimum depth (arm is closer than background)
    "depth_max_mm": 800,  # Maximum depth (arm is not too close to camera)
    
    # Skin detection (HSV color space - adjustable per ethnicity)
    "skin_lower_H": 0,
    "skin_upper_H": 20,
    "skin_lower_S": 10,
    "skin_upper_S": 255,
    "skin_lower_V": 60,
    "skin_upper_V": 255,
    
    # Morphological operations
    "morphology_kernel_size": 5,
    "morphology_iterations": 2,
    
    # Contour filtering
    "min_contour_area": 500,  # Pixels^2, filter out noise
    "max_contour_area": 500000,  # Pixels^2, filter out background
}

# ============================================================================
# CHARUCO DETECTION PARAMETERS
# ============================================================================

CHARUCO_DETECTOR_CONFIG = {
    "adaptive_thresh_win_size_min": 3,
    "adaptive_thresh_win_size_max": 23,
    "adaptive_thresh_win_size_step": 10,
    "min_marker_perimeter_rate": 0.03,
    "max_marker_perimeter_rate": 4.0,
    "corner_refinement_method": "CORNER_REFINE_SUBPIX",  # SUBPIX, CONTOUR, or NONE
    "corner_refinement_win_size": 5,
    "relative_corner_size": 0.4,
    "detect_inverted_marker": False,
}

# ============================================================================
# VISUALIZATION & OUTPUT
# ============================================================================

OUTPUT_CONFIG = {
    "draw_board_corners": True,
    "draw_pose_axes": True,
    "draw_arm_mask": True,
    "axis_length": 0.05,  # 5 cm axes
    "fps_counter": True,
    "output_fps": 30,
    "frame_output_dir": "output/frames",
    "json_output_dir": "output/data",
    "video_output_dir": "output/videos",
}

# ============================================================================
# CAMERA SPECIFICATIONS
# ============================================================================

CAMERA_CONFIGS = {
    "kinect": {
        "type": "rgb-d",
        "resolution": (1920, 1080),
        "fps": 30,
        "has_depth": True,
    },
    "picamera": {
        "type": "rgb-d",
        "resolution": (1280, 720),
        "fps": 30,
        "has_depth": True,  # If using Kinect v2 or pivariety stereo module
    },
    "webcam": {
        "type": "rgb",
        "resolution": (1280, 720),
        "fps": 30,
        "has_depth": False,
    },
}

# ============================================================================
# RUNTIME SETTINGS
# ============================================================================

RUNTIME_CONFIG = {
    "target_fps": 30,
    "enable_gpu": False,
    "verbose": True,
    "auto_camera_fallback": True,  # Try next camera if current fails
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_camera_calibration(camera_type: str = "webcam") -> dict:
    """Load camera calibration from yaml file or return default."""
    calib_path = Path(__file__).parent.parent / "calibration" / f"{camera_type}_calib.yaml"
    
    if calib_path.exists():
        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)
            return {
                "matrix": data.get("camera_matrix", DEFAULT_CAMERA_MATRIX),
                "distortion": data.get("distortion_coefficients", DEFAULT_DISTORTION)
            }
    else:
        return {
            "matrix": DEFAULT_CAMERA_MATRIX,
            "distortion": DEFAULT_DISTORTION
        }


def save_camera_calibration(camera_type: str, camera_matrix: dict, distortion: dict):
    """Save camera calibration to yaml file."""
    calib_path = Path(__file__).parent.parent / "calibration" / f"{camera_type}_calib.yaml"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "camera_type": camera_type,
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion,
    }
    
    with open(calib_path, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Calibration saved to {calib_path}")


def create_output_directories():
    """Create output directories if they don't exist."""
    base_path = Path(__file__).parent.parent
    for dir_key in OUTPUT_CONFIG:
        if "dir" in dir_key:
            dir_path = base_path / OUTPUT_CONFIG[dir_key]
            dir_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_output_directories()
    print("Output directories created successfully")
    print(f"Board config: {BOARD_CONFIG}")
    print(f"Camera calibration (default): {load_camera_calibration()}")
