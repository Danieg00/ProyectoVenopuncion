"""
ChArUco board detection and pose estimation.
Core module for detecting the ChArUco board and estimating its 3D pose.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from utils.config import BOARD_CONFIG, CHARUCO_DETECTOR_CONFIG


@dataclass
class BoardDetection:
    """Results of ChArUco board detection."""
    success: bool
    corners: Optional[np.ndarray] = None  # (N, 2) detected board corner points
    ids: Optional[np.ndarray] = None      # (N,) IDs of detected corners
    marker_corners: Optional[List] = None
    marker_ids: Optional[np.ndarray] = None


@dataclass
class BoardPose:
    """3D pose of the ChArUco board."""
    success: bool
    rvec: Optional[np.ndarray] = None     # (3,) Rotation vector
    tvec: Optional[np.ndarray] = None     # (3,) Translation vector
    distance: Optional[float] = None      # Distance from camera to board center


class CharucoDetector:
    """ChArUco board detector with pose estimation."""
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize ChArUco detector.
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3). If None, uses default.
            dist_coeffs: Camera distortion coefficients (5x1 or 4x1). If None, uses default.
        """
        
        # Initialize board
        self.board_squares = (BOARD_CONFIG["squares_x"], BOARD_CONFIG["squares_y"])
        self.square_length = BOARD_CONFIG["square_length"]
        self.marker_length = BOARD_CONFIG["marker_length"]
        
        # Get dictionary
        dict_name = BOARD_CONFIG["dictionary"]
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name)
        )
        
        # Create ChArUco board
        self.board = cv2.aruco.CharucoBoard(
            size=self.board_squares,
            squareLength=self.square_length,
            markerLength=self.marker_length,
            dictionary=self.dictionary
        )
        
        # Initialize detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # Configure detector parameters from config
        self.detector_params.adaptiveThreshWinSizeMin = \
            CHARUCO_DETECTOR_CONFIG["adaptive_thresh_win_size_min"]
        self.detector_params.adaptiveThreshWinSizeMax = \
            CHARUCO_DETECTOR_CONFIG["adaptive_thresh_win_size_max"]
        self.detector_params.adaptiveThreshWinSizeStep = \
            CHARUCO_DETECTOR_CONFIG["adaptive_thresh_win_size_step"]
        self.detector_params.minMarkerPerimeterRate = \
            CHARUCO_DETECTOR_CONFIG["min_marker_perimeter_rate"]
        self.detector_params.maxMarkerPerimeterRate = \
            CHARUCO_DETECTOR_CONFIG["max_marker_perimeter_rate"]
        self.detector_params.detectInvertedMarker = \
            CHARUCO_DETECTOR_CONFIG["detect_inverted_marker"]
        
        # Corner refinement
        corner_refine_name = CHARUCO_DETECTOR_CONFIG["corner_refinement_method"]
        if corner_refine_name == "CORNER_REFINE_SUBPIX":
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        elif corner_refine_name == "CORNER_REFINE_CONTOUR":
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        else:
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        
        self.detector_params.cornerRefinementWinSize = \
            CHARUCO_DETECTOR_CONFIG["corner_refinement_win_size"]
        
        # Create ArUco detector (detects individual markers)
        self.arucoDetector = cv2.aruco.ArucoDetector(
            self.dictionary, self.detector_params
        )
        
        # Create ChArUco detector with CharucoParameters
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        self.charucoDetector = cv2.aruco.CharucoDetector(
            self.board, charuco_params, self.detector_params
        )
        
        # Camera parameters
        if camera_matrix is None:
            # Default camera matrix (1920x1080, ~60° FOV)
            self.camera_matrix = np.array([
                [800.0, 0.0, 960.0],
                [0.0, 800.0, 540.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix.astype(np.float32)
        
        if dist_coeffs is None:
            # No distortion by default
            self.dist_coeffs = np.zeros((5,), dtype=np.float32)
        else:
            self.dist_coeffs = np.array(dist_coeffs).astype(np.float32)
            if self.dist_coeffs.size == 4:
                self.dist_coeffs = np.append(self.dist_coeffs, 0.0)
        
        print("ChArUco Detector initialized")
        print(f"  Board: {self.board_squares[0]}x{self.board_squares[1]} squares")
        print(f"  Dictionary: {dict_name}")
        print(f"  Marker size: {self.marker_length*100:.1f} cm")
    
    def set_camera_params(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Update camera parameters (e.g., after calibration)."""
        self.camera_matrix = camera_matrix.astype(np.float32)
        self.dist_coeffs = np.array(dist_coeffs).astype(np.float32)
        if self.dist_coeffs.size == 4:
            self.dist_coeffs = np.append(self.dist_coeffs, 0.0)
    
    def detect_board(self, frame: np.ndarray) -> BoardDetection:
        """
        Detect ChArUco board in frame.
        
        Args:
            frame: Input image (BGR, any size)
        
        Returns:
            BoardDetection with detected corners and IDs
        """
        
        if frame is None or frame.size == 0:
            return BoardDetection(success=False)
        
        try:
            # Detect ChArUco board
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                self.charucoDetector.detectBoard(frame)
            
            if charuco_corners is None or len(charuco_corners) < 4:
                return BoardDetection(success=False)
            
            # Ensure corners are in correct shape (N, 2)
            if charuco_corners.ndim == 3:
                charuco_corners = charuco_corners.reshape(-1, 2)
            
            return BoardDetection(
                success=True,
                corners=charuco_corners,
                ids=charuco_ids,
                marker_corners=marker_corners,
                marker_ids=marker_ids
            )
        
        except Exception as e:
            print(f"Error detecting board: {e}")
            return BoardDetection(success=False)
    
    def estimate_pose(self, detection: BoardDetection) -> BoardPose:
        """
        Estimate 3D pose of the board from detected corners.
        
        Args:
            detection: BoardDetection result
        
        Returns:
            BoardPose with rotation and translation vectors
        """
        
        if not detection.success or detection.corners is None:
            return BoardPose(success=False)
        
        try:
            # Get 3D object points from board
            obj_points, img_points = self.board.matchImagePoints(
                detection.marker_corners,
                detection.marker_ids
            )
            
            if obj_points is None or len(obj_points) < 4:
                return BoardPose(success=False)
            
            # Solve PnP (Perspective-n-Point)
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=obj_points,
                imagePoints=img_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return BoardPose(success=False)
            
            # Calculate distance from camera to board
            distance = np.linalg.norm(tvec)
            
            return BoardPose(
                success=True,
                rvec=rvec.flatten(),
                tvec=tvec.flatten(),
                distance=distance
            )
        
        except Exception as e:
            print(f"Error estimating pose: {e}")
            return BoardPose(success=False)
    
    def detect_and_estimate_pose(self, frame: np.ndarray) -> Tuple[BoardDetection, BoardPose]:
        """
        Detect board and estimate pose in one call.
        
        Args:
            frame: Input image
        
        Returns:
            (BoardDetection, BoardPose) tuple
        """
        detection = self.detect_board(frame)
        pose = self.estimate_pose(detection)
        return detection, pose
    
    def draw_detection(self, frame: np.ndarray, detection: BoardDetection, 
                      pose: Optional[BoardPose] = None) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Input image
            detection: BoardDetection result
            pose: BoardPose result (optional, draws axes if provided)
        
        Returns:
            Annotated frame
        """
        
        output = frame.copy()
        
        if not detection.success:
            return output
        
        # Draw detected corners
        if detection.corners is not None and len(detection.corners) > 0:
            corners = detection.corners
            # Ensure corners are in shape (N, 2)
            if corners.ndim == 3:
                corners = corners.reshape(-1, 2)
            corners = corners.astype(np.int32)
            
            for i, corner in enumerate(corners):
                if len(corner) >= 2:
                    cv2.circle(output, tuple(corner[:2]), 5, (0, 255, 0), -1)
                    if detection.ids is not None and i < len(detection.ids):
                        cv2.putText(output, str(detection.ids[i]), 
                                  tuple(corner[:2] + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 255, 0), 1)
        
        # Draw ArUco markers
        if detection.marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(output, detection.marker_corners, 
                                         detection.marker_ids)
        
        # Draw board bounding box
        if detection.corners is not None and len(detection.corners) > 2:
            corners = detection.corners.astype(np.int32)
            hull = cv2.convexHull(corners)
            cv2.polylines(output, [hull], True, (255, 0, 0), 2)
        
        # Draw pose axes (if available)
        if pose is not None and pose.success:
            try:
                cv2.drawFrameAxes(
                    output,
                    self.camera_matrix,
                    self.dist_coeffs,
                    pose.rvec,
                    pose.tvec,
                    length=CHARUCO_DETECTOR_CONFIG.get("axis_length", 0.05),
                    thickness=3
                )
            except Exception as e:
                print(f"Error drawing axes: {e}")
        
        return output


if __name__ == "__main__":
    # Test detection with a simple example
    print("ChArUco Detector test module")
    detector = CharucoDetector()
