"""
Arm segmentation module with flexible backends.
Supports depth-based and ML-based arm detection.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from utils.config import ARM_DETECTION_CONFIG

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class ArmSegmentation:
    """Results of arm segmentation."""
    success: bool
    mask: Optional[np.ndarray] = None      # Binary mask (H, W)
    contours: Optional[list] = None        # Detected contours
    bbox: Optional[Tuple] = None           # Bounding box (x, y, w, h)


class ArmSegmenter:
    """Arm segmentation with multiple backend strategies."""
    
    def __init__(self, use_depth: bool = False, use_ml: bool = True):
        """
        Initialize arm segmenter.
        
        Args:
            use_depth: Use depth-based segmentation if available
            use_ml: Use MediaPipe pose detection if available
        """
        self.use_depth = use_depth
        self.use_ml = use_ml and MEDIAPIPE_AVAILABLE
        
        # Initialize MediaPipe if available
        if self.use_ml:
            mp_holistic = mp.solutions.holistic
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_holistic = mp_holistic
            print("MediaPipe Holistic initialized")
        else:
            print("WARNING: MediaPipe not available. Using HSV skin detection only.")
        
        # Morphological kernel for cleanup
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (ARM_DETECTION_CONFIG["morphology_kernel_size"],
             ARM_DETECTION_CONFIG["morphology_kernel_size"])
        )
    
    def segment_arm_depth(self, frame: np.ndarray, 
                         depth_frame: np.ndarray) -> ArmSegmentation:
        """
        Segment arm using depth information.
        
        Args:
            frame: RGB frame (BGR format)
            depth_frame: Depth frame (16-bit, mm)
        
        Returns:
            ArmSegmentation with arm mask and contours
        """
        
        if depth_frame is None or depth_frame.size == 0:
            return ArmSegmentation(success=False)
        
        try:
            # Ensure depth_frame is single-channel uint16
            if len(depth_frame.shape) == 3:
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            
            depth_frame = depth_frame.astype(np.uint16)
            # Depth threshold to isolate arm
            depth_min = ARM_DETECTION_CONFIG["depth_min_mm"]
            depth_max = ARM_DETECTION_CONFIG["depth_max_mm"]
            
            # Create depth mask
            depth_mask = cv2.inRange(depth_frame, depth_min, depth_max)
            
            # Morphological operations to clean up
            for _ in range(ARM_DETECTION_CONFIG["morphology_iterations"]):
                depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, self.kernel)
                depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = ARM_DETECTION_CONFIG["min_contour_area"]
            max_area = ARM_DETECTION_CONFIG["max_contour_area"]
            contours = [c for c in contours 
                       if min_area < cv2.contourArea(c) < max_area]
            
            # Get bounding box of largest contour
            bbox = None
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                bbox = cv2.boundingRect(largest_contour)
            
            return ArmSegmentation(
                success=True,
                mask=depth_mask,
                contours=contours,
                bbox=bbox
            )
        
        except Exception as e:
            print(f"Error in depth-based segmentation: {e}")
            return ArmSegmentation(success=False)
    
    def segment_arm_skin_detection(self, frame: np.ndarray) -> ArmSegmentation:
        """
        Segment arm using HSV skin detection.
        
        Args:
            frame: RGB frame (BGR format)
        
        Returns:
            ArmSegmentation with arm mask
        """
        
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define skin color range (HSV)
            lower_skin = np.array([
                ARM_DETECTION_CONFIG["skin_lower_H"],
                ARM_DETECTION_CONFIG["skin_lower_S"],
                ARM_DETECTION_CONFIG["skin_lower_V"]
            ])
            
            upper_skin = np.array([
                ARM_DETECTION_CONFIG["skin_upper_H"],
                ARM_DETECTION_CONFIG["skin_upper_S"],
                ARM_DETECTION_CONFIG["skin_upper_V"]
            ])
            
            # Create skin mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Morphological operations
            for _ in range(ARM_DETECTION_CONFIG["morphology_iterations"]):
                skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel)
                skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            min_area = ARM_DETECTION_CONFIG["min_contour_area"]
            max_area = ARM_DETECTION_CONFIG["max_contour_area"]
            contours = [c for c in contours
                       if min_area < cv2.contourArea(c) < max_area]
            
            # Get bounding box
            bbox = None
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                bbox = cv2.boundingRect(largest_contour)
            
            return ArmSegmentation(
                success=True,
                mask=skin_mask,
                contours=contours,
                bbox=bbox
            )
        
        except Exception as e:
            print(f"Error in skin detection: {e}")
            return ArmSegmentation(success=False)
    
    def segment_arm_mediapipe(self, frame: np.ndarray) -> ArmSegmentation:
        """
        Segment arm using MediaPipe Holistic pose detection.
        Creates a mask based on upper body landmarks.
        
        Args:
            frame: RGB frame (BGR format, will be converted to RGB)
        
        Returns:
            ArmSegmentation with arm mask
        """
        
        if not self.use_ml or self.holistic is None:
            return ArmSegmentation(success=False)
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run MediaPipe detection
            results = self.holistic.process(rgb_frame)
            
            if results.pose_landmarks is None:
                return ArmSegmentation(success=False)
            
            # Create mask for upper body
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Extract arm landmarks (indices for upper limbs)
            # MediaPipe pose: shoulders (11, 12), elbows (13, 14), wrists (15, 16)
            arm_indices = [11, 12, 13, 14, 15, 16]
            
            try:
                arm_points = []
                for idx in arm_indices:
                    if idx < len(results.pose_landmarks.landmark):
                        lm = results.pose_landmarks.landmark[idx]
                        if lm.visibility > 0.5:  # Only if visible
                            x = int(lm.x * w)
                            y = int(lm.y * h)
                            arm_points.append([x, y])
                
                # Create convex hull from arm points
                if len(arm_points) > 2:
                    arm_points = np.array(arm_points, dtype=np.int32)
                    hull = cv2.convexHull(arm_points)
                    cv2.drawContours(mask, [hull], 0, 255, -1)
                    
                    # Dilate to capture full arm area
                    mask = cv2.dilate(mask, self.kernel, iterations=2)
                    
                    # Find contours from dilated mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    
                    bbox = None
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        bbox = cv2.boundingRect(largest)
                    
                    return ArmSegmentation(
                        success=True,
                        mask=mask,
                        contours=contours,
                        bbox=bbox
                    )
            
            except Exception as e:
                print(f"Error processing MediaPipe landmarks: {e}")
            
            return ArmSegmentation(success=False)
        
        except Exception as e:
            print(f"Error in MediaPipe segmentation: {e}")
            return ArmSegmentation(success=False)
    
    def segment_arm(self, frame: np.ndarray, 
                   depth_frame: Optional[np.ndarray] = None) -> ArmSegmentation:
        """
        Segment arm with automatic backend selection.
        Priority: depth-based → MediaPipe → skin detection
        
        Args:
            frame: RGB frame (BGR format)
            depth_frame: Optional depth frame
        
        Returns:
            ArmSegmentation result
        """
        
        # Try depth-based first (if available and enabled)
        if self.use_depth and depth_frame is not None:
            result = self.segment_arm_depth(frame, depth_frame)
            if result.success:
                return result
        
        # Try MediaPipe second (if available)
        if self.use_ml and MEDIAPIPE_AVAILABLE:
            result = self.segment_arm_mediapipe(frame)
            if result.success:
                return result
        
        # Fallback to skin detection
        return self.segment_arm_skin_detection(frame)
    
    def draw_segmentation(self, frame: np.ndarray, 
                         segmentation: ArmSegmentation,
                         alpha: float = 0.3) -> np.ndarray:
        """
        Draw segmentation results on frame.
        
        Args:
            frame: Input frame
            segmentation: ArmSegmentation result
            alpha: Transparency of mask overlay (0-1)
        
        Returns:
            Annotated frame
        """
        
        output = frame.copy()
        
        if not segmentation.success or segmentation.mask is None:
            return output
        
        # Create colored mask
        mask_colored = np.zeros_like(frame)
        mask_colored[segmentation.mask > 0] = [0, 255, 0]  # Green
        
        # Blend with original frame
        output = cv2.addWeighted(output, 1 - alpha, mask_colored, alpha, 0)
        
        # Draw contours
        if segmentation.contours:
            cv2.drawContours(output, segmentation.contours, -1, (0, 255, 0), 2)
        
        # Draw bounding box
        if segmentation.bbox:
            x, y, w, h = segmentation.bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(output, f"Area: {w*h}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return output


if __name__ == "__main__":
    print("Arm Segmenter test module")
    segmenter = ArmSegmenter()
