"""
Visualization and frame annotation utilities.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from models.charuco_detector import BoardDetection, BoardPose
from models.arm_segmenter import ArmSegmentation


class FrameAnnotator:
    """Annotates frames with detection results."""
    
    def __init__(self, show_fps: bool = True, show_info: bool = True):
        """
        Initialize annotator.
        
        Args:
            show_fps: Display FPS counter
            show_info: Display detection info text
        """
        self.show_fps = show_fps
        self.show_info = show_info
        self.fps_values = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_color = (0, 255, 0)
        self.text_thickness = 1
    
    def update_fps(self, fps: float):
        """Update FPS counter."""
        self.fps_values.append(fps)
        if len(self.fps_values) > 30:
            self.fps_values.pop(0)
    
    def get_average_fps(self) -> float:
        """Get average FPS."""
        if not self.fps_values:
            return 0.0
        return np.mean(self.fps_values)
    
    def annotate_frame(self, frame: np.ndarray,
                      board_detection: Optional[BoardDetection] = None,
                      board_pose: Optional[BoardPose] = None,
                      arm_segmentation: Optional[ArmSegmentation] = None,
                      fps: Optional[float] = None) -> np.ndarray:
        """
        Annotate frame with all detection results.
        
        Args:
            frame: Input frame
            board_detection: ChArUco detection results
            board_pose: Board pose estimation results
            arm_segmentation: Arm segmentation results
            fps: Current FPS
        
        Returns:
            Annotated frame
        """
        
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Position for text display
        text_y = 30
        text_step = 25
        
        # Draw board detection results
        if board_detection and board_detection.success:
            # Draw corners
            if board_detection.corners is not None:
                corners = board_detection.corners.astype(np.int32)
                for i, corner in enumerate(corners):
                    cv2.circle(output, tuple(corner), 4, (0, 255, 0), -1)
            
            # Draw convex hull
            if board_detection.corners is not None and len(board_detection.corners) > 2:
                corners = board_detection.corners.astype(np.int32)
                hull = cv2.convexHull(corners)
                cv2.polylines(output, [hull], True, (255, 0, 0), 2)
            
            if self.show_info:
                cv2.putText(output, "Board: DETECTED", (10, text_y),
                           self.font, self.font_scale, (0, 255, 0), self.text_thickness)
                text_y += text_step
        else:
            if self.show_info:
                cv2.putText(output, "Board: NOT DETECTED", (10, text_y),
                           self.font, self.font_scale, (0, 0, 255), self.text_thickness)
                text_y += text_step
        
        # Draw pose information
        if board_pose and board_pose.success:
            if self.show_info:
                distance_cm = board_pose.distance * 100 if board_pose.distance else 0
                cv2.putText(output, f"Distance: {distance_cm:.1f}cm", (10, text_y),
                           self.font, self.font_scale, self.font_color, self.text_thickness)
                text_y += text_step
        
        # Draw arm segmentation
        if arm_segmentation and arm_segmentation.success:
            # Draw bounding box
            if arm_segmentation.bbox:
                x, y, w_bbox, h_bbox = arm_segmentation.bbox
                cv2.rectangle(output, (x, y), (x + w_bbox, y + h_bbox),
                            (0, 255, 255), 2)
            
            if self.show_info:
                cv2.putText(output, "Arm: DETECTED", (10, text_y),
                           self.font, self.font_scale, (0, 255, 0), self.text_thickness)
                text_y += text_step
        else:
            if self.show_info:
                cv2.putText(output, "Arm: NOT DETECTED", (10, text_y),
                           self.font, self.font_scale, (0, 0, 255), self.text_thickness)
                text_y += text_step
        
        # Draw FPS counter
        if self.show_fps and fps is not None:
            self.update_fps(fps)
            avg_fps = self.get_average_fps()
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(output, fps_text, (w - 150, 30),
                       self.font, 0.7, (0, 255, 0), 2)
        
        return output


class VideoWriter:
    """Write output frames to video file."""
    
    def __init__(self, output_path: str, fps: int = 30, 
                 frame_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            frame_size: (width, height) of frames
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not self.writer.isOpened():
            print(f"ERROR: Could not open video writer for {output_path}")
        else:
            print(f"Video writer initialized: {output_path}")
    
    def write(self, frame: np.ndarray) -> bool:
        """
        Write frame to video.
        
        Args:
            frame: Frame to write (should match initialized size)
        
        Returns:
            True if successful
        """
        if not self.writer.isOpened():
            return False
        
        # Resize if necessary
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
        return True
    
    def release(self):
        """Release video writer and finalize file."""
        if self.writer.isOpened():
            self.writer.release()
            print(f"Video saved: {self.output_path}")


if __name__ == "__main__":
    print("Visualization utilities test module")
    annotator = FrameAnnotator()
