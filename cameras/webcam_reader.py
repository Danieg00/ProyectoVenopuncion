"""
Webcam reader implementation using OpenCV.
Fallback camera source for systems without RGB-D cameras.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .base_camera import BaseCamera


class WebcamReader(BaseCamera):
    """OpenCV VideoCapture-based webcam reader."""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720), 
                 fps: int = 30):
        """
        Initialize webcam reader.
        
        Args:
            camera_id: Webcam index (0 for default, 1 for secondary, etc.)
            resolution: Target (width, height)
            fps: Target frames per second
        """
        super().__init__(camera_id, resolution, fps)
        self.cap = None
        
    def open(self) -> bool:
        """Open webcam connection."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"ERROR: Could not open webcam {self.camera_id}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_open = True
            
            # Get actual resolution (might differ from requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = (actual_width, actual_height)
            
            print(f"Webcam {self.camera_id} opened: {self.resolution} @ {self.fps} FPS")
            return True
            
        except Exception as e:
            print(f"ERROR opening webcam: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read frame from webcam.
        
        Returns:
            (success, rgb_frame, depth_frame)
            depth_frame is always None for webcam
        """
        if not self.is_open or self.cap is None:
            return False, None, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                return False, None, None
            
            # Ensure correct resolution (resize if needed)
            if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                frame = cv2.resize(frame, self.resolution)
            
            return True, frame, None
            
        except Exception as e:
            print(f"ERROR reading frame: {e}")
            return False, None, None
    
    def release(self):
        """Release webcam resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            print(f"Webcam {self.camera_id} released")
    
    def has_depth(self) -> bool:
        """Webcam does not provide depth frames."""
        return False
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()
