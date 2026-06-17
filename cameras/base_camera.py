"""
Base camera class defining interface for all camera sources.
Ensures consistent API across RGB, RGB-D, and other camera types.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class BaseCamera(ABC):
    """Abstract base class for camera implementations."""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720), 
                 fps: int = 30):
        """
        Initialize camera.
        
        Args:
            camera_id: Camera index or identifier
            resolution: (width, height) in pixels
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.is_open = False
        
    @abstractmethod
    def open(self) -> bool:
        """Open camera connection. Return True if successful."""
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read frame from camera.
        
        Returns:
            (success, rgb_frame, depth_frame)
            - success: True if frame was read successfully
            - rgb_frame: np.ndarray (H, W, 3) in BGR format
            - depth_frame: np.ndarray (H, W) in mm, or None if not available
        """
        pass
    
    @abstractmethod
    def release(self):
        """Release camera resources."""
        pass
    
    @abstractmethod
    def has_depth(self) -> bool:
        """Return True if camera provides depth frames."""
        pass
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution (width, height)."""
        return self.resolution
    
    def get_fps(self) -> int:
        """Get target fps."""
        return self.fps
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
