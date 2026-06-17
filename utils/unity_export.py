"""
Export detection data to Unity via UDP or JSON files.
For real-time streaming from Pi camera to Unity.
"""

import json
import socket
from typing import Optional
from datetime import datetime
from pathlib import Path


class UnityExporter:
    """Export detection results in Unity-compatible format."""
    
    def __init__(self, udp_host: str = None, udp_port: int = 5000):
        """
        Initialize exporter.
        
        Args:
            udp_host: IP address to send UDP packets to (e.g., '192.168.1.100')
            udp_port: UDP port (default 5000)
        """
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.socket = None
        
        if udp_host:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(f"UDP sender ready: {udp_host}:{udp_port}")
            except Exception as e:
                print(f"Warning: Could not initialize UDP socket: {e}")
                self.socket = None
    
    def build_frame_data(self, frame_num: int, board_detection, board_pose, arm_segmentation) -> dict:
        """Build detection data for a single frame."""
        
        frame_data = {
            "frame": frame_num,
            "timestamp": datetime.now().isoformat(),
            "board": {
                "detected": board_detection.success if board_detection else False,
                "corners": board_detection.corners.tolist() if board_detection and board_detection.corners is not None else None,
                "ids": board_detection.ids.tolist() if board_detection and board_detection.ids is not None else None,
                "pose": None
            },
            "arm": {
                "detected": arm_segmentation.success if arm_segmentation else False,
                "bbox": None
            }
        }
        
        # Add pose if available
        if board_pose and board_pose.success:
            frame_data["board"]["pose"] = {
                "rvec": board_pose.rvec.tolist() if board_pose.rvec is not None else None,
                "tvec": board_pose.tvec.tolist() if board_pose.tvec is not None else None,
                "distance_m": float(board_pose.distance) if board_pose.distance else None
            }
        
        # Add bbox if available
        if arm_segmentation and arm_segmentation.bbox:
            x, y, w, h = arm_segmentation.bbox
            frame_data["arm"]["bbox"] = {
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h)
            }
        
        return {"detections": [frame_data]}
    
    def send_to_unity(self, frame_data: dict) -> bool:
        """Send detection data to Unity via UDP."""
        
        if not self.socket or not self.udp_host:
            return False
        
        try:
            json_str = json.dumps(frame_data)
            self.socket.sendto(json_str.encode('utf-8'), (self.udp_host, self.udp_port))
            return True
        except Exception as e:
            print(f"UDP send error: {e}")
            return False
    
    def save_frame(self, frame_data: dict, output_dir: str = "output/data"):
        """Save frame data to JSON file."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        frame_num = frame_data["detections"][0]["frame"]
        filepath = Path(output_dir) / f"frame_{frame_num:06d}_unity.json"
        
        with open(filepath, 'w') as f:
            json.dump(frame_data, f)
    
    def close(self):
        """Close UDP socket."""
        if self.socket:
            self.socket.close()
