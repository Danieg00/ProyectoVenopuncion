"""
Data export module for saving detection results to JSON and CSV.
"""

import json
import csv
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
from models.charuco_detector import BoardDetection, BoardPose
from models.arm_segmenter import ArmSegmentation


class DetectionDataExporter:
    """Export detection results to JSON/CSV files."""
    
    def __init__(self, output_dir: str = "output/data"):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to save export files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_file = self.output_dir / f"detections_{timestamp}.json"
        self.csv_file = self.output_dir / f"detections_{timestamp}.csv"
        
        self.frame_count = 0
        self.detections = []
        
        # Initialize CSV file with headers
        self.csv_writer = None
        self.csv_file_handle = None
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file_handle,
            fieldnames=[
                'frame_number',
                'timestamp',
                'board_detected',
                'board_distance_m',
                'board_rvec_x',
                'board_rvec_y',
                'board_rvec_z',
                'board_tvec_x',
                'board_tvec_y',
                'board_tvec_z',
                'arm_detected',
                'arm_bbox_x',
                'arm_bbox_y',
                'arm_bbox_w',
                'arm_bbox_h'
            ]
        )
        self.csv_writer.writeheader()
    
    def export_frame(self, board_detection: Optional[BoardDetection] = None,
                    board_pose: Optional[BoardPose] = None,
                    arm_segmentation: Optional[ArmSegmentation] = None):
        """
        Export detection results for a single frame.
        
        Args:
            board_detection: ChArUco detection result
            board_pose: Board pose estimation result
            arm_segmentation: Arm segmentation result
        """
        
        frame_data = {
            'frame': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'board': self._serialize_detection(board_detection, board_pose),
            'arm': self._serialize_segmentation(arm_segmentation)
        }
        
        self.detections.append(frame_data)
        
        # Write to CSV
        csv_row = {
            'frame_number': self.frame_count,
            'timestamp': frame_data['timestamp'],
            'board_detected': (board_detection and board_detection.success) or False,
            'board_distance_m': (board_pose and board_pose.distance) or None,
            'board_rvec_x': (board_pose and board_pose.rvec[0]) or None if board_pose and board_pose.rvec is not None else None,
            'board_rvec_y': (board_pose and board_pose.rvec[1]) or None if board_pose and board_pose.rvec is not None else None,
            'board_rvec_z': (board_pose and board_pose.rvec[2]) or None if board_pose and board_pose.rvec is not None else None,
            'board_tvec_x': (board_pose and board_pose.tvec[0]) or None if board_pose and board_pose.tvec is not None else None,
            'board_tvec_y': (board_pose and board_pose.tvec[1]) or None if board_pose and board_pose.tvec is not None else None,
            'board_tvec_z': (board_pose and board_pose.tvec[2]) or None if board_pose and board_pose.tvec is not None else None,
            'arm_detected': (arm_segmentation and arm_segmentation.success) or False,
            'arm_bbox_x': (arm_segmentation and arm_segmentation.bbox[0]) if arm_segmentation and arm_segmentation.bbox else None,
            'arm_bbox_y': (arm_segmentation and arm_segmentation.bbox[1]) if arm_segmentation and arm_segmentation.bbox else None,
            'arm_bbox_w': (arm_segmentation and arm_segmentation.bbox[2]) if arm_segmentation and arm_segmentation.bbox else None,
            'arm_bbox_h': (arm_segmentation and arm_segmentation.bbox[3]) if arm_segmentation and arm_segmentation.bbox else None,
        }
        self.csv_writer.writerow(csv_row)
        self.csv_file_handle.flush()
        
        self.frame_count += 1
    
    def _serialize_detection(self, detection: Optional[BoardDetection],
                            pose: Optional[BoardPose]) -> Dict[str, Any]:
        """Serialize board detection and pose to dictionary."""
        
        if not detection or not detection.success:
            return {
                'detected': False,
                'corners': None,
                'ids': None,
                'pose': None
            }
        
        corners = None
        if detection.corners is not None:
            corners = detection.corners.tolist()
        
        ids = None
        if detection.ids is not None:
            ids = detection.ids.tolist()
        
        pose_data = None
        if pose and pose.success:
            pose_data = {
                'rvec': pose.rvec.tolist(),
                'tvec': pose.tvec.tolist(),
                'distance_m': float(pose.distance) if pose.distance else None
            }
        
        return {
            'detected': True,
            'corners': corners,
            'ids': ids,
            'pose': pose_data
        }
    
    def _serialize_segmentation(self, segmentation: Optional[ArmSegmentation]) -> Dict[str, Any]:
        """Serialize arm segmentation to dictionary."""
        
        if not segmentation or not segmentation.success:
            return {
                'detected': False,
                'bbox': None
            }
        
        bbox = None
        if segmentation.bbox:
            x, y, w, h = segmentation.bbox
            bbox = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        
        return {
            'detected': True,
            'bbox': bbox
        }
    
    def save_json(self):
        """Save all detections to JSON file."""
        with open(self.json_file, 'w') as f:
            json.dump(self.detections, f, indent=2)
        print(f"JSON exported: {self.json_file}")
    
    def close(self):
        """Close export files."""
        if self.csv_file_handle:
            self.csv_file_handle.close()
        self.save_json()
        print(f"CSV exported: {self.csv_file}")
    
    def __del__(self):
        """Ensure files are closed on deletion."""
        self.close()


if __name__ == "__main__":
    print("Data export test module")
    exporter = DetectionDataExporter()
