import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from .helper import save_factor_data

class JointMap:
    """Analyzes body language using pose estimation and movement patterns."""
    
    def __init__(self, model_path: str = 'pose_landmarker.task', timestamp: str = None):
        self.states = {}  # Frame: {Joint: (x, y, z)}
        self.model_path = Path(model_path).absolute()
        
        # MediaPipe configuration
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.timestamp = timestamp
        self.video_path = self.get_video_path()

    def map_recording(self, video_path: str) -> pd.DataFrame:
        """Process video and store joint positions in parquet format."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        # Use context manager for proper resource handling
        with self._create_landmarker() as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._process_frame(landmarker, rgb_frame, frame_count, fps)
                frame_count += 1

        cap.release()
        self._save_results(video_path)

    def _create_landmarker(self):
        """Initialize pose landmarker with proper configuration"""
        options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=self.VisionRunningMode.VIDEO,
            num_poses=1 
        )
        return self.PoseLandmarker.create_from_options(options)

    def _process_frame(self, landmarker, frame: np.ndarray, 
                      frame_number: int, fps: float):
        """Process individual frame and store landmark data"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int((frame_number / fps) * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        self._store_landmarks(result, frame, frame_number)

    def _store_landmarks(self, result, frame: np.ndarray, frame_number: int):
        """Extract and normalize landmark coordinates"""
        height, width, _ = frame.shape
        self.states[frame_number] = {}
        
        if result.pose_landmarks:
            for idx, lmk in enumerate(result.pose_landmarks[0]):
                self.states[frame_number][idx] = (
                    lmk.x * width, 
                    lmk.y * height,
                    lmk.z * width
                )

    def _save_results(self, video_path: str) -> pd.DataFrame:
        """Convert results to DataFrame and save as parquet"""
        df = pd.DataFrame(
            {frame: self._flatten_landmarks(data) 
             for frame, data in self.states.items()}
        ).T
        
        save_factor_data(df, 'body_language', self.timestamp)

    def _flatten_landmarks(self, landmarks: dict) -> dict:
        """Create multi-index columns for joint coordinates"""
        return {
            (joint, coord): value
            for joint, coords in landmarks.items()
            for coord, value in zip(['x', 'y', 'z'], coords)
        }

    def get_video_path(self):
        current_file = Path(__file__).resolve()

        project_root = current_file.parents[2]

        data_dir = project_root / 'data' / 'recordings' / 'video'
        src_dir = project_root / 'src' / 'factors'

        return str(data_dir) + self.timestamp        