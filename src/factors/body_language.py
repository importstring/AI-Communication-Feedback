import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import datetime
from ..parquet_management import ReadWrite

class JointMap():
    """
    Analyzes body language using pose estimation and movement patterns.
    """
    def __init__(self):
        self.states = {} # Frame: {Joint: (x, y, z)}
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Initialize the PoseLandmarker options
        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path='pose_landmarker.task'),
            running_mode=self.VisionRunningMode.Video
        )
        
        self.pose_landmarker = self.PoseLandmarker.create_from_options(self.options)
        

    def map_recording(self, video_path):
        """
        Maps the joints in a video file and stores the results in parquet format
        """
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._process_frame(rgb_frame, frame_count, fps)
            frame_count += 1

        cap.release()
    
        df = self._convert_to_dataframe()
        video_path = datetime.now().strftime("%Y-%m-%d")
        ReadWrite().write_parquet(
            data=df, file=f"{video_path}.parquet"
            )
        return df

    def _convert_to_dataframe(self):
        columns = pd.MultiIndex.from_product(
            [range(33), ['x', 'y', 'z']],
            names=['joint', 'coord']
        )
        
        data = np.zeros((len(self.states), 33*3))
        for frame, landmarks in self.states.items():
            data[frame] = np.array([
                coord for j in sorted(landmarks) 
                for coord in landmarks[j]
            ])
        
        return pd.DataFrame(data, columns=columns)

    def __del__(self):
        """
        Properly clean up MediaPipe resources
        """
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()


    def _process_frame(self, frame, frame_number, fps):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        height, width, _ = frame.shape

        self.states[frame_number] = {
            i: (lmk.x * width, lmk.y * height, lmk.z * width) 
            for i, lmk in enumerate(result.pose_landmarks[0])
        }

        timestamp_ms = int((frame_number / fps) * 1000)

        result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        self.states[frame_number] = {
            i: (lmk.x, lmk.y, lmk.z) 
            for i, lmk in enumerate(result.pose_landmarks[0])
        }

    def load_video(self, video_path):
        """
        Loads a video file and initializes the pose landmarker.
        """
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.pose_landmarker = self.PoseLandmarker.create_from_options(self.options)
