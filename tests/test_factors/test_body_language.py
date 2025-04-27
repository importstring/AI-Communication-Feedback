import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.factors.body_language import JointMap

class TestJointMap(unittest.TestCase):
    
    @patch('src.factors.body_language.mp.tasks.vision.PoseLandmarker.create_from_options')
    def setUp(self, mock_create_from_options):
        self.joint_map = JointMap()
        
    def test_init(self):
        """Test the initialization of JointMap"""
        self.assertEqual(self.joint_map.states, {})
        self.assertTrue(hasattr(self.joint_map, 'BaseOptions'))
        self.assertTrue(hasattr(self.joint_map, 'PoseLandmarker'))
        self.assertTrue(hasattr(self.joint_map, 'PoseLandmarkerOptions'))
        self.assertTrue(hasattr(self.joint_map, 'VisionRunningMode'))
        self.assertTrue(hasattr(self.joint_map, 'pose_landmarker'))
        
    @patch('src.factors.body_language.cv2.VideoCapture')
    @patch('src.factors.body_language.ReadWrite')
    def test_map_recording(self, mock_readwrite, mock_video_capture):
        """Test the map_recording method"""
        # Setup mocks
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.get.return_value = 30.0  # 30 fps
        
        # Mock read() to return one frame then False
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        
        # Mock _process_frame to add data to states
        self.joint_map._process_frame = MagicMock()
        self.joint_map._convert_to_dataframe = MagicMock(return_value=pd.DataFrame())
        
        # Call the method
        result = self.joint_map.map_recording("test_video.mp4")
        
        # Assertions
        mock_video_capture.assert_called_once_with("test_video.mp4")
        self.joint_map._process_frame.assert_called_once()
        self.joint_map._convert_to_dataframe.assert_called_once()
        mock_cap.release.assert_called_once()
        
    def test_convert_to_dataframe(self):
        """Test converting joint states to DataFrame"""
        # Setup test data
        self.joint_map.states = {
            0: {
                0: (0.1, 0.2, 0.3),
                1: (0.4, 0.5, 0.6)
            },
            1: {
                0: (0.7, 0.8, 0.9),
                1: (1.0, 1.1, 1.2)
            }
        }
        
        # When not all 33 joints are present, pad with zeros
        with patch('src.factors.body_language.np.zeros') as mock_zeros:
            # Mock the zeros array
            mock_zeros.return_value = np.zeros((2, 99))
            self.joint_map._convert_to_dataframe()
            mock_zeros.assert_called_once_with((2, 33*3))
    
    @patch('src.factors.body_language.cv2.VideoCapture')
    def test_load_video(self, mock_video_capture):
        """Test loading a video"""
        # Setup mocks
        mock_video_capture.return_value = MagicMock()
        
        # Create fresh instance with create_from_options already mocked in setUp
        with patch('src.factors.body_language.mp.tasks.vision.PoseLandmarker.create_from_options') as mock_create:
            joint_map = JointMap()
            joint_map.pose_landmarker = None  # Reset to test creation
            mock_create.return_value = MagicMock()
            
            # Call the method
            joint_map.load_video("test_path.mp4")
            
            # Assertions
            self.assertEqual(joint_map.video_path, "test_path.mp4")
            mock_video_capture.assert_called_once_with("test_path.mp4")
            mock_create.assert_called_once()

if __name__ == "__main__":
    unittest.main() 