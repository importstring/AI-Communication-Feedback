import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.factors.emotions import Emotion, MultimodalEmotionAnalyzer

class TestEmotion(unittest.TestCase):
    
    def test_init(self):
        """Test the initialization of Emotion class"""
        emotion = Emotion()
        self.assertIsInstance(emotion.emotions, dict)
        self.assertEqual(emotion.emotions, {})
    
    def test_add_emotion(self):
        """Test adding an emotion at a specific timestamp"""
        emotion = Emotion()
        emotion.add_emotion(10.5, 'happy', 0.8)
        
        self.assertIn(10.5, emotion.emotions)
        self.assertEqual(emotion.emotions[10.5], {'emotion': 'happy', 'score': 0.8})
    
    def test_get_emotions_dataframe(self):
        """Test converting emotions to DataFrame"""
        emotion = Emotion()
        emotion.add_emotion(1.0, 'happy', 0.8)
        emotion.add_emotion(2.0, 'neutral', 0.6)
        emotion.add_emotion(3.0, 'sad', 0.4)
        
        df = emotion.get_emotions_dataframe()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df.columns), ['timestamp', 'emotion', 'score'])
        self.assertListEqual(list(df['emotion']), ['happy', 'neutral', 'sad'])

class TestMultimodalEmotionAnalyzer(unittest.TestCase):
    
    @patch('src.factors.emotions.transformers.pipeline')
    def test_init(self, mock_pipeline):
        """Test initialization of MultimodalEmotionAnalyzer"""
        # Setup mocks
        mock_pipeline.return_value = MagicMock()
        
        # Initialize the analyzer
        analyzer = MultimodalEmotionAnalyzer()
        
        # Check that the pipelines are set up
        self.assertTrue(hasattr(analyzer, 'visual_emotion_classifier'))
        self.assertTrue(hasattr(analyzer, 'audio_emotion_classifier'))
        self.assertTrue(hasattr(analyzer, 'text_emotion_classifier'))
        
        # Verify pipeline was called
        self.assertEqual(mock_pipeline.call_count, 3)
    
    @patch('src.factors.emotions.transformers.pipeline')
    @patch('src.factors.emotions.pd.DataFrame')
    def test_analyze_video(self, mock_dataframe, mock_pipeline):
        """Test analyzing a video file"""
        # Setup mocks
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{'label': 'happy', 'score': 0.9}]
        mock_pipeline.return_value = mock_classifier
        
        # Initialize the analyzer
        analyzer = MultimodalEmotionAnalyzer()
        
        # Set up patchers for extract methods
        with patch.object(analyzer, '_extract_frames') as mock_extract_frames, \
             patch.object(analyzer, '_extract_audio') as mock_extract_audio, \
             patch.object(analyzer, '_extract_text') as mock_extract_text, \
             patch.object(analyzer, '_combine_emotions') as mock_combine:
            
            # Setup mock returns
            mock_extract_frames.return_value = [(0.5, np.zeros((224, 224, 3)))]
            mock_extract_audio.return_value = [(1.0, "audio_segment")]
            mock_extract_text.return_value = [(1.5, "Hello world")]
            mock_combine.return_value = pd.DataFrame()
            
            # Call the method
            result = analyzer.analyze_video("test_video.mp4")
            
            # Assertions
            mock_extract_frames.assert_called_once_with("test_video.mp4")
            mock_extract_audio.assert_called_once_with("test_video.mp4")
            mock_extract_text.assert_called_once_with("test_video.mp4")
            mock_combine.assert_called_once()
    
    @patch('src.factors.emotions.transformers.pipeline')
    @patch('src.factors.emotions.cv2.VideoCapture')
    def test_extract_frames(self, mock_video_capture, mock_pipeline):
        """Test frame extraction"""
        # Setup mocks
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.get.return_value = 30.0  # 30 fps
        
        # Mock read() to return 2 frames then False
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        
        # Initialize the analyzer
        analyzer = MultimodalEmotionAnalyzer()
        
        # Call the method
        frames = analyzer._extract_frames("test_video.mp4")
        
        # Assertions
        self.assertEqual(len(frames), 2)
        for timestamp, frame in frames:
            self.assertIsInstance(timestamp, float)
            self.assertIsInstance(frame, np.ndarray)

if __name__ == "__main__":
    unittest.main() 