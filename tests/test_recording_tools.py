import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import tempfile
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recording_tools import RecordVideo, record

class TestRecordVideo(unittest.TestCase):
    
    @patch('src.recording_tools.cv2.VideoCapture')
    @patch('src.recording_tools.cv2.VideoWriter')
    def test_init(self, mock_writer, mock_capture):
        # Mock camera open success
        mock_capture.return_value.isOpened.return_value = True
        
        # Initialize the class
        recorder = RecordVideo()
        
        # Assert attributes were properly set
        self.assertEqual(recorder.frame_rate, 60)
        self.assertEqual(recorder.chunk, 1024)
        self.assertEqual(recorder.channels, 1)
        self.assertEqual(recorder.sampling_rate, 44100)
        self.assertEqual(recorder.base_dir, '../data/recordings')
        self.assertFalse(recorder.recording)
        
        # Check if VideoCapture was called
        mock_capture.assert_called_once_with(0)
        
    @patch('src.recording_tools.cv2.VideoCapture')
    @patch('src.recording_tools.cv2.VideoWriter')
    def test_update_path(self, mock_writer, mock_capture):
        # Mock camera open success
        mock_capture.return_value.isOpened.return_value = True
        
        # Initialize the class
        recorder = RecordVideo()
        
        # Test with empty parameters
        result = recorder.update_path()
        self.assertEqual(result, '../data/recordings')
        
        # Test with only sub_dir
        result = recorder.update_path('/video/')
        self.assertEqual(result, '../data/recordings/video/')
        
        # Test with both parameters
        result = recorder.update_path('/audio/', 'test.wav')
        self.assertEqual(result, '../data/recordings/audio/test.wav')
    
    @patch('src.recording_tools.cv2.VideoCapture')
    @patch('src.recording_tools.cv2.VideoWriter')
    @patch('src.recording_tools.pyaudio.PyAudio')
    @patch('src.recording_tools.wave.open', new_callable=mock_open)
    def test_record_audio(self, mock_wave, mock_pyaudio, mock_writer, mock_capture):
        # Mock camera open success
        mock_capture.return_value.isOpened.return_value = True
        
        # Setup PyAudio mock
        mock_audio = MagicMock()
        mock_pyaudio.return_value = mock_audio
        mock_stream = MagicMock()
        mock_audio.open.return_value = mock_stream
        mock_stream.read.return_value = b'test_audio_data'
        mock_audio.get_sample_size.return_value = 2
        
        # Initialize the class
        recorder = RecordVideo()
        
        # Set recording to True, then False after two iterations
        recorder.recording = True
        
        def side_effect(*args, **kwargs):
            recorder.recording = False
            return b'test_audio_data'
            
        mock_stream.read.side_effect = side_effect
        
        # Call record_audio
        recorder.record_audio()
        
        # Check if audio operations were performed
        mock_audio.open.assert_called_once()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_audio.terminate.assert_called_once()
        mock_wave.assert_called_once()

    @patch('src.recording_tools.RecordVideo')
    @patch('src.recording_tools.threading.Thread')
    @patch('src.recording_tools.cv2.waitKey')
    @patch('src.recording_tools.cv2.imshow')
    def test_record_function(self, mock_imshow, mock_waitkey, mock_thread, mock_recordvideo):
        # Setup mocks
        mock_instance = MagicMock()
        mock_recordvideo.return_value = mock_instance
        mock_instance.cap.read.return_value = (True, "frame_data")
        
        # Set up waitKey to return 'q' press on second call
        mock_waitkey.side_effect = [0, ord('q') & 0xFF]
        
        # Call record function
        record("test_dir", "test_file")
        
        # Check if RecordVideo was initialized
        mock_recordvideo.assert_called_once()
        
        # Check if update_path was called
        mock_instance.update_path.assert_called_with("test_dir", "test_file")
        
        # Check if thread was started
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
        
        # Check if recording was set to True
        self.assertTrue(mock_instance.recording)
        
        # Check if resources were cleaned up
        mock_instance.cap.release.assert_called_once()
        mock_instance.video_writer.release.assert_called_once()

if __name__ == "__main__":
    unittest.main() 