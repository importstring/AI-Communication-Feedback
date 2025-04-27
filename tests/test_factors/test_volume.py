import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import os
import sys
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.factors.volume import VolumeVarience

class TestVolumeVarience(unittest.TestCase):
    
    def setUp(self):
        self.volume = VolumeVarience()
    
    def test_init(self):
        """Test the initialization of VolumeVarience"""
        self.assertEqual(self.volume.frame_size, 2048)
        self.assertEqual(self.volume.hop_length, 512)
        self.assertEqual(self.volume.sampling_rate, 22050)
        self.assertIsNone(self.volume.audio_data)
        self.assertIsNone(self.volume.volume_stats)
    
    @patch('src.factors.volume.librosa.load')
    @patch('src.factors.volume.librosa.feature.rms')
    def test_analyze_audio(self, mock_rms, mock_load):
        """Test analyzing audio file"""
        # Setup mocks
        mock_audio = np.random.rand(22050 * 5)  # 5 seconds of audio
        mock_sr = 22050
        mock_load.return_value = (mock_audio, mock_sr)
        
        # Mock RMS values (volume envelopes)
        mock_rms_values = np.random.rand(1, 100)  # 100 frames
        mock_rms.return_value = mock_rms_values
        
        # Call the method
        self.volume.analyze_audio("test_audio.wav")
        
        # Assertions
        mock_load.assert_called_once_with("test_audio.wav", sr=22050)
        mock_rms.assert_called_once()
        self.assertIsNotNone(self.volume.audio_data)
        self.assertIsNotNone(self.volume.volume_envelope)
    
    @patch('src.factors.volume.librosa.load')
    @patch('src.factors.volume.librosa.feature.rms')
    def test_compute_stats(self, mock_rms, mock_load):
        """Test computing volume statistics"""
        # Setup mocks
        mock_audio = np.random.rand(22050 * 5)  # 5 seconds of audio
        mock_sr = 22050
        mock_load.return_value = (mock_audio, mock_sr)
        
        # Create an envelope with known patterns
        mock_envelope = np.array([[0.1, 0.5, 0.9, 0.5, 0.1]])
        mock_rms.return_value = mock_envelope
        
        # Call analyze_audio first (to set up the envelope)
        self.volume.analyze_audio("test_audio.wav")
        
        # Then call compute_stats
        stats = self.volume.compute_stats()
        
        # Assertions
        self.assertIsNotNone(stats)
        self.assertIn('mean_volume', stats)
        self.assertIn('std_dev', stats)
        self.assertIn('max_volume', stats)
        self.assertIn('min_volume', stats)
        self.assertIn('dynamic_range', stats)
        
        # Check specific values
        self.assertAlmostEqual(stats['mean_volume'], np.mean(mock_envelope))
        self.assertAlmostEqual(stats['max_volume'], np.max(mock_envelope))
        self.assertAlmostEqual(stats['min_volume'], np.min(mock_envelope))
        self.assertAlmostEqual(stats['dynamic_range'], np.max(mock_envelope) - np.min(mock_envelope))
    
    @patch('src.factors.volume.librosa.load')
    @patch('src.factors.volume.librosa.feature.rms')
    @patch('src.factors.volume.plt')
    def test_plot_volume_envelope(self, mock_plt, mock_rms, mock_load):
        """Test plotting volume envelope"""
        # Setup mocks
        mock_audio = np.random.rand(22050 * 5)  # 5 seconds of audio
        mock_sr = 22050
        mock_load.return_value = (mock_audio, mock_sr)
        
        # Create an envelope
        mock_envelope = np.array([[0.1, 0.5, 0.9, 0.5, 0.1]])
        mock_rms.return_value = mock_envelope
        
        # Call analyze_audio first
        self.volume.analyze_audio("test_audio.wav")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            # Call the plot method
            self.volume.plot_volume_envelope(tmp.name)
            
            # Assertions
            mock_plt.figure.assert_called_once()
            mock_plt.plot.assert_called_once()
            mock_plt.savefig.assert_called_once_with(tmp.name)
            mock_plt.close.assert_called_once()
    
    @patch('src.factors.volume.librosa.load')
    @patch('src.factors.volume.librosa.feature.rms')
    @patch('src.factors.volume.pd.DataFrame')
    def test_get_dataframe(self, mock_df, mock_rms, mock_load):
        """Test getting DataFrame representation"""
        # Setup mocks
        mock_audio = np.random.rand(22050 * 5)  # 5 seconds of audio
        mock_sr = 22050
        mock_load.return_value = (mock_audio, mock_sr)
        
        # Create an envelope
        mock_envelope = np.array([[0.1, 0.5, 0.9, 0.5, 0.1]])
        mock_rms.return_value = mock_envelope
        
        # Mock DataFrame creation
        mock_df_instance = MagicMock()
        mock_df.return_value = mock_df_instance
        
        # Call analyze_audio first
        self.volume.analyze_audio("test_audio.wav")
        
        # Call the method
        df = self.volume.get_dataframe()
        
        # Assertions
        mock_df.assert_called_once()
        self.assertEqual(df, mock_df_instance)

if __name__ == "__main__":
    unittest.main() 