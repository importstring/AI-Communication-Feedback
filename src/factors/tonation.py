import librosa
import numpy as np
import pandas as pd
from disvoice.prosody import Prosody
from ..parquet_management import ReadWrite  # Reuse existing Parquet handler

class TonalAnalyzer:
    """Analyzes speech prosody features for machine learning applications.
    
    Captures pitch, intensity, and spectral characteristics at 10ms intervals
    for temporal alignment with video analysis pipelines.
    """
    
    def __init__(self, sr=16000, frame_length=2048, hop_length=512):
        """Initialize audio processing parameters.
        
        Args:
            sr: Sampling rate matching video analysis temporal resolution
            frame_length: Power-of-two for optimal FFT performance
            hop_length: 10ms frames for alignment with 30fps video (512/16000 = 0.032s)
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.prosody = Prosody()
        self.rw = ReadWrite()  # Reuse Parquet handler from existing pipeline

    def extract_features(self, audio_path: str) -> pd.DataFrame:
        """Extract prosodic features aligned with video analysis timeline.
        
        Process flow:
        1. Load audio with temporal resolution matching video frames
        2. Extract pitch contours using YIN algorithm (optimal for speech)
        3. Calculate intensity RMS values
        4. Combine features with timestamps for Parquet storage
        
        Returns:
            DataFrame with columns [timestamp, f0, intensity, ...prosody_features]
        """
        try:
            # Load audio with librosa using parameters from video pipeline
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Extract pitch using YIN algorithm (robust to speech aperiodicity)
            f0 = librosa.yin(y, fmin=50, fmax=2000,
                           frame_length=self.frame_length,
                           hop_length=self.hop_length)
            
            # Calculate intensity (RMS) with same windowing as pitch
            intensity = librosa.feature.rms(y=y, 
                                          frame_length=self.frame_length,
                                          hop_length=self.hop_length)

            # Get prosody features optimized for ML input
            static_features = self.prosody.prosody_static(audio_path)
            dynamic_features = self.prosody.prosody_dynamic(audio_path)

            return self._create_feature_dataframe(f0, intensity[0], static_features, dynamic_features, sr)
            
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            raise

    def _create_feature_dataframe(self, f0, intensity, static, dynamic, sr) -> pd.DataFrame:
        """Structure features for temporal alignment with video data."""
        feature_df = pd.DataFrame({
            'timestamp': librosa.frames_to_time(
                np.arange(len(f0)), 
                sr=sr, 
                hop_length=self.hop_length
            ),
            'f0': f0,
            'intensity': intensity,
            **static,
            'dynamic_features': dynamic.tolist()
        })
        
        # Use existing Parquet handler from video pipeline
        self.rw.write_parquet(
            data=feature_df, 
            file=f"tonal_features_{datetime.now().strftime('%Y-%m-%d')}.parquet"
        )
        return feature_df

    def real_time_analysis(self, audio_buffer: np.ndarray) -> dict:
        """Streaming analysis compatible with video processing frame rates.
        
        Args:
            audio_buffer: 10ms audio chunk (512 samples at 16kHz)
            
        Returns:
            Dict with 'pitch_variability' and 'mean_intensity' for ML input
        """
        f0 = librosa.yin(audio_buffer, fmin=50, fmax=2000,
                        frame_length=self.frame_length,
                        hop_length=self.hop_length)
        
        return {
            # Standard deviation captures pitch modulation (emotional content)
            'pitch_variability': np.std(f0),
            
            # RMS intensity indicates vocal effort (stress detection)
            'mean_intensity': np.mean(librosa.feature.rms(
                y=audio_buffer,
                frame_length=self.frame_length
            ))
        }
