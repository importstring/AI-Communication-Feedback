import librosa
import numpy as np
import pandas as pd
from datetime import datetime
from ..parquet_management import ReadWrite  
from .helper import save_factor_data

class TonalAnalyzer:
    """Modern prosody analysis with temporal alignment for ML pipelines"""
    
    def __init__(self, timestamp, sr=16000, frame_length=2048, hop_length=512):
        """
        Initialize audio processing parameters with video alignment.
        
        Args:
            sr: Sampling rate matching video analysis requirements
            frame_length: FFT window size (power-of-two for performance)
            hop_length: 10ms frames (512 samples @16kHz = 32ms frames)
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.rw = ReadWrite()
        self.timestamp = timestamp

    def extract_features(self, audio_path: str) -> pd.DataFrame:
        """
        Extract time-aligned prosodic features with 10ms resolution.
        
        Returns:
            DataFrame with columns:
            - timestamp: Frame-aligned timestamps
            - f0: Fundamental frequency (Hz)
            - intensity: RMS energy
            - [static features]: Global statistics
            - [dynamic features]: Time-series contours
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Core acoustic features
            f0 = librosa.yin(y, fmin=50, fmax=2000,
                           frame_length=self.frame_length,
                           hop_length=self.hop_length)
            
            intensity = librosa.feature.rms(y=y, 
                                          frame_length=self.frame_length,
                                          hop_length=self.hop_length)[0]

            # Custom feature computation
            static_features = self._compute_static_features(f0, intensity)
            dynamic_features = self._compute_dynamic_features(f0, intensity)

            return self._create_feature_dataframe(
                f0, intensity, static_features, dynamic_features, sr
            )
            
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            raise

    def _compute_static_features(self, f0, intensity):
        """Compute global statistics for presentation analysis"""
        f0_valid = f0[f0 > 0]
        return {
            'f0_mean': np.nanmean(f0_valid) if len(f0_valid) > 0 else 0,
            'f0_std': np.nanstd(f0_valid) if len(f0_valid) > 0 else 0,
            'intensity_var': np.nanvar(intensity),
            'speaking_rate': self._estimate_speaking_rate(intensity),
            'pitch_range': np.ptp(f0_valid) if len(f0_valid) > 0 else 0
        }

    def _compute_dynamic_features(self, f0, intensity):
        """Compute time-aligned features for sequence modeling"""
        return {
            'f0_contour': self._smooth_contour(f0),
            'intensity_contour': self._smooth_contour(intensity),
            'voicing_ratio': np.mean(f0 > 0)
        }

    def _estimate_speaking_rate(self, intensity, threshold=0.1):
        """Estimate syllables/sec using robust peak detection"""
        peaks = librosa.util.peak_pick(
            intensity, 
            pre_max=3, 
            post_max=3, 
            pre_avg=3, 
            post_avg=5, 
            delta=threshold, 
            wait=10
        )
        return len(peaks) / (len(intensity) * self.hop_length / self.sr)

    def _smooth_contour(self, values, window_size=5):
        """Apply causal moving average filter"""
        return np.convolve(
            values, 
            np.ones(window_size)/window_size, 
            mode='same'
        )

    def _create_feature_dataframe(self, f0, intensity, static, dynamic, sr) -> pd.DataFrame:
        """Structure features for Parquet storage and temporal alignment"""
        feature_df = pd.DataFrame({
            'timestamp': librosa.frames_to_time(
                np.arange(len(f0)), 
                sr=sr, 
                hop_length=self.hop_length
            ),
            'f0': f0,
            'intensity': intensity
        })
        
        # Add static features (broadcast to all frames)
        for k, v in static.items():
            feature_df[k] = v
            
        # Add dynamic features (time-aligned)
        for k, v in dynamic.items():
            feature_df[k] = v[:len(feature_df)]  # Ensure length match
            
        save_factor_data(feature_df, 'tonation', self.timestamp)
        return feature_df

    def real_time_analysis(self, audio_buffer: np.ndarray) -> dict:
        """
        Process 10ms audio chunks for real-time feedback.
        
        Args:
            audio_buffer: 512 samples @16kHz (32ms window)
            
        Returns:
            Dict with pitch variability and intensity metrics
        """
        f0 = librosa.yin(audio_buffer, 
                        fmin=50, 
                        fmax=2000,
                        frame_length=self.frame_length,
                        hop_length=self.hop_length)
        
        return {
            'pitch_variability': np.nanstd(f0),
            'mean_intensity': np.nanmean(
                librosa.feature.rms(
                    y=audio_buffer,
                    frame_length=self.frame_length
                )
            )
        }
