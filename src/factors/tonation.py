import librosa
import numpy as np
from pydub import AudioSegment
import pyloudnorm as pyln
import crepe
from typing import Dict, Union, Optional

class AudioAnalyzer:
    def __init__(self, file_path: Optional[str] = None, audio_data: Optional[np.ndarray] = None):
        if file_path:
            self.waveform = librosa.load(file_path, sr=None)[0]
        elif audio_data is not None:
            self.waveform = audio_data
        else:
            raise ValueError("Either file_path or audio_data must be provided")
            
        # EBU R128 loudness normalization
        self.waveform = pyln.normalize.peak(self.waveform, -3.0)
        self.sr = 16000  # set sample rate

    def get_tonation_variance(self) -> float:
        """Improved pitch tracking using CREPE neural model"""
        _, f0 = crepe.predict(self.waveform, self.sr, viterbi=True)
        return f0[f0 > 0].std()

    def get_volume_dynamics(self) -> float:
        """Calculates dynamic range compression ratio"""
        rms = librosa.feature.rms(y=self.waveform)
        peak = np.max(rms)
        trough = np.min(rms)
        return 20 * np.log10(peak/trough) if trough != 0 else 0

class Tonation:
    """
    Analyzes tonation variance and patterns in speech.
    """
    def __init__(self):
        self.analyzer = None
        self.pitch_tracker = crepe
        
    def analyze_tonation(self, audio_file: Union[str, np.ndarray]) -> Dict:
        """
        Analyzes tonation variance in audio.
        
        Args:
            audio_file: Path to audio file or numpy array of audio data
            
        Returns:
            dict: Dictionary containing tonation metrics
        """
        # Initialize AudioAnalyzer
        self.analyzer = AudioAnalyzer(audio_file if isinstance(audio_file, str) else None,
                                    audio_file if isinstance(audio_file, np.ndarray) else None)
        
        # Get basic analysis
        basic_analysis = {
            'tonation_variance': self.analyzer.get_tonation_variance(),
            'volume_dynamics': self.analyzer.get_volume_dynamics()
        }
        
        # Extract pitch using CREPE
        time, frequency, confidence, _ = self.pitch_tracker.predict(
            self.analyzer.waveform, 
            self.analyzer.sr, 
            viterbi=True
        )
        
        # Calculate pitch statistics
        valid_pitch = frequency[confidence > 0.5]  # Only use high-confidence pitch estimates
        if len(valid_pitch) > 0:
            pitch_mean = np.mean(valid_pitch)
            pitch_std = np.std(valid_pitch)
            pitch_range = np.max(valid_pitch) - np.min(valid_pitch)
        else:
            pitch_mean = pitch_std = pitch_range = 0
        
        # Calculate pitch variation metrics
        pitch_variability = self._calculate_pitch_variability(frequency, confidence)
        pitch_modulation = self._calculate_pitch_modulation(frequency, confidence, time)
        
        return {
            'basic_analysis': basic_analysis,
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'pitch_range': float(pitch_range),
            'pitch_variability': pitch_variability,
            'pitch_modulation': pitch_modulation,
            'tonation_score': self._calculate_tonation_score(pitch_std, pitch_modulation)
        }
    
    def _calculate_pitch_variability(self, frequency: np.ndarray, 
                                   confidence: np.ndarray) -> float:
        """Calculate pitch variability metrics"""
        valid_freq = frequency[confidence > 0.5]
        if len(valid_freq) < 2:
            return 0.0
            
        # Calculate local variability
        local_variability = np.abs(np.diff(valid_freq))
        return float(np.mean(local_variability))
    
    def _calculate_pitch_modulation(self, frequency: np.ndarray, 
                                  confidence: np.ndarray, 
                                  time: np.ndarray) -> float:
        """Calculate pitch modulation metrics"""
        valid_indices = confidence > 0.5
        valid_freq = frequency[valid_indices]
        valid_time = time[valid_indices]
        
        if len(valid_freq) < 2:
            return 0.0
            
        # Calculate modulation depth
        modulation_depth = np.max(valid_freq) - np.min(valid_freq)
        # Calculate modulation rate
        zero_crossings = np.where(np.diff(np.signbit(valid_freq - np.mean(valid_freq))))[0]
        modulation_rate = len(zero_crossings) / (valid_time[-1] - valid_time[0])
        
        return float(modulation_depth * modulation_rate)
    
    def _calculate_tonation_score(self, pitch_std: float, 
                                pitch_modulation: float) -> float:
        """Calculate overall tonation score"""
        # Normalize scores to 0-1 range
        normalized_std = min(1.0, pitch_std / 100)  # Assuming max pitch std of 100 Hz
        normalized_mod = min(1.0, pitch_modulation / 50)  # Assuming max modulation of 50
        
        # Weighted combination
        return 0.6 * normalized_std + 0.4 * normalized_mod