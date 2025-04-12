import librosa
import numpy as np
from pydub import AudioSegment
import pyloudnorm as pyln
import crepe

class AudioAnalyzer:
    def __init__(self, file_path):
        # EBU R128 loudness normalization
        self.waveform = librosa.load(file_path, sr=None)[0]
        self.waveform = pyln.normalize.peak(self.waveform, -3.0)
        self.sr = 16000 # set sample rate

    def get_tonation_variance(self):
        """Improved pitch tracking using CREPE neural model"""
        _, f0 = crepe.predict(self.waveform, self.sr, viterbi=True)
        return f0[f0 > 0].std()

    def get_volume_dynamics(self):
        """Calculates dynamic range compression ratio"""
        rms = librosa.feature.rms(y=self.waveform)
        peak = np.max(rms)
        trough = np.min(rms)
        return 20 * np.log10(peak/trough) if trough != 0 else 0