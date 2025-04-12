import numpy as np
import crepe
from gentle import Transcription

class ParalinguisticProcessor:
    def _calc_speech_rate(self, y, sr):
        """Forced alignment with Gentle ASR"""
        aligner = Transcription(y, sr)
        words = [w for w in aligner.transcribe() if w['word'] != '<sil>']
        duration = words[-1]['end'] - words[0]['start'] if words else 1
        return len(words) / (duration / 60)  # Words per minute

    def _measure_jitter(self, y, sr):
        """Perturbation analysis using REAPER algorithm"""
        _, f0 = crepe.predict(y, sr, viterbi=True)
        valid_f0 = f0[f0 > 0]
        diffs = np.abs(np.diff(valid_f0))
        return np.mean(diffs) / np.mean(valid_f0)  # Relative jitter
