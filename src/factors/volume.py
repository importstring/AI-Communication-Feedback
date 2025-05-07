import wave
import numpy as np
import os
import glob
from datetime import datetime
from .helper import save_factor_data, get_video_path, get_audio_path, get_transcript_path
import pandas as pd

class VolumeVarience:
    def __init__(self, timestamp: str = None):
        self.volume = None
        self.volume_variance = None
        self.audio_path = get_audio_path(timestamp)
        self.timestamp = timestamp     
        
    def calculate_rms(self, audio_path: str, interval: float = 0.25) -> float:
        """
        Calculate the root mean square (RMS) of an audio file at every point in time at 0.25 second intervals.
            Returns a list of RMS values for each interval.
        """
        with wave.open(audio_path, 'rb') as wav_file:
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            frames_per_interval = int(frame_rate * interval)
            rms_values = []
            for start in range(0, n_frames, frames_per_interval):
                wav_file.setpos(start)
                frames = wav_file.readframes(frames_per_interval)
                if len(frames) == 0:
                    break
                
                audio_array = np.frombuffer(frames, dtype=np.int16) / 32768.0
                rms = np.sqrt(np.mean(np.square(audio_array)))
                rms_values.append(rms)

            return rms_values

    def save_data(self, data):
        """
        Takes the input data and the path of the input file 
        Saves the data to a file in the ../data/measurements/{date recorded}/analysis.parquet
        """
        timestamps = np.arange(len(data)) * 0.25  
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'rms_value': data,
            'audio_source': os.path.basename(self.audio_path)
        })
        
        self.volume_variance = np.var(data)
        df['volume_variance'] = self.volume_variance
        
        save_factor_data(df, 'volume', self.timestamp)

    def analyze_and_save(self):
        data = self.calculate_rms(self.audio_path)
        self.save_data(data)