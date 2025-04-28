import wave
import numpy as np
import os
import glob
from datetime import datetime
from .helper import save_factor_data
import pandas as pd

class VolumeVarience:
    def __init__(self):
        self.volume = None
        self.volume_variance = None
        self.base_audio_path = '../data/recording/audio/'
        self.filename = self.get_path()
    
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

    def calculate_and_save(self) -> dict:
        """
        Calculate the variance of the RMS values and save the data.
        
        Returns:
            Dictionary with analysis results and save path
        """
        path = self.get_path()
        if not path:
            return {'error': 'No audio file found'}
            
        data = self.calculate_rms(path)
        
        save_path = self.save_data(data, path)
        
        return {
            'rms_values': data,
            'volume_variance': self.volume_variance,
            'audio_path': path,
            'save_path': save_path
        }

    def save_data(self, data, audio_path):
        """
        Takes the input data and the path of the input file 
        Saves the data to a file in the ../data/measurements/{date recorded}/analysis.parquet
        """
        timestamps = np.arange(len(data)) * 0.25  
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'rms_value': data,
            'audio_source': os.path.basename(audio_path)
        })
        
        self.volume_variance = np.var(data)
        df['volume_variance'] = self.volume_variance
        
        return save_factor_data(df, 'volume')

    def get_path(self):
        """
        Get the path to the most recent audio file.
        
        When recorded here are the file names:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            video_filename = f'output_{timestamp}.mp4'
            audio_filename = f'output_{timestamp}.wav'

            base_dir = '../data/recordings'

            video_filename = base_dir + '/video/' video_filename
            audio_filename = base_dir + '/audio/' audio_filename
        
        Return the most recent .wav file
        """
        try:
            files = glob.glob(os.path.join(self.base_audio_path, 'output_*.wav'))
            if not files:
                raise FileNotFoundError(f"No WAV files found in {self.base_audio_path}")
            
            file_times = []
            for f in files:
                timestamp_str = os.path.basename(f).split('_')[1].split('.')[0]
                file_time = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                file_times.append((f, file_time))
            
            latest = max(file_times, key=lambda x: x[1])
            return latest[0]
        
        except Exception as e:
            print(f"Error finding audio file: {str(e)}")
            return None