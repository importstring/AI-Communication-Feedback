import wave
import numpy as np
import os
import glob
from datetime import datetime

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
            frame_rate = wav_file.getframerate
            n_frames= wav_file.getnframes()
            frames_per_interval = int(frame_rate * interval)
            rms_values = []
            for start in range(0, n_frames, frames_per_interval):
                wav_file.setpos(start)
                frames = wav_file.readframes(frames_per_interval)
                if len(frames) == 0:
                    break
                
                audio_array = np.frombuffer(frames, dtype=np.int16) / 32768.0
                rms = np.sqrt(np.mean(np.sqaure(audio_array)))
                rms_values.append(rms)

            return rms_values

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
            
            # Extract timestamps from filenames
            file_times = []
            for f in files:
                timestamp_str = os.path.basename(f).split('_')[1].split('.')[0]
                file_time = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                file_times.append((f, file_time))
            
            # Sort by timestamp and return latest
            latest = max(file_times, key=lambda x: x[1])
            return latest[0]
        
        except Exception as e:
            print(f"Error finding audio file: {str(e)}")
            return None