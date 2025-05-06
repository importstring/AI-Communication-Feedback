import pandas as pd
import numpy as np
from datetime import datetime
from pydub import AudioSegment
import os
import wave
import json
import glob
from vosk import Model, KaldiRecognizer
from pathlib import Path
from ..parquet_management import ReadWrite

class AudioTranscriber:
    def __init__(self, audio_path, model_path="model"):
        self.audio_path = audio_path
        self.model_path = Path(model_path)
        self.transcript = ""
        self.save_path = '../data/recordings/transcripts'
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Vosk model not found at {self.model_path}. Download from https://alphacephei.com/vosk/models")

    def convert_to_wav(self):
        """Convert audio to WAV format (mono, 16kHz, 16-bit PCM)."""
        if not self.audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(self.audio_path)
            
            audio = audio.set_frame_rate(16000).set_channels(1)
            if audio.sample_width != 2:
                audio = audio.set_sample_width(2)  
                
            wav_path = Path(self.audio_path).with_suffix('.wav')
            audio.export(wav_path, format="wav")
            self.audio_path = str(wav_path)

    def transcribe(self):
        """Transcribe audio using Vosk with proper error handling."""
        self.convert_to_wav()
        
        try:
            with wave.open(self.audio_path, "rb") as wf:
                if not (wf.getnchannels() == 1 and 
                        wf.getsampwidth() == 2 and 
                        wf.getframerate() == 16000):
                    raise ValueError("Invalid audio format after conversion. Must be mono 16kHz 16-bit PCM.")
                
                model = Model(str(self.model_path))
                rec = KaldiRecognizer(model, wf.getframerate())
                rec.SetWords(True)  
                
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        results.append(json.loads(rec.Result()))
                
                final_result = json.loads(rec.FinalResult())
                results.append(final_result)
                
                self.transcript = " ".join([res.get('text', '') for res in results if res])
                
        except Exception as e:
            self.transcript = f"[Transcription Error: {str(e)}]"
            
        return self.transcript.strip()

    def save_transcript(self, output_path=None):
        """Save transcript with directory creation."""
        if not output_path:
            output_path = Path(self.audio_path).with_suffix('.txt')
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.transcript)
            
        return str(output_path.resolve())


class FileManager:
    def __init__(self):
        self.base_audio_path = '../data/recording/audio/'
        self.base_video_path = '../data/recording/video/'
        self.base_transcript_path = '../data/recording/transcripts/'
        self.base_measurements_path = '../data/recording/measurements/'
        self.transcripts_path = '/transcripts/'

    def get_timestamp(self) -> str:
        """
        Get the timestamp of the most recent recording
        """
        path = self.get_path(video=True, audio=False, transcript=False)
        if path is None:
            return None
        
        filename = os.path.basename(path)

        try: # Expected format: output_YYYY-MM-DD_HH-MM-SS.ext
            parts = filename.split('_')
            
            if len(parts) < 3 or parts[0] != 'output':
                raise ValueError(f"Filename {filename} does not match expected format 'output_YYYY-MM-DD_HH-MM-SS.ext'")
            
            date_part = parts[1]
            time_part = parts[2].split('.')[0]
            timestamp_str = f"{date_part}_{time_part}"
            
            datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
            
            return timestamp_str
        except (IndexError, ValueError) as e:
            print(f"Filename {filename} does not contain a valid timestamp: {str(e)}")
            return None


    def get_path(self, video: bool = False, audio: bool = False, transcript: bool = False) -> str:
        """
        Get the path to the most recent file of the specified type.
        
        When recorded, files are named with a timestamp:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        video_filename = f'output_{timestamp}.mp4'
        audio_filename = f'output_{timestamp}.wav'
        
        Returns the full path to the most recent file of the requested type.
        """
        file_extensions = {
            'audio': 'wav', 
            'video': 'mp4',
            'transcript': 'txt'
        }
        paths = {
            'audio': self.base_audio_path,
            'video': self.base_video_path,
            'transcript': self.base_transcript_path
        }

        file_type = 'audio' if audio else 'video' if video else 'transcript'
        file_extension = file_extensions[file_type]
        base_path = paths[file_type]
        
        try:
            files = glob.glob(os.path.join(base_path, f'output_*.{file_extension}'))
            if not files:
                raise FileNotFoundError(f"No {file_extension} files found in {base_path}")
            
            file_times = []

            for f in files:
                try:
                    timestamp_str = os.path.basename(f).split('_')[1].split('.')[0]
                    file_time = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                    file_times.append((f, file_time))
                except (IndexError, ValueError):
                    continue
            
            if not file_times:
                raise ValueError(f"No files with valid timestamp format found in {base_path}")
                
            latest = max(file_times, key=lambda x: x[1])
            return latest[0]
        
        except Exception as e:
            print(f"Error finding {file_type} file: {str(e)}")
            return None

def save_factor_data(data, factor_name: str, timestamp: str):
    """
    Enhanced data saver with format validation.
    """
    rw = ReadWrite()
    
    filename = f"{factor_name}.parquet" 
    
    sub_dir = timestamp

    return rw.write_parquet(
        data=data,
        file=filename,
        sub_dir=str(sub_dir)
    )

def get_video_path(self):
    current_file = Path(__file__).resolve()

    project_root = current_file.parents[2]

    data_dir = project_root / 'data' / 'recordings' / 'video'
    src_dir = project_root / 'src' / 'factors'

    return str(data_dir) + self.timestamp        

def get_transcript_path(self):
    current_file = Path(__file__).resolve()

    project_root = current_file.parents[2]

    data_dir = project_root / 'data' / 'recordings' / 'transcripts'
    src_dir = project_root / 'src' / 'factors'

    return str(data_dir) + self.timestamp        

def get_audio_path(self):
    current_file = Path(__file__).resolve()

    project_root = current_file.parents[2]

    data_dir = project_root / 'data' / 'recordings' / 'audio'
    src_dir = project_root / 'src' / 'factors'

    return str(data_dir) + self.timestamp        

def read_transcript(self):
    transcript_path = self.get_transcript_path()
    file = open(transcript_path, 'r')
    text = file.read()
    file.close()
    return text

if __name__ == "__main__":
    transcriber = AudioTranscriber("input.mp3", model_path="model")
    print("Transcribing...")
    transcript = transcriber.transcribe()
    print(f"Transcript: {transcript}")
    saved_path = transcriber.save_transcript()
    print(f"Saved to: {saved_path}")
