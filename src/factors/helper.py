# Imports
import pandas as pd
import numpy as np
from datetime import datetime
import speech_recognition as sr
from pydub import AudioSegment
import os
from ..parquet_management import ReadWrite

# Main Class
class AudioTranscriber:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.recognizer = sr.Recognizer()
        self.transcript = ""
        self.save_path = '../data/measurements/'

    def convert_to_wav(self):
        """Convert audio to .wav if not already in that format."""
        if not self.audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(self.audio_path)
            wav_path = os.path.splitext(self.audio_path)[0] + ".wav"
            audio.export(wav_path, format="wav")
            self.audio_path = wav_path

    def transcribe(self):
        """Transcribe the audio file to text."""
        self.convert_to_wav()
        with sr.AudioFile(self.audio_path) as source:
            audio_data = self.recognizer.record(source)
            try:
                self.transcript = self.recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                self.transcript = "[Unrecognizable audio]"
            except sr.RequestError as e:
                self.transcript = f"[API unavailable: {e}]"
        return self.transcript

    def save_transcript(self, output_path=None):
        """Save the transcript to a text file."""
        if not output_path:
            output_path = os.path.splitext(self.audio_path)[0] + ".txt"
        with open(output_path, 'w') as f:
            f.write(self.transcript)
        return output_path

def save_factor_data(data, factor_name, custom_filename=None):
    """
    Standard function to save factor analysis data to parquet format.
    
    Args:
        data: Data to save (dict, list, or DataFrame)
        factor_name: Name of the factor (e.g., 'volume', 'emotions')
        custom_filename: Optional custom filename, otherwise uses timestamp
        
    Returns:
        Filepath where data was saved
    """
    # Initialize ReadWrite utility
    rw = ReadWrite()
    
    # Create filename with timestamp if not provided
    if not custom_filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{factor_name}_{timestamp}.parquet"
    else:
        filename = custom_filename
        if not filename.endswith('.parquet'):
            filename += '.parquet'
            
    # Save to measurements/factor_name directory
    sub_dir = os.path.join("measurements", factor_name)
    
    # Save and return filepath
    return rw.write_parquet(data=data, file=filename, sub_dir=sub_dir)

# Test case
if __name__ == "__main__":
    pass