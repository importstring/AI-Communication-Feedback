# Imports
import pandas as pd
import numpy as np
from datetime import datetime
import speech_recognition as sr
from pydub import AudioSegment
import os

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

# Test case
if __name__ == "__main__":
    pass