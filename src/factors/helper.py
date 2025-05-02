import pandas as pd
import numpy as np
from datetime import datetime
from pydub import AudioSegment
import os
import wave
import json
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

def save_factor_data(data, factor_name, custom_filename=None):
    """
    Enhanced data saver with format validation.
    """
    rw = ReadWrite()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{factor_name}_{timestamp}.parquet" if not custom_filename else custom_filename
    filename = filename if filename.endswith('.parquet') else f"{filename}.parquet"
    
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data if isinstance(data, list) else [data])
    
    sub_dir = Path("measurements") / factor_name
    return rw.write_parquet(
        data=data,
        file=filename,
        sub_dir=str(sub_dir)
    )

if __name__ == "__main__":
    transcriber = AudioTranscriber("input.mp3", model_path="model")
    print("Transcribing...")
    transcript = transcriber.transcribe()
    print(f"Transcript: {transcript}")
    saved_path = transcriber.save_transcript()
    print(f"Saved to: {saved_path}")
