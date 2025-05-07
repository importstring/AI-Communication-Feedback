from factors.body_language import JointMap
from factors.emotions import Emotion, MultimodalEmotionAnalyzer
from factors.helper import AudioTranscriber
from factors.structure import Structure
from factors.tonation import TonalAnalyzer
from factors.volume import VolumeVarience
from factors.coherence import CoherenceAnalyzer
from factors.speech_patterns import SpeechPatternAnalyzer
from factors.main import run_factors

from recording_tools import record

from navigation import ask_multiple_choice
from factors.main import FactorAnalyzer

from datetime import datetime
import os
import sys


class Recording:
	def __init__(self):
		recording_dir = '/data/recordings/'
		self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		self.video_path = recording_dir + f'video/output_{self.timestamp}.mp4'
		self.audio_path = recording_dir + f'audio/output_{self.timestamp}.wav'
		self.transcript_path = recording_dir + f'transcript/output_{self.timestamp}.txt'

	def save_transcript(self):
		self.timestamp()
		transcriber = AudioTranscriber()
		transcriber.save_transcript(self.audio_path, self.transcript_path)

	def capture(self):
		record(self.timestamp)
		self.save_transcript()
		
def record_video():
	recording = Recording()
	recording.capture()
	recording.save_transcript()
	analyzer = FactorAnalyzer(recording.timestamp)
	analyzer.analyze_and_save() 
	return recording
	
