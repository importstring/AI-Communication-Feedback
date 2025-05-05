from utils import ask_multiple_choice
from src.analysis import analyze_video
from src.recording_tools import record
from src.factors.helper import AudioTranscriber
from datetime import datetime
import sys
import os

class Recording:
	def __init__(self):
		recording_dir = '/data/recordings/'
		timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		self.video_path = recording_dir + f'video/output_{timestamp}.mp4'
		self.audio_path = recording_dir + f'audio/output_{timestamp}.wav'
		self.transcript_path = recording_dir + f'transcript/output_{timestamp}.txt'

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
	return recording
	
def main():
	while True:
		choices = [
			'Record video', 'Analyze video', 'Exit'
		]

		command = ask_multiple_choice(choices) 
	
		if command == 'Record video':
			recording = record_video()
		elif command == 'Analyze video':
			pass
		else:
			sys.exit(0)
			
if __name__ == "__main__":
	main()