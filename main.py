import sys
import os
from utils import ask_multiple_choice
from src.analysis import analyze_video
from datetime import datetime
from src.data_extraction import record_video

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