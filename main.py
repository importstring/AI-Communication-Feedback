import sys
import os
import time
from src.navigation import multiple_choice
from src.factors.main import Factors
from src.analysis import analyze_video
from datetime import datetime
from src.recording_tools import record as record_video
from src.video_analysis.main import analyze_video

def main():
	print('got here')
	while True:
		choices = [
			'Record video', 'Analyze video', 'Exit'
		]
		print('Select an option:')
		command = multiple_choice(choices) 
		print('You selected:', command)
		time.sleep(10)
	
		if command == 'Record video':
			timestamp = record_video()
			factors = Factors(timestamp)
			factors.extract_factors(timestamp)
		elif command == 'Analyze video':
			analyze_video(timestamp) # In future allow user to pick timestamp
		else:
			sys.exit(0)

main()

if __name__ == "__main__":
	main()