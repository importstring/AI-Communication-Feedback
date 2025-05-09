import sys
import os
from src.navigation import multiple_choice
# from src.factors.main import Factors
# from src.analysis import analyze_video
from datetime import datetime
from src.recording_tools import record as record_video

def main():
	while True:
		choices = [
			'Record video', 'Analyze video', 'Exit'
		]

		command = multiple_choice(choices) 
	
		if command == 'Record video':
			timestamp = record_video()
		#	factors = Factors()
		#	factors.extract_factors(timestamp)
		#	factors.save_factors() # TODO: PLACEHOLDER
	
		elif command == 'Analyze video':
			pass
		else:
			sys.exit(0)
			
if __name__ == "__main__":
	main()