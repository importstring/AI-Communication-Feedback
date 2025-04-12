from utils import record_video, ask_multiple_choice

def main():
	while True:
		choices = {
			'Record video': record_video,
			'Review feedback': None,
			'Exit': None
		} # More will be added if needed later on

		command = ask_multiple_choice(choices) # input: array, output: command
	
if __name__ == "__main__":
	main()
