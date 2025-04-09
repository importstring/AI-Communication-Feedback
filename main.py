from utils import record_video, AskMultipleChoice

def main():
	while True:
		choices = {
			'Record video': record_video
		} # More will be added if needed later on

		command = ask_multiple_choice(choices) # input: array, output: command
	
if __name__ == "__main__":
	main()
