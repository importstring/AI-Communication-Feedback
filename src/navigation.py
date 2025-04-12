import curses
import sys

def ask_multiple_choice(stdscr, menu):
	curses.curs_set(0)
	current_row = 0
	
	while True:
		stdscr.clear()
		
		for idx, row in enumerate(menu):
			if idx == current_row:
				stdscr.addstr(idx, 0, f"> {row}")
			else: 
				stdscr.addstr(idx, 0, row)
		
		key = stdscr.getch()

		if key == curses.KEY_UP and current_row > 0:
			current_row -= 1
		elif key == curses.KEY_DOWN and current_row < len(menu) - 1:
			current_row += 1
		elif key == ord("\n"):
			stdscr.addstr(len(menu) + 1, 0, f"Selected {menu[current_row]}")
			stdscr.refresh()
			stdscr.getch()
			return menu[current_row]		
		
		stdscr.refresh()

		current_row = 0 if current_row >= len(menu) else current_row
		current_row = len(menu) - 1 if current_row < 0 else current_row

def multiple_choice(menu):
	return curses.wrapper(ask_multiple_choice, menu)


if __name__ == "__main__":
	while True:
		if multiple_choice (
			[r"Isn't this great?", "This sucks.", "Exit"]
		) == "Exit":
			sys.exit(1)
		
