from pathlib import Path

current_file = Path(__file__).resolve()

project_root = current_file.parents[2]

data_dir = project_root / 'data' / 'recordings' / 'transcripts'
src_dir = project_root / 'src' / 'factors'

for file in data_dir.iterdir():
    print(file)