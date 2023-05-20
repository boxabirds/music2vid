import json
import argparse
from pathlib import Path
from tinytag import TinyTag

def get_duration(file_path):
    tag = TinyTag.get(file_path)
    return tag.duration

# Handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Path to the directory containing MP3 files')
args = parser.parse_args()

mp3_directory = Path(args.input)

for mp3_file in mp3_directory.glob("*.mp3"):
    mp3_stem = mp3_file.stem
    analysis_directory = mp3_directory / f"{mp3_stem}-analysis"
    
    if analysis_directory.exists():
        metadata_file = analysis_directory / "full-metadata.json"
        metadata = {}
        
        if metadata_file.exists():
            with metadata_file.open("r") as fp:
                metadata = json.load(fp)
        
        duration = get_duration(mp3_file)
        metadata['duration'] = duration
        
        with metadata_file.open("w") as fp:
            json.dump(metadata, fp)
    else:
        print(f"Error: Analysis directory '{analysis_directory}' not found. Skipping {mp3_file.stem}.")
