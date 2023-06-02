import json
import glob
import argparse
import re
from pathlib import Path
from collections import OrderedDict

def read_file(file_path):
    with file_path.open("r") as file:
        content = file.read()
    return content

def write_file(file_path, content):
    with file_path.open("w") as file:
        file.write(content)

def remove_quotes(s):
    if s.startswith('"') and s.endswith('",'):
        return s[1:-2]
    return s

parser = argparse.ArgumentParser(description="Migrate text files associated with mp3 files")
parser.add_argument("--input", required=True, help="Location of the mp3 files")
args = parser.parse_args()
input_folder = Path(args.input)

mp3_files = list(input_folder.glob("*.mp3"))

for mp3_file in mp3_files:
    mp3_stem = mp3_file.stem
    analysis_folder = input_folder / f"{mp3_stem}-analysis"

    prompts_raw_path = analysis_folder / f"{mp3_stem}-visual-prompts.json"
    prompts_raw = read_file(prompts_raw_path)
    prompts_raw_lines = prompts_raw.splitlines()

    full_metadata_path = analysis_folder / "full-metadata.json"
    full_metadata = json.loads(read_file(full_metadata_path))

    # the second line of the visual-prompts file is the visual description e.g. 0: "Vibrant colors, dynamic movement, and infectious energy of a jazzy, hip-hop night at Birdland.",
    prompt = prompts_raw_lines[1].split(": ")[1]
    full_metadata["visual_description"] = remove_quotes(prompt)

    write_file(full_metadata_path, json.dumps(full_metadata, indent=2))
