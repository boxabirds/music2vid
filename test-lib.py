# This is a test of the library interface to deforum.

import json
from pathlib import Path
import argparse

def process_json_file(file_name: str):
    json_file = Path(file_name)
    with json_file.open() as f:
        data = json.load(f)

    compositions = data['compositions']
    for mp3_file in compositions:
        analysis_file = Path(f"{mp3_file.stem}-analysis/full-metadata.json")
        with analysis_file.open() as f:
            analysis_data = json.load(f)

        zoom = analysis_data['animation']['keyframe_zoom_animations']
        prompts = {}
        for key, value in analysis_data['keyframes'].items():
            prompts[int(key)] = value['lyric']

        # Do something with zoom and prompts

parser = argparse.ArgumentParser(description="Process a JSON file containing compositions.")
parser.add_argument("--input", required=True, help="Path to the input JSON file.")
args = parser.parse_args()
input_file = args.input

process_json_file(input_file)
