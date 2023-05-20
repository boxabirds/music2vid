# This is a test of the library interface to deforum to create a session for a single composition.

import json
from pathlib import Path
import argparse
from lib.parameters import DeforumAnimArgs, DeforumArgs, Root

def initialise_args_from_combination_configuration(file_name: str, mp3_dir: str):
    json_file = Path(file_name)
    with json_file.open() as f:
        data = json.load(f)

    mp3_files_path = Path(mp3_dir)

    compositions = data['compositions']
    for mp3_file in compositions:
        analysis_file = mp3_files_path / Path(Path(mp3_file).stem + "-analysis/full-metadata.json")
        analysis_data = analysis_file.read_text()
        # read analysis data into a json object 
        analysis_data = json.loads(analysis_data)

        zoom = analysis_data['animation']['keyframe_zoom_animations']
        prompts = {}
        for key, value in analysis_data['keyframes'].items():
            prompts[int(key)] = value['lyric']

        print(f'== {mp3_file}:')
        print(f'zoom: "{zoom[:60]}..."')
        print(f'prompts:\n{json.dumps(prompts,indent=2)}\n\n')

parser = argparse.ArgumentParser(description="Process a JSON file containing compositions.")
parser.add_argument("--input", required=True, help="Path to the input JSON file.")
parser.add_argument("--mp3-dir", default="music", help="Location of MP3 files in config file.")
args = parser.parse_args()
input_file = args.input
mp3_dir = args.mp3_dir

initialise_args_from_combination_configuration(input_file, mp3_dir)
