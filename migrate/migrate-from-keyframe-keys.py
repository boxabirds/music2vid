# before this script is run, the animation prompts are stored with respect to keyframes. We need to abstract that
# away so that the animation prompts are stored with respect to time. This script does that.
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


def find_nearest_keyframe_prompt(metadata, timestamp: float, kfps: int):
    # create a keyframe based on timestamp
    estd_keyframe = int(timestamp * kfps)

    # we are mapping a course keyframe value to a realtime timestamp so there could be rounding error. So check each of these in order and return the first one that exists: metadata["keyframes"][str(estd_keyframe)], metadata["keyframes"][str(estd_keyframe-1)], metadata["keyframes"][str(estd_keyframe+1)]
    for key in [str(estd_keyframe), str(estd_keyframe-1), str(estd_keyframe+1)]:
        if key in metadata["keyframes"]:
            return metadata["keyframes"][key]["prompt"]

    # if none of the keyframes exist, return None
    print(f"ERROR: could not find keyframe for timestamp '{timestamp}': {estd_keyframe}+/-1")
    return None


parser = argparse.ArgumentParser(description="Migrate prompt section of the full-metadata.json file from keyframe-based to time-based.")
parser.add_argument("--input", required=True, help="Location of the mp3 files with analysis folders containing full-metadata.json files.")
args = parser.parse_args()
input_folder = Path(args.input)

mp3_files = list(input_folder.glob("*.mp3"))

for mp3_file in mp3_files:
    mp3_stem = mp3_file.stem
    analysis_folder = input_folder / f"{mp3_stem}-analysis"

    full_metadata_path = analysis_folder / "full-metadata.json"
    full_metadata = json.loads(read_file(full_metadata_path))

    # the frame rate used for the keyframes is defined in the metadata file itself
    kfps = full_metadata["animation"]["keyframes_per_second"]

    for line in full_metadata["full_transcript"]["segments"]:
        prompt = find_nearest_keyframe_prompt(full_metadata, line["start"], kfps)
        line["prompt"] = prompt

    del full_metadata["keyframes"]
    write_file(full_metadata_path, json.dumps(full_metadata, indent=2))
