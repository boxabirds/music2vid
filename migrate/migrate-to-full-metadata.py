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

    zoom_path = analysis_folder / f"{mp3_stem}-keyframe-zoom.json"
    prompts_raw_path = analysis_folder / f"{mp3_stem}-visual-prompts.json"
    full_transcript_path = analysis_folder / f"{mp3_stem}-medium.json"
    transcript_frame_timings_raw_path = analysis_folder / f"{mp3_stem}-lyrics-medium.txt"
    metadata_path = analysis_folder / "metadata.json"
    full_metadata_path = analysis_folder / "full-metadata.json"

    zoom = read_file(zoom_path)
    prompts_raw = read_file(prompts_raw_path)
    full_transcript = read_file(full_transcript_path)
    transcript_frame_timings_raw = read_file(transcript_frame_timings_raw_path)
    estd_bpm = read_file(metadata_path)

    zoom_data = json.loads(zoom)
    prompts_raw_lines = prompts_raw.splitlines()
    style_match = re.search(r'"style":\s*"(.*?)",', prompts_raw_lines[0])
    style = style_match.group(1) if style_match else None
    prompts = {int(line.split(": ")[0]): line.split(": ")[1] for line in prompts_raw_lines[1:]}
    print(f"prompts:\n {prompts}")
    full_transcript_data = json.loads(full_transcript)
    transcript_frame_timings = {}
    for line in transcript_frame_timings_raw.splitlines():
        try:
            key, value = line.split(": ")
            transcript_frame_timings[int(key)] = value
        except ValueError:
            # Skip lines that don't contain a colon separator
            continue
        except KeyError:
            # Skip missing keys
            continue
    estd_bpm_data = json.loads(estd_bpm)

    keyframes = OrderedDict()
    keyframes[0] = {"prompt": remove_quotes(prompts[0])}  # Add the first entry manually

    for key in prompts.keys():
        if key in transcript_frame_timings:
            keyframes[key] = {"lyric": remove_quotes(transcript_frame_timings[key]), "prompt": remove_quotes(prompts[key])}

    full_metadata = {
        "animation": zoom_data,
        "estd_bpm": estd_bpm_data["tempo"],
        "style": style,
        "keyframes": keyframes,
        "full_transcript": full_transcript_data
    }

    write_file(full_metadata_path, json.dumps(full_metadata, indent=2))
