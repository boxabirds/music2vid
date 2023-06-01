from pathlib import Path
from pydub import AudioSegment
import librosa
import numpy as np
from typing import List
import soundfile as sf
from tqdm import tqdm
from lib import time_based_pulses
import json


def ensure_wav_conversion_exists(mp3_file_path: Path, wav_file_path: Path):
    # skip if the wav file already exists
    if wav_file_path.exists():
        print(f"Skipping {mp3_file_path} because {wav_file_path} already exists.")
    else:
        sound = AudioSegment.from_mp3(mp3_file_path)
        sound.export(wav_file_path, format="wav")


def calculate_onsets(wav_file_path: Path, debug_generate_onset_clicks = True) -> np.ndarray:
    """
    Calculate significant onset times in a given audio file.

    Onsets are the beginning of a rapid change in volume -- someone starts speaking or a drum hit. 
    We want to reflect this visually with an increase in zoom speed. 
    This function calculates the onsets in an audio file, filters out the ones below a given
    percentile threshold, and returns the significant onset times as a NumPy array.

    :param wav_file_path: The path to the audio file in WAV format.
    :type wav_file_path: Path
    :return: A numpy array of significant onset times in seconds.
    :rtype: np.ndarray
    """

    # we calculate all the onsets then filter out the ones that are below the threshold
    # it's an arbitrary heuristic as a starting point that users can tweak
    PERCENTILE_THRESHOLD = 97
    y, sr = librosa.load(wav_file_path)

    # we want the same impact of a drum track regardless of the volume of the track
    y = librosa.util.normalize(y)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Detect onsets using the onset strength envelope
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    # Convert onset frames to onset times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # if we don't threshold we'll end up with onsets for every single percussive sound which is… a lot
    threshold = np.percentile(onset_env, PERCENTILE_THRESHOLD)

    # Filter out onsets with strength below the threshold
    significant_onset_indices = [idx for idx, frame in enumerate(onset_frames) if onset_env[frame] > threshold]
    significant_onset_frames = onset_frames[significant_onset_indices]
    # Convert significant onset frames to onset times
    significant_onset_times = librosa.frames_to_time(significant_onset_frames, sr=sr)

    # audio debugging: generate a click file
    # TODO we really want the clicks on the original file but that's a bit more work
    if debug_generate_onset_clicks:
        y_clicks = librosa.clicks(times=significant_onset_times, length=len(y), sr=sr)
        sf.write(wav_file_path.parent / "drums-with-onset-clicks.wav", y+y_clicks, sr)
    return significant_onset_times


def generate_animations_from_onsets(analsys_dir_path: Path, mp3_file:Path, times_in_sec: np.ndarray, pulse_boost=0.4, pulse_duration_in_sec=0.5):
    # pulse_boost: how much amplitude a pulse adds to the base zoom value
    # pulse_duration_in_sec: how long a pulse lasts in seconds before returning back to the base zoom value

    BASE_ZOOM = "1.01"
    pulse_times = {0: BASE_ZOOM}
    metadata_path = analsys_dir_path / "full-metadata.json"

    # TODO do other animations as well like rotation and translation
    timings_string = time_based_pulses.generate_pulses(pulse_duration_in_sec, pulse_boost, 1.0, times_in_sec)

    # read in the metadata file
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # remove the old entry for keyframe zoom animations if there is one
    del metadata["animation"]["keyframe_zoom_animations"]
    metadata["animation"]["zoom_animation_times"] = timings_string

    # now write the metadata file back out
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

# take a folder through command line argument "--input" and iterate through it finding all mp3 files. 
# each mp3 has a corresponding analysis folder named "<songname>-analysis" in the same location. 
# there is also a drum stems file inside the analysis folder named "<songname>-drums.mp3"
# take that drum stem file and convert to wav
# take the wav and calculate the onsets
# convert onsets into a Disco-compatible keyframe string (for use with deforum)

import argparse
from pathlib import Path

def main(input_folder: Path):
    # Iterate through the input folder and find all MP3 files
    for mp3_file in tqdm(input_folder.glob("*.mp3"), desc="detecting pulses and creating zoom animations for them..."):
        print(f"{mp3_file.stem} ")
        # Locate the corresponding analysis folder and drum stems file
        analysis_folder = mp3_file.parent / (mp3_file.stem + "-analysis")
        drum_stems_file = analysis_folder / (mp3_file.stem + "-drums.mp3")

        # Convert the drum stems file to WAV format
        drum_stems_wav = analysis_folder / (mp3_file.stem + "-drums.wav")
        ensure_wav_conversion_exists(drum_stems_file, drum_stems_wav)

        # Calculate the onsets for the WAV file
        onsets = calculate_onsets(drum_stems_wav)

        # Convert the onsets into a Disco-compatible keyframe string
        generate_animations_from_onsets(analysis_folder, mp3_file, onsets)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take a folder of MP3 files and add zoom animations to the corresponding analysis/full-metadta.json file based on drum beats.")
    parser.add_argument("--input", type=Path, required=True, help="Input folder containing MP3 files.")
    args = parser.parse_args()

    main(args.input)
