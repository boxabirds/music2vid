from pathlib import Path
from pydub import AudioSegment
import librosa
import numpy as np
from typing import List
import soundfile as sf
from tqdm import tqdm


def convert_to_wav(mp3_file_path: Path, wav_file_path: Path):
    sound = AudioSegment.from_mp3(mp3_file_path)
    sound.export(wav_file_path, format="wav")


def calculate_onsets(wav_file_path: Path, debug_generate_onset_clicks = True) -> np.ndarray:
    """
    Calculate significant onset times in a given audio file.

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


def create_key_frame_strings(analysis_folder: Path, mp3_file:Path, times_in_sec: np.ndarray, kfps=3, pulse=0.4):
    BASE_ZOOM = "1.03"
    kf = {0: BASE_ZOOM}

    num_keyframes = len(times_in_sec) * kfps
    for time_in_sec in times_in_sec:
        key_frame_number = int(np.ceil(time_in_sec * kfps))
        kf[key_frame_number] = f"{1 + pulse:.2f}"
        
        if key_frame_number + 1 < num_keyframes:
            kf[key_frame_number + 1] = f"{1 + pulse / 2:.2f}"
        
        if key_frame_number + 2 < num_keyframes:
            kf[key_frame_number + 2] = f"{1 + pulse / 4:.2f}"
        
        if key_frame_number + 3 < num_keyframes:
            kf[key_frame_number + 3] = BASE_ZOOM

    # Generate the final key frame string
    kf_string = ", ".join(f"{key}: ({value})" for key, value in kf.items())

    # now write the keyframe zoom string to a file "<songname>-keyframe-zoom.disco.json"
    keyframe_zoom_file = analysis_folder / (mp3_file.stem + "-keyframe-zoom.json")
    with open(keyframe_zoom_file, "w") as f:
        # write a json entry with two keys: "kfps" and "keyframe_zoom_animations"
        f.write(f'{{"keyframes_per_second": {kfps}, "keyframe_zoom_animations": "{kf_string}"}}')

    return kf_string

# take a folder through command line argument "--input" and iterate through it finding all mp3 files. 
# each mp3 has a corresponding analysis folder named "<songname>-analysis" in the same location. 
# there is also a drum stems file inside the analysis folder named "<songname>-drums.mp3"
# take that drum stem file and convert to wav
# take the wav and calculate the onsets
# convert onsets into a Disco-compatible keyframe string (for use with deforum)

import argparse
from pathlib import Path

def main(input_folder: Path, kfps:int):
    # Iterate through the input folder and find all MP3 files
    for mp3_file in tqdm(input_folder.glob("*.mp3"), desc="keyframe timings => deforum zoom animation strings..."):
        print(f"{mp3_file.stem} ")
        # Locate the corresponding analysis folder and drum stems file
        analysis_folder = mp3_file.parent / (mp3_file.stem + "-analysis")
        drum_stems_file = analysis_folder / (mp3_file.stem + "-drums.mp3")

        # Convert the drum stems file to WAV format
        drum_stems_wav = analysis_folder / (mp3_file.stem + "-drums.wav")
        convert_to_wav(drum_stems_file, drum_stems_wav)

        # Calculate the onsets for the WAV file
        onsets = calculate_onsets(drum_stems_wav)

        # Convert the onsets into a Disco-compatible keyframe string
        keyframe_string = create_key_frame_strings(analysis_folder, mp3_file, onsets, kfps)

        print(f"Keyframe string for {mp3_file.stem}: '{keyframe_string}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP3 files and generate Disco-compatible keyframe strings.")
    parser.add_argument("--input", type=Path, required=True, help="Input folder containing MP3 files.")
    parser.add_argument("--kfps", type=int, help="Keyframes per second used in deforum", default=3)
    args = parser.parse_args()

    main(args.input, args.kfps)
