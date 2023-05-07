
import os
from typing import Dict
from tqdm import tqdm
import whisper_timestamped as whisper
import argparse
import demucs.api
import demucs.audio
import json
from pathlib import Path
import csv
import librosa


#import soundfile as sf

parser = argparse.ArgumentParser()
KEY_FRAME_FPS = 4 # how many key frames are generated per second
MODEL_SIZE = "base" # from openai-whisper: tiny, base, small, medium, large https://github.com/openai/whisper

# function for transcribing audio file

def stem_path_from_source_path(audio_file_path: Path, output_folder: Path, stem: str) -> Path:
    return Path(output_folder) / Path(audio_file_path.stem + "-" + stem + ".mp3")

def stems_dict_from_source_path(audio_file_path: Path, output_folder: Path) -> Dict[str, Path]:
    stems_dict = {}
    for stem in ["vocals", "drums", "bass", "other"]:
        stems_dict[stem] = stem_path_from_source_path(audio_file_path, output_folder, stem)
    return stems_dict


def separate_stems(audio_file_path: Path, output_folder:Path) -> Dict[str,Path]: 
    
    stems = stems_dict_from_source_path(audio_file_path, output_folder)
    if os.path.exists(stems["vocals"]):
        print(f"Stems already exist for '{stems['vocals']}' so no stem separation will be performed -- delete the stems if you want to re-run the separation")
        return stems
    else:
        print(f"Stems file '{stems['vocals']}' does not exist so stem separation will be performed")
    
    separator = demucs.api.Separator()
    print(f"Loading model")
    separator.load_model(model='mdx_extra')
    print(f"Loading audio into model")
    separator.load_audios_to_model(audio_file_path)
    print(f"Separating audio into stems")
    separated = separator.separate_loaded_audio()
    for file, sources in separated:
        for stem, source in sources.items():
            print(f"Saving stem file '{stems[stem]}'")
            demucs.audio.save_audio(source, stems[stem], samplerate=separator._samplerate)
    
    return stems

def transcribe_audio(audio_file):
    audio = whisper.load_audio(audio_file)
    model = whisper.load_model(MODEL_SIZE, device="cpu")

    # the vad, beam size, best of, temperature parameters are recommended to improve transcription quality
    result = whisper.transcribe_timestamped(model, audio, language="en", vad=True, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    return result


# Add the input argument
parser.add_argument("--input", help="Input file or folder", required=True)

# Parse the arguments
args = parser.parse_args()
input_path = Path(args.input)
if input_path.is_file():
    input_files = [str(input_path)]
elif input_path.is_dir():
    input_files = [str(file) for file in input_path.glob("*") if file.is_file()]
else:
    raise ValueError(f"Invalid input path: {input_path}")

# show progress using tqdm
for input_filename in tqdm(input_files, desc="Processing music files"):
    file_path = Path(input_filename)

    # filepath.stem is the filename without the extension not the audio stem :)
    output_folder = file_path.parent / (file_path.stem + "-analysis")
    plain_transcript_filename = Path(output_folder) / (file_path.stem +  "-plain-" + MODEL_SIZE + ".txt")
    # Skip processing if the analysis folder is not empty
    if os.path.exists(output_folder):
        if os.path.exists(plain_transcript_filename):
            tqdm.write(f"Skipping {file_path.name} because {plain_transcript_filename} already exists.")
            continue
    else:
        os.mkdir(output_folder)

    
    # tqdm_instance = tqdm(total=1, desc=f"Processing {input_filename}", leave=False)
    print(f"separating stems for '{file_path.name}'")
    stems = separate_stems(file_path, output_folder)

    # Calculate tempo using librosa.beat.tempo
    print(f"detecting tempo '{input_filename}'")
    y, sr = librosa.load(stems["drums"])
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)

    # Store tempo in metadata.json
    metadata = {"tempo": tempo[0]}
    metadata_filepath = os.path.join(output_folder, "metadata.json")
    with open(metadata_filepath, "w") as metadata_file:
        json.dump(metadata, metadata_file)

    print(f"transcribing '{stems['vocals'].name}'")
    result = transcribe_audio(stems["vocals"])

    print(f"writing transcription files '{file_path.name}'")
    # Update all the output file paths to use the output_folder
    full_transcript_filename = Path(output_folder) / (file_path.stem + "-" + MODEL_SIZE + ".json")
    # write the result to the output file
    with open(full_transcript_filename, "w") as f:
        f.write(json.dumps(result, indent = 2, ensure_ascii = False))

    segment_transcript_filename = Path(output_folder) / (file_path.stem + "-" + MODEL_SIZE + "-segments.json")
    # take the result and create a new object with just the segments with their start and end times
    result_segments = {
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in result["segments"]
        ]
    }
    with open(segment_transcript_filename, "w") as f:
        f.write(json.dumps(result_segments, indent = 2, ensure_ascii = False))

    csv_filename = Path(output_folder) / (file_path.stem + "-" + MODEL_SIZE + "-segments.csv")
    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        for segment in result_segments["segments"]:
            writer.writerow([segment["text"], segment["start"], segment["end"]])

    ap_filename = Path(output_folder) / (file_path.stem +  "-animation-prompts-" + MODEL_SIZE + ".txt")
    with open(ap_filename, "w") as f:
        for segment in result_segments["segments"]:
            frame_num = round(segment["start"] * KEY_FRAME_FPS)
            f.write(str(frame_num) + ": \"" + segment["text"] + "\"," + "\n")

    # finally the segments as one plain text transcription
    
    with open(plain_transcript_filename, "w") as f:
        for segment in result_segments["segments"]:
            f.write(segment["text"] + "\n")
    # tqdm_instance.update(1)
    # tqdm_instance.close()
