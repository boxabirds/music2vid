import os
import whisper_timestamped as whisper
import argparse
import demucs.api
import demucs.audio
import json
from pathlib import Path

#import soundfile as sf

parser = argparse.ArgumentParser()
KEY_FRAME_FPS = 4 # how many key frames are generated per second
MODEL_SIZE = "base" # from openai-whisper: tiny, base, small, medium, large https://github.com/openai/whisper

# function for transcribing audio file

def stem_path_from_source_path(source_audio_path: Path, stem: str) -> Path:
    return Path(source_audio_path.stem + "_" + stem + ".mp3")

def separate_vocals(audio_file:str) -> Path: 
    # don't do it again if the stems already exist
    source_audio_path = Path(audio_file)
    vocal_stem = stem_path_from_source_path(source_audio_path, "vocals")

    if os.path.exists(vocal_stem):
        print(f"Stems already exist for '{vocal_stem}' so no stem separation will be performed -- delete the stems if you want to re-run the separation")
        return vocal_stem
    
    stem_paths = {}
    separator = demucs.api.Separator()
    separator.load_model(model='mdx_extra')
    separator.load_audios_to_model(source_audio_path)
    separated = separator.separate_loaded_audio()
    for file, sources in separated:
        print(f"Saving file '{file}'")
        for stem, source in sources.items():
            stem_path = source_audio_path.stem + "_" + stem + ".mp3"
            stem_paths[stem] = stem_path
            print(f"Saving stem file '{stem_path}'")
            demucs.audio.save_audio(source, stem_path, samplerate=separator._samplerate)
    
    # TODO fix this because random folder
    return stem_paths["vocals"]

def transcribe_audio(audio_file):
    audio = whisper.load_audio(audio_file)
    model = whisper.load_model(MODEL_SIZE, device="cpu")
    result = whisper.transcribe_timestamped(model, audio, language="en", vad=True, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    return result



# use argparse take input from user from parameter "--input" and output to optional parameter "--output"


# Add the input argument
parser.add_argument("--input", help="Input file path", required=True)

# Parse the arguments
args = parser.parse_args()

input_filename = args.input

# TODO add this in when demucs has rudimentary API support https://github.com/facebookresearch/demucs/pull/474
#demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "track with space.mp3"])

# librosa's vocal stemming has so many artefacts that transcription quality is actually worse
# but we'll probably use librosa for beat detection
# y, sr = librosa.load(args.input)

# # Get harmonic component
# y_harmonic = librosa.effects.harmonic(y)

# librosa POC: write harmonic component to WAV file
# harmony_component_filename = args.input + "-harmonic.wav"
# sf.write(harmony_component_filename, y_harmonic, sr)
#subprocess.call(['ffmpeg', '-i', harmony_component_filename, 'your_file_harmonic.mp3'])

vocal_stem = separate_vocals(input_filename)
result = transcribe_audio(vocal_stem)


# if output is not specified, use the input file name with .json extension

full_transcript_filename = args.input + "-" + MODEL_SIZE + ".json"
# write the result to the output file
with open(full_transcript_filename, "w") as f:
    f.write(json.dumps(result, indent = 2, ensure_ascii = False))


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

segment_transcript_filename = args.input + "-" + MODEL_SIZE + "-segments.json"
with open(segment_transcript_filename, "w") as f:
    f.write(json.dumps(result_segments, indent = 2, ensure_ascii = False))

# write result_segments into a CSV file

import csv

csv_filename = args.input + "-" + MODEL_SIZE + "-segments.csv"
with open(csv_filename, "w") as f:
    writer = csv.writer(f)
    for segment in result_segments["segments"]:
        writer.writerow([segment["text"], segment["start"], segment["end"]])

# write the result as animation prompts
ap_filename = args.input +  "-animation-prompts-" + MODEL_SIZE + ".txt"
with open(ap_filename, "w") as f:
    for segment in result_segments["segments"]:
        frame_num = round(segment["start"] * KEY_FRAME_FPS)
        f.write(str(frame_num) + ": \"" + segment["text"] + "\"," + "\n")


# finally print out all the segments as one transcription
for segment in result_segments["segments"]:
    print(segment["text"])