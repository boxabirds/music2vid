import os
import re
import sys
import argparse
import openai
from openai.error import RateLimitError

from typing import Dict, List, Tuple
from pathlib import Path
import time

openai.api_key = os.environ.get('OPENAI_API_KEY')


def rate_limit_tolerant_openai_completion(prompt, temperature=0.7, max_tokens=1024, frequency_penalty=0.7):
    wait_time = 10
    #print(f"generating prompt for {prompt}:")
    while True:
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
            )
            break  # Exit the loop if no error occurs
        except RateLimitError as e:
            print(f"RateLimitError: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2
    response_text = response['choices'][0]['text'].strip(' \n"\'')
    return response_text

def read_lyrics_file(filepath: Path) -> Tuple[str,List[Tuple[int, str]]]:
    values = []
    lyrics = ""
    with open(filepath, 'r') as f:
        for line in f:
            number, text = line.strip().split(':')
            lyrics += text.strip() + '\n'
            values.append((int(number), text.strip()))
        
    return lyrics, values

def generate_general_style(lyrics: str) -> str:
    prompt = (
        "Pick one of these styles that best matches the sentiment of the lyrics below:" +
        "'origami, silhouette, watercolor, steam punk, candy town, neon, pencil sketch, mystic, bladerunner, leafy jungle, frozen, fire, underwater, art deco, industrial factory'.\n\n"
    )
    response_text = rate_limit_tolerant_openai_completion(prompt + lyrics, temperature=0.7, max_tokens=256, frequency_penalty=0.5)
    response_text = response_text.splitlines()[-1]
    print(f"generate_general_style result: '{response_text}'")
    return response_text

def generate_overall_song_prompt(lyrics: str) -> str:
    prompt = (
        "To create a uniquely visual illustrative style for the song lyrics below, "
        "please come up with a text prompt of no more than 25 words that can be used for AI image generation tools "
        "that best matches the mood and sentiment of the song lyrics below."
        "Exclude any specific actions like 'text prompt' just provide the visual imagery without quotes.\n\n"
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt + lyrics,
        temperature=0.7,
        max_tokens=256,
        frequency_penalty=0.5,
    )
    response_text = rate_limit_tolerant_openai_completion(prompt + lyrics, temperature=0.7, max_tokens=256, frequency_penalty=0.5)
    response_text = response_text.splitlines()[-1]
    print(f"generate_overall_song_prompt: result: '{response_text}'")
    return response_text


def generate_visual_prompts(full_lyrics: str, lines:List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    prompt = """
        "Below is a single line from the lyrics of a song, followed by the full lyrics.  We want to create a supporting visual for this single line.
        Using all lyrics as context for meaning, generate a corresponding vivid visual prompt that
        can be used to communicate the meaning of the single line visually. Do not prefix with any text like 'visual prompt' or 'vivid visual prompt'.
         \n\n"
        """

    visual_prompts = []

    # iterate over the lines and generate a prompt for each line
    for i,line in enumerate(lines): # line[1] is the text
        generated_line = rate_limit_tolerant_openai_completion(prompt + line[1] + full_lyrics, temperature=0.7, max_tokens=1024, frequency_penalty=0.1)
        # remove all but the last line of generated text to remove any extraneous guff OpenAI sends our way
        # TODO we may need to employ other heuristics like "keep only the longest of these lines etc"
        generated_line = generated_line.splitlines()[-1]

        print(f"generate_visual_prompt for '{line[1]}': result: \n'{generated_line}'\n")
        visual_prompts.append((lines[i][0], generated_line))

    return visual_prompts


def write_visual_prompts_file(output_file: Path, lines: List[Tuple[int, str]], song_prompt:str, general_style:str):
    print(f"Writing visual prompts to '{output_file}':\n{lines}")
    # TODO write to json for later reading
    output_format = "python"
    with open(output_file, "w") as f:
        if output_format == "python":
            f.write(f'"style": "{general_style}",\n')
            f.write(f'0: "{song_prompt}",\n')
            # iterate through the dict and write each line to the file
            for frame, visual in lines:
                f.write(f'{frame}: "{visual}",\n')


def process_song(song_file: Path):
    analysis_folder = Path(song_file.parent / f"{song_file.stem}-analysis")
    lyrics_file = analysis_folder / Path(song_file.stem + "-lyrics-medium.txt")
    output_file = analysis_folder / Path(song_file.stem + '-visual-prompts.json')

    if not os.path.exists(lyrics_file):
        print(f"Lyrics file not found for '{song_file}'. Was looking for '{lyrics_file}'. Skipping.")
        return
    else:
        print(f"Processing '{song_file}'")

    # do nothing if the output file already exists
    if os.path.exists(output_file):
        print(f"Visual prompts file already exists for {song_file}. Skipping as it hits the chargeable OpenAI API. Delete if you want it again")
        return

    print(f"Reading lyrics from {lyrics_file}.")
    full_lyrics, lines = read_lyrics_file(lyrics_file)
    general_style = generate_general_style(full_lyrics)
    overall_song_prompt = generate_overall_song_prompt(full_lyrics)
    visual_prompts = generate_visual_prompts(full_lyrics, lines)

    write_visual_prompts_file(output_file, visual_prompts, overall_song_prompt, general_style)
    print(f"Visual prompts for {song_file} have been written to {output_file}.")

def find_all_mp3_files(folder: Path) -> List[Path]:
    mp3_files = [file for file in folder.glob('**/*.mp3')]
    return mp3_files

def main():
    parser = argparse.ArgumentParser(description='Process MP3 file metadata and generate visual prompts.')
    parser.add_argument('--input', required=True, help='Path to an MP3 file or a folder containing MP3 files that have a corresponding analysis folder.')
    args = parser.parse_args()

    input_path = Path(args.input)

    # process single file
    if input_path.is_file() and input_path.suffix == '.mp3':
        process_song(Path(args.input))

    # process all mp3 files in a folder
    elif input_path.is_dir():
        mp3_files = find_all_mp3_files(Path(args.input))
        for mp3_file in mp3_files:
            process_song(mp3_file)
    else:
        print(f"Invalid input: {args.input}. Please provide either an MP3 file or a folder containing MP3 files.")

if __name__ == '__main__':
    main()
