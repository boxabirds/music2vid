# TODO sort out source code in this project this is hacky
import sys
import os
import re
from tqdm import tqdm
import time
from PIL import Image, ImageDraw, ImageFont

from collections import namedtuple
Pos = namedtuple('Pos', ['x', 'y'])

from helpers.model_load import get_model_output_paths, load_model
from helpers.save_images import calculate_output_folder

# hack because the deforum source is sitting inside src for now
sys.path.extend([
    'src'
])
import json
import argparse
from lib.parameters import DeforumAnimArgs, Root, DeforumArgs
from pathlib import Path
from helpers.render import do_render, init_seed
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
import math


# This is a hard-coded value that all the music keyframes are based on so changing it will break things
KEYFRAMES_PER_SECOND = 3

# Interpolation frame logic will generate 7 interpolation frames if depth is 3 to make 23
INTERPOLATION_RECURSION_DEPTH = 3

# Calclated based on the hard-coded values above. 
# WARNING: changing this might result in a non-standard frame rate that then needs to be resampled to a standard
# frame rate for video playback (24, 25, 30, 60)
FPS = KEYFRAMES_PER_SECOND * 2^INTERPOLATION_RECURSION_DEPTH # = 24fps


METADATA_FILE_NAME = "metadata.json"
METADATA_PROPERTY = "average_frame_render_time"

def store_render_time(combination_path, average_frame_render_time):
    data = {METADATA_PROPERTY: average_frame_render_time}
    metadata_file_path = Path(combination_path) / METADATA_FILE_NAME

    with metadata_file_path.open("w") as outfile:
        json.dump(data, outfile)

def get_render_time(combination_path):
    metadata_file_path = Path(combination_path) / METADATA_FILE_NAME
    with metadata_file_path.open("r") as infile:
        data = json.load(infile)
    return data[METADATA_PROPERTY]


def cv2_to_pil(cv2_frame):
    return Image.fromarray(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB))

def draw_transparent_text(pil_image, text, margin, font, color, transparency=200, align='top'):
    # Create a blank image with the same size as the original
    blank_image = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    
    # Define the top-left position of the text
    if align == 'top':
        position = (margin, margin)
    elif align == 'bottom':
        # Calculate the text height and adjust the position accordingly
        text_width, text_height = font.getsize(text)
        position = (margin, pil_image.size[1] - text_height - margin)
    
    # Draw the text on the blank image
    draw = ImageDraw.Draw(blank_image)
    draw.text(position, text, font=font, fill=color + (transparency,))
    
    # Alpha composite the text image over the original image
    return Image.alpha_composite(pil_image.convert('RGBA'), blank_image)

def add_text_to_frame(cv2_frame:np.ndarray, text:str, margin:int = 20, align:str = 'top', text_size:int=28) -> np.ndarray:
    # Convert cv2 frame to PIL image
    pil_image = cv2_to_pil(cv2_frame)
    
    # Define the font and color
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", text_size)
    color = (255, 255, 255)

    # Draw the text on the PIL image
    pil_image_with_text = draw_transparent_text(pil_image, text, margin, font, color, align=align)
    
    # Convert the PIL image back to a cv2 frame
    return cv2.cvtColor(np.array(pil_image_with_text), cv2.COLOR_RGB2BGR)


def get_optimal_font_scale(text, width):
    # hack to fit the font into the width by iteratively scaling font down. 
    # no idea if this is the best approach but phind thought so so I'm ok for now
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale/10, thickness=2)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

def create_composition_path(parent_str:str, composition_name:str) -> Path:
    return Path(parent_str) / composition_name

def get_image_file_names_and_count(dir_path:Path) -> Tuple[Dict[str, List[str]], int]:
    # takes a directory and cycles through all immediate subdirectories. 
    # Creates a dictionary that maps the subdirectory name with a list of all png files in it. 
    image_dict = {}
    count:int = 0
    for subdir in dir_path.iterdir():
        if subdir.is_dir():
            png_files = [f for f in sorted(subdir.glob('*.png'))]
            if count == 0:
                count = len(png_files)
            image_dict[subdir.name] = png_files
    return (image_dict, count)

def generate_comparison_video(
        composition_name:str, 
        composition_path:Path, 
        seed:int, 
        prompts:Dict[int,str],
        tile_dimensions=(512, 512),
        fps=1):  
    # composition dir > combination dir > images (frames)
    # create a video that contains every combination of the composition for this batch, tiled for frame-by-fram comparison. 
    # This is useful for comparing the different combinations of a composition to see which one is best
    
    tile_dimensions = (int(tile_dimensions[0]), int(tile_dimensions[1]))
    combinations_path = sorted([combination_path for combination_path in composition_path.iterdir() if combination_path.is_dir() and f"seed={seed}-" in combination_path.name])

    # calculate tile dimensions. 
    # We have to be careful not to have too many tiles as this will cause the video to be too large to play back. 
    # 16 is probably as much as you want
    num_batches = len(combinations_path)
    #print(f"num_batches: {num_batches}")
    num_rows = math.ceil(math.sqrt(num_batches))
    num_cols = math.ceil((num_batches) / num_rows)

    video_dimensions = (tile_dimensions[0] * num_cols, tile_dimensions[1] * num_rows)
    #print(f"video_dimensions: {video_dimensions}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_dir = str(composition_path / f"{composition_name}-comparison-seed={seed}.mp4")
    out = cv2.VideoWriter(output_video_dir, fourcc, fps, video_dimensions)
    add_title_screen( out, composition_name, video_dimensions, fps)

    # # count the number of png images in the first directory inside combinations_path -- we can assume it's the same number in each
    (image_dict, num_frames) = get_image_file_names_and_count(composition_path)

    #print(f"num_frames: {num_frames}")
    current_prompt = prompts[0]

    for frame_idx in range(num_frames):
        tiled_frame = np.zeros((*video_dimensions, 3), dtype=np.uint8)

        for i, combination_dir in enumerate(combinations_path):
            #print( f"combination: {combination_dir.name}")
            images = image_dict[combination_dir.name]
            #print( f"images: '{images}'")

            if frame_idx < len(images):
                img = images[frame_idx]
                frame = cv2.imread(str(img))

                row = i // num_cols
                col = i % num_cols
                #print(f"placing image at row: {row}, col: {col}")

                # calculate the tile placement inside the video
                x_start = col * tile_dimensions[0]
                x_end = x_start + tile_dimensions[0]
                y_start = row * tile_dimensions[1]
                y_end = y_start + tile_dimensions[1]

                combination_name = strip_seed_prefix(combination_dir)
                average_render_time = get_render_time(combination_dir)
                #print(f"average_frame_render_time: {average_render_time}")
                frame = add_text_to_frame(frame, f"{combination_name}\nt={average_render_time:.2f}s")
                tiled_frame[y_start:y_end, x_start:x_end] = frame

        current_prompt = prompts.get(frame_idx, current_prompt)
        tiled_frame = add_text_to_frame(tiled_frame, current_prompt, align="bottom", text_size=14)
        out.write(tiled_frame)
    out.release()

FONT = cv2.FONT_HERSHEY_SIMPLEX


def add_text(frame:np.ndarray, text, pos:Pos, max_width):
    optimal_font_scale = get_optimal_font_scale(text, max_width)
    text_size, _ = cv2.getTextSize(text, FONT, optimal_font_scale, 2)

    # Draw the white text on top of the black outline
    cv2.putText(frame, text, (pos.x, pos.y), FONT, optimal_font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (pos.x, pos.y), FONT, optimal_font_scale, (255, 255, 255), 2, cv2.LINE_AA)

def strip_seed_prefix(combination_dir):
    # combination directories are named in this format "seed=1234-combination_name"
    return combination_dir.name.split('-', 1)[1]
def generate_showcase(
    composition_name:str, 
    composition_path:Path,
    seed:int,
    average_frame_render_times:Dict[str, float],
    dimensions=(512, 512),
    fps=1
    ):  
    FRAME_DURATION = 1 # seconds

    # make sure dimensions are ints
    dimensions = (int(dimensions[0]), int(dimensions[1]))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_dir = str(composition_path / f"{composition_name}-showcase-seed={seed}.mp4")
    #print(f"showcase output_video_path: {output_video_dir}")
    out = cv2.VideoWriter(output_video_dir, fourcc, fps, dimensions)

    # Add black screen with set name
    add_title_screen( out, composition_name, dimensions, fps)

    for combination_path in composition_path.iterdir():
        if combination_path.is_dir() and f"seed={seed}-" in combination_path.name:
            images = []

            # Add black screen with combination name excluding the seed prefix
            combination_name = combination_path.name.split('-', 1)[1]
            add_title_screen( out, combination_name, dimensions, fps )

            for img in sorted(combination_path.iterdir()):
                if img.suffix == ".png":
                    images.append(img)

            # Show each image in batch folder at the specified fps
            for j, image in enumerate(images):
                frame = cv2.imread(str(image))
                for i in range(FRAME_DURATION * fps):
                    out.write(frame)

    out.release()

def create_black_screen(dimensions):
    return np.zeros((*dimensions, 3), dtype=np.uint8)

def add_title_screen(video_writer:cv2.VideoWriter, text, dimensions, fps):
    # Generate a basic title screen with black background and white text
    TITLE_SCREEN_DURATION = 3  # seconds
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    
    optimal_font_scale = get_optimal_font_scale(text, dimensions[0])
    text_size, _ = cv2.getTextSize(text, font, optimal_font_scale, font_thickness)

    text_x = int((dimensions[0] - text_size[0]) / 2)
    text_y = int((dimensions[1] + text_size[1]) / 2)
    title_screen = create_black_screen(dimensions)
    cv2.rectangle(title_screen, (text_x - 20, text_y - text_size[1] - 20), (text_x + text_size[0] + 20, text_y + 20), (0, 0, 0), -1)
    cv2.putText(title_screen, text, (text_x, text_y), font, optimal_font_scale, (0, 0, 0), font_thickness*2, cv2.LINE_AA)
    cv2.putText(title_screen, text, (text_x, text_y), font, optimal_font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    for j in range(TITLE_SCREEN_DURATION * fps):  # Multiply by fps to ensure correct number of frames
        video_writer.write(title_screen)


def init_deforumargs(img:DeforumArgs, root:Root, motion:DeforumAnimArgs, combination_name: str, optional_initial_seed, style:str):
    # Additional properties of DeforumArgs that are visible in colab.
    # The DeforumArgs class is autogenerated from that, so we can't add them to the class definition.
    img.batch_name = combination_name
    img.general_style = style

    # initial_seed might not be set
    init_seed(img, optional_initial_seed)


    img.W, img.H = map(lambda x: x - x % 64, (img.W, img.H))  # resize to integer multiple of 64
    img.n_samples = 1 # type: ignore # doesnt do anything
    img.precision = 'autocast' 
    img.C = 4
    img.f = 8

    img.prompt = ""
    img.timestring = ""
    img.init_latent = None
    img.init_sample = None
    img.init_sample_raw = None
    img.mask_sample = None
    img.init_c = None
    img.seed_internal = 0

    img.dynamic_threshold = None
    img.static_threshold = None   

    img.strength_0_no_init = True

    img.outdir = calculate_output_folder(img.seed, root.output_path, img.batch_name)
    #print(f"### output folder: '{img.outdir}' using combination '{combination_name}'") # type: ignore

def create_movie_frames(img:DeforumArgs, motion:DeforumAnimArgs, root:Root, prompts:dict):
    do_render(img, motion, root, prompts)

def generate_combinations(properties, current_combination, index:int):
    if index == len(properties):
        yield current_combination
    else:
        prop_names, prop_values = properties[index]
        for value_set in prop_values:
            new_combination = current_combination.copy()
            if len(prop_names) == 1:  # Single property e.g. "img.sampler": ["dpm2", "heun",  "euler_ancestral"]
                prop_name = prop_names[0]
                if prop_name.startswith("img."):
                    img_prop_name = prop_name.split(".")[1]
                    if "img" not in new_combination:
                        new_combination["img"] = {}
                    new_combination["img"][img_prop_name] = value_set
                elif prop_name.startswith("motion."):
                    motion_prop_name = prop_name.split(".")[1]
                    if "motion" not in new_combination:
                        new_combination["motion"] = {}
                    new_combination["motion"][motion_prop_name] = value_set
                elif prop_name.startswith("root."):
                    root_prop_name = prop_name.split(".")[1]
                    if "root" not in new_combination:
                        new_combination["root"] = {}
                    new_combination["root"][root_prop_name] = value_set

            else:  # Multiple properties that only make sense together e.g. "img.H, img.W": [[512,512],[768,768]],
                img_dict = {}
                motion_dict = {}
                root_dict = {}
                for prop_name, value in zip(prop_names, value_set):
                    if prop_name.startswith("img."):
                        img_prop_name = prop_name.split(".")[1]
                        img_dict[img_prop_name] = value
                    elif prop_name.startswith("motion."):
                        motion_prop_name = prop_name.split(".")[1]
                        motion_dict[motion_prop_name] = value
                    elif prop_name.startswith("root."):
                        root_prop_name = prop_name.split(".")[1]
                        root_dict[root_prop_name] = value
                if img_dict:
                    if "img" not in new_combination:
                        new_combination["img"] = {}
                    new_combination["img"].update(img_dict)
                if motion_dict:
                    if "motion" not in new_combination:
                        new_combination["motion"] = {}
                    new_combination["motion"].update(motion_dict)
                if root_dict:
                    if "root" not in new_combination:
                        new_combination["root"] = {}
                    new_combination["root"].update(root_dict)
            yield from generate_combinations(properties, new_combination, index + 1)


_previous_model_checkpoint = None
_cached_model_tuple = None
def load_model_cached(root, load_on_run_all=True, check_sha256=True, map_location=None) -> Tuple[Any, Any]:
    global _previous_model_checkpoint
    global _cached_model_tuple

    model_checkpoint = root.model_checkpoint

    if model_checkpoint != _previous_model_checkpoint:
        print(f"### Cache: '{model_checkpoint}' not present, loading model")
        _cached_model_tuple = load_model(root, load_on_run_all=load_on_run_all, check_sha256=check_sha256, map_location=map_location)
        _previous_model_checkpoint = model_checkpoint
    # else:
    #     print(f"### Cache hit: using existing cached model '{model_checkpoint}'")

    return _cached_model_tuple

def init_rootargs(root:Root, composition_name:str, dry_run:bool=False ):
    # extend output path to include the name of the composition
    root.output_path = root.output_path + "/" + Path(composition_name).stem
    # TODO unclear function name. Should be called "ensure exists"
    root.models_path, root.output_path = get_model_output_paths(root)

    print(f"### init_rootargs output folder: '{root.output_path}'") 
    # TODO model and device are not input parameters in the original ipynb so didn't propagate to Root class in my export script
    if dry_run:
        root.model = None # type: ignore
        root.device = None # type: ignore
    else:
        root.model, root.device = load_model_cached(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location) 
        print(f"loaded model {root.model_checkpoint} on device {root.device}") # type: ignore

def extract_properties(combinations, dictionary_name):
    extracted_properties = []
    for keys, values in combinations.items():
        key_list = keys.split(",")
        prop_names = [key.strip() for key in key_list]
        if dictionary_name == "img":
            prop_names = [prop_name for prop_name in prop_names if prop_name.startswith("img.")]
        elif dictionary_name == "motion":
            prop_names = [prop_name for prop_name in prop_names if prop_name.startswith("motion.")]
        elif dictionary_name == "root":
            prop_names = [prop_name for prop_name in prop_names if prop_name.startswith("root.")]
        if prop_names:
            extracted_properties.append((prop_names, values))
    if not extracted_properties:
        print(f"No {dictionary_name} properties found")
    else:
        print(f"Extracted {dictionary_name} properties: {extracted_properties}")
    return extracted_properties

def generate_combination_name(root_combination, motion_combination, img_combination):
    root = str(root_combination.get('root', ''))
    motion = str(motion_combination.get('motion', ''))
    img = str(img_combination.get('img', ''))

    strings = [root, motion, img]
    filtered_strings = list(filter(lambda s: len(s) > 0, strings))

    raw_name = "&".join(filtered_strings)

    # Remove single quotes, curly braces, and spaces using list comprehension
    # so you get something like this "root:model_checkpoint:Protogen_V2.2.ckpt__"
    result =  "".join([c for c in raw_name if c not in "'.{} "])
    result = result.replace(":", "=")
    return result

EXTRA_END_FRAMES = 10
def calculate_max_frames(duration_in_seconds:float, num_keyframes_override:int) -> int:
    if num_keyframes_override is not None:
        return num_keyframes_override
    else:
        max_frames = int(duration_in_seconds * KEYFRAMES_PER_SECOND)
        return max_frames + EXTRA_END_FRAMES


def delete_extra_frames_from_dir(frame_dir: Path, max_frames: int, dry_run: bool = False):
    for file in sorted(frame_dir.glob("*.png")):
        match = re.match(rf"(\d+)_(\d+)\.png", file.name)
        if match:
            number, frame_num = match.groups()
            if int(frame_num) > max_frames:
                if not dry_run:
                    os.remove(file)
                    print(f"Deleted extra frame {file}")
                else:
                    print(f"### dry run: would have deleted extra frame {file}")
        else:
            print(f"skipping file {file} as it doesn't match the expected pattern")



parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file')
parser.add_argument("--mp3-dir", default="music", help="Location of MP3 files in config file.")
# add a flag to indicate whether to run the rendering or not
parser.add_argument("--dry-run", action="store_true", help="If set, don't render video")
parser.add_argument("--generate-showcases", action="store_true", help="If set, generates videos of all frames created together for comparison")
parser.add_argument("--initial-seed", type=int, nargs='?', default=None, help="Override random seed generation with a reproduceable starting point every run")

# temporary hack to delete extra frames -- might be useful in future for other utilities
# this batch script is rapidly becoming a general purpose pipeline batch tool.
parser.add_argument("--delete-extra-frames", action="store_true", default=None, help="cleanup function to remove any extra frames accidentally generated. ")

cli_args = parser.parse_args()
dry_run = cli_args.dry_run
input_file = cli_args.input
initial_seed = cli_args.initial_seed
delete_extra_frames = cli_args.delete_extra_frames

with open(input_file, "r") as file:
    batch_settings = json.load(file)

# pull data from the batch settings file
compositions = batch_settings["compositions"]
combinations = batch_settings["combinations"]

# we may want to restrict the number of frames from a specific item in the batch to do sampling
# sometimes we just want a fixed clip length for all the items in the batch
num_keyframes_override = batch_settings.get("num_keyframes_override", None)

print(f"Combinations: {combinations}")
img_properties = extract_properties(combinations, "img")
motion_properties = extract_properties(combinations, "motion")
root_properties = extract_properties(combinations, "root")

img_combinations = [{} for _ in range(len(img_properties))]
motion_combinations = [{} for _ in range(len(motion_properties))]
root_combinations = [{} for _ in range(len(root_properties))]

mp3_dir = cli_args.mp3_dir
combinations_generated = 0

# use for showcases
default_root:Root = Root()
default_img:DeforumArgs = DeforumArgs()
average_frame_render_time = {}

for composition_file_name in compositions:
    # pull metadata from analysis file
    composition_name = Path(composition_file_name).stem
    music_metadata_file = mp3_dir / Path(composition_name + "-analysis/full-metadata.json")
    music_metadata = music_metadata_file.read_text()
    # read analysis data into a json object 
    music_metadata = json.loads(music_metadata)

    zoom = music_metadata['animation']['keyframe_zoom_animations']
    # style is appended to the end of every prompt to present a consistent style
    style = music_metadata['style']
    prompts:Dict[int,str] = {}
    for key, value in music_metadata['keyframes'].items():
        prompts[int(key)] = value['prompt']

    duration = music_metadata['duration']
    max_frames = calculate_max_frames(duration, num_keyframes_override)

    for img_combination in generate_combinations(img_properties, {}, 0):
        for motion_combination in generate_combinations(motion_properties, {}, 0):
            for root_combination in generate_combinations(root_properties, {}, 0):
                root_instance = Root() # => "root"
                if "root" in root_combination:
                    for prop_name, value in root_combination["root"].items():
                        setattr(root_instance, prop_name, value)
                init_rootargs(root_instance, composition_file_name, dry_run=dry_run)
                
                anim_args = DeforumAnimArgs()  # => "motion"
                if "motion" in motion_combination:
                    for prop_name, value in motion_combination["motion"].items():
                        setattr(anim_args, prop_name, value)
                anim_args.max_frames = max_frames
                anim_args.zoom = zoom

                args = DeforumArgs() # => "img"
                if "img" in img_combination:
                    for prop_name, value in img_combination["img"].items():
                        setattr(args, prop_name, value)
                combination_name = generate_combination_name(root_combination, motion_combination, img_combination)
                init_deforumargs(args, root_instance, anim_args, combination_name, initial_seed, style)
                
                # we don't generate the same combination twice even with separate runs
                combination_path = Path(args.outdir) # type: ignore
                if combination_path.exists(): # type: ignore
                    if delete_extra_frames:
                        delete_extra_frames_from_dir(combination_path, max_frames, dry_run=dry_run)
                    else:
                        print(f"Skipping {composition_file_name} with combination '{combination_name}' because it already exists. Delete it to regenerate.")
                else:
                    if cli_args.dry_run:
                        print(f"### dry run: skipping rendering of {composition_file_name} with combination {combination_name} saved in {combination_path}")   
                    else:
                        combinations_generated += 1
                        combination_path.mkdir(parents=True) # type: ignore
                        start_time = time.monotonic()

                        ###
                        create_movie_frames(args, anim_args, root_instance, prompts)
                        ###

                        end_time = time.monotonic()
                        time_taken = end_time - start_time
                        average_frame_render_time[combination_name] = time_taken / anim_args.max_frames
                        print(f"Rendered {composition_file_name} with combination {combination_name} saved in {combination_path} in {time_taken:.2f} seconds averaging {average_frame_render_time[combination_name]:.2f}s per frame")   
                        store_render_time(combination_path, average_frame_render_time[combination_name]) 

    if cli_args.generate_showcases and not cli_args.dry_run and not cli_args.delete_extra_frames:
        composition_path = create_composition_path(default_root.output_path, composition_name)
        generate_showcase(
            composition_path=composition_path,
            composition_name=composition_name, 
            average_frame_render_times=average_frame_render_time,
            seed=initial_seed, 
            dimensions=(default_img.W, default_img.H)
        )
        generate_comparison_video(
            composition_path=composition_path, 
            composition_name=composition_name, 
            prompts=prompts,
            seed=initial_seed, 
            tile_dimensions=(default_img.W, default_img.H)
        )

print(f"{combinations_generated} combinations generated")
