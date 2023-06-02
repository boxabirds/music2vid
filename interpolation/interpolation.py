# requires mediapy, numpy and tensorflow_hub
# portions of this code thanks to https://www.tensorflow.org/hub/tutorials/tf_hub_film_example

import argparse
import glob
import time
from datetime import datetime
import requests
import os
from typing import Generator, Iterable, List, Optional
import ffmpeg
import re

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import mediapy as media
from moviepy.editor import concatenate_videoclips, VideoFileClip
import json
from pathlib import Path

from tqdm import tqdm



_UINT8_MAX_F = float(np.iinfo(np.uint8).max)


METADATA_FILE_STEM = "metadata"
METADATA_PROPERTY = "average_frame_render_time"
FPS = 24

render_data: List[float] = []

def store_render_time(result_path):
  average_time = get_average_render_time()
  data = {METADATA_PROPERTY: average_time}
  metadata_file_path = Path(result_path) / (METADATA_FILE_STEM + "-" + datetime.now().strftime("%H%M%S") + ".json")
  #print(f"Storing average render time {average_time} in {metadata_file_path}")

  with metadata_file_path.open("w") as outfile:
      json.dump(data, outfile)


def add_render_result(new_result:float):
  #print(f"Adding render result: {new_result}")
  render_data.append(new_result)

def get_average_render_time():
    length = len(render_data)
    return sum(render_data) / len(render_data) if length > 0 else 0


def load_image(image_path: Path):
  """Returns an image with shape [height, width, num_channels], with pixels in [0..1] range, and type np.float32."""
  image_data = tf.io.read_file(str(image_path))

  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  return image_numpy / _UINT8_MAX_F


def _pad_to_align(x, align):
  """Pads image batch x so width and height divide by align.

  Args:
    x: Image batch to align.
    align: Number to align to.

  Returns:
    1) An image padded so width % align == 0 and height % align == 0.
    2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
      to undo the padding.
  """
  # Input checking.
  assert np.ndim(x) == 4
  assert align > 0, 'align must be a positive number.'

  height, width = x.shape[-3:-1]
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop


class Interpolator:
  """A class for generating interpolated frames between two input frames.

  Uses the Film model from TFHub
  """

  def __init__(self, align: int = 64) -> None:
    """Loads a saved model.

    Args:
      align: 'If >1, pad the input size so it divides with this before
        inference.'
    """
    self._model = hub.load("https://tfhub.dev/google/film/1")
    self._align = align

  def __call__(self, x0: np.ndarray, x1: np.ndarray,
               dt: np.ndarray) -> np.ndarray:
    """Generates an interpolated frame between given two batches of frames.

    All inputs should be np.float32 datatype.

    Args:
      x0: First image batch. Dimensions: (batch_size, height, width, channels)
      x1: Second image batch. Dimensions: (batch_size, height, width, channels)
      dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

    Returns:
      The result with dimensions (batch_size, height, width, channels).
    """
    if self._align is not None:
      x0, bbox_to_crop = _pad_to_align(x0, self._align)
      x1, _ = _pad_to_align(x1, self._align)

    inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}

    start_time = time.monotonic()

    ### INNER LOOP
    result = self._model(inputs, training=False)
    ###

    end_time = time.monotonic()
    add_render_result(end_time - start_time)

    image = result['image']
    #print(f"image = {image}")

    if self._align is not None:
      #print(f"bbox_to_crop = {bbox_to_crop}")
      image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    return image.numpy()
  
def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: Interpolator) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(
        np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator)


def interpolate_recursively(
    frame_paths: List[Path], num_recursions: int,
    interpolator: Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Args:
    frame_filenames: List of input frame filenames. The colors should be
      in the range[0, 1] and in gamma space.
    num_recursions: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frame_paths)
  for i in tqdm(range(1, n), desc="Processing frames", unit="frames"):
      frame1 = load_image(frame_paths[i - 1])
      frame2 = load_image(frame_paths[i])
      yield from _recursive_generator(frame1, frame2,
                                      num_recursions, interpolator)
  # Separately yield the final frame.
  yield load_image(frame_paths[-1])

def concatenate_videos(batch_video_paths:List[Path], output_file_path:Path):
    video_clips = [VideoFileClip(str(video)) for video in batch_video_paths]
    final_video = concatenate_videoclips(video_clips)
    final_video.write_videofile(str(output_file_path))

def generate_video_batches(keyframe_paths:List[Path], batch_dest_dir_path:Path, batch_size:int, recursion_depth:int) ->List[Path]:
    # for better resilience and memory efficiency we render the frames in batches and stitch them together afterwards
    # keyframe_paths: a list of all the keyframes to be interpolated
    # combination_path: 
    interpolator = Interpolator()
    total_frames = len(keyframe_paths)
    num_batches = (total_frames - 1) // batch_size + 1
    print(f"Generating {num_batches} batches of {batch_size} frames each")
    batch_video_paths:List[Path] = []

    for batch_idx in range(num_batches):
        batch_video_path = create_batch_video_path(total_frames, batch_dest_dir_path, batch_size, batch_idx)
        batch_keyframe_paths:List[Path] = []

        if os.path.exists(batch_video_path):
           print(f"Skipping generation of interim '{batch_video_path}' because it already exists. Delete it if you want to regenerate it.")
        else:
          start_idx = batch_idx * batch_size
          end_idx = min((batch_idx + 1) * batch_size, total_frames - 1)
          if batch_idx > 0:
              start_idx -= 1

          # get our slice of the frames and do interpolation of them. 
          batch_keyframe_paths = keyframe_paths[start_idx:end_idx]
          frames = list(interpolate_recursively(batch_keyframe_paths, recursion_depth, interpolator))

          # Each cycle adds the keyframe + x interpolation frames + keyframe. 
          # For batches to be stitched together, we need to remove the first frame of each batch except the first one.
          if batch_idx > 0:
              frames = frames[1:]  
          
          print(f'Creating {batch_video_path} with {len(frames)} frames')
          media.write_video(batch_video_path, frames, fps=FPS)
        batch_video_paths.append(batch_video_path)
    
    return batch_video_paths

def create_batch_video_path(total_frames, batch_dest_dir_path:Path, batch_size, batch_idx) -> Path:
    start_frame = batch_idx * batch_size
    end_frame = min((batch_idx + 1) * batch_size, total_frames)
    start_frame_padded = str(start_frame).zfill(5)
    end_frame_padded = str(end_frame).zfill(5)
    batch_video_path = batch_dest_dir_path / f"batch-{start_frame_padded}-{end_frame_padded}-{FPS}fps.mp4"
    return batch_video_path


def add_audio_to_video(input_video_path:Path, input_audio_path:Path, output_video_path:Path):
    input_video_str = str(input_video_path)
    input_audio_str = str(input_audio_path)
    output_video_str = str(output_video_path)
    print(f"Adding audio from '{input_audio_str}' to '{input_video_str}' and saving to '{output_video_str}'")

    # Extract the duration of the input video
    input_video_duration = float(ffmpeg.probe(input_video_str)['streams'][0]['duration'])

    # Concatenate the video and audio streams
    input_video_stream = ffmpeg.input(input_video_str)
    input_audio_stream = ffmpeg.input(input_audio_str)

    audio_stream = input_audio_stream.audio
    video_stream = input_video_stream.video

    # Limit the output audio duration to the input video duration
    audio_stream = audio_stream.filter("atrim", duration=input_video_duration)

    # Combine the video and audio streams
    output_stream = ffmpeg.concat(video_stream, audio_stream, v=1, a=1).output(output_video_str)
    
    # Run the FFmpeg command
    output_stream.run()



def get_combination_paths_with_seed(composition_path:Path, seed_used)->List[Path]:
  combinations_path = sorted([combination_path for combination_path in composition_path.iterdir() if combination_path.is_dir() and f"seed={seed_used}-" in combination_path.name])    
  return combinations_path


def generate_interpolated_video_combination( composition_dir_name:str, combination_path:Path, mp3:Path, batch_size:int, recursion_depth:int ):
    output_file_path = combination_path / f"{composition_dir_name}-interpolated.mp4"
    output_file_with_audio_path = combination_path / f"{composition_dir_name}-interpolated-with-audio.mp4"
    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists, skipping. Delete it if you want to regenerate it.")
    
    # get a list of all the keyframes in the combination and generate video batches for them
    else:
      keyframe_paths:List[Path] = sorted(combination_path.glob("*.png"))
      print(f"Found {len(keyframe_paths)} keyframes in {combination_path}")
      batch_file_paths = generate_video_batches( keyframe_paths, combination_path, batch_size, recursion_depth )
      store_render_time( combination_path )
      concatenate_videos( batch_file_paths, output_file_path )
      add_audio_to_video( output_file_path, mp3, output_file_with_audio_path )
 

def generate_interpolated_video_combinations( batch_config_path:Path, compositions_path:Path, mp3_dir_path:Path, seed_used: int, batch_size:int, recursion_depth:int, key_fps:int ):   
    ### TODO some of this code is shared between run_batches and here
    print(f"Generating interpolated videos for {batch_config_path} using seed {seed_used} batch size {batch_size}")
    with open(batch_config_path, "r") as file:
      batch_config = json.load(file)

    # pull data from the batch settings file. We only process the compositions listed in the batch settings file
    compositions = batch_config["compositions"]

    print(f"Found {len(compositions)} compositions in {batch_config_path}")
    for composition_file_name in compositions:
      # this is where we're expecting each of the combination folders containing keyframes that have been generated
      composition_dir_name = Path(composition_file_name).stem #e.g. 'Wandering Eye' -- note the removal of .mp3 with .stem
      composition_path = Path(compositions_path / composition_dir_name) # e.g. 'outputs/Wandering Eye'
      combination_paths = get_combination_paths_with_seed(composition_path, seed_used) # e.g. '[outputs/Wandering Eye/seed=1234-sampler=dpm…, outputs/Wandering Eye/seed=1234-sampler=euler…]'
      print(f"Found {len(combination_paths)} combinations for {composition_dir_name}")
      mp3_path = Path(mp3_dir_path / composition_file_name) # e.g. 'mp3/Wandering Eye.mp3' 
      for combination_path in combination_paths:
        print(f"Generating interpolated video for {combination_path} with audio from {mp3_path}")
        generate_interpolated_video_combination( composition_dir_name, combination_path, mp3_path, batch_size, recursion_depth )


def exponential_decay_interpolation( image_dest_dir_path: Path, first_image_path: Path, last_image_path:Path, num_interpolations:int):
  # take two images and generate a series of interpolations between them
  # for exponential decay interpolation, the interpolation is always against the most recently interpolated image and the last image
  pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-config', type=str, required=True, help='the configuration used to generate the images contains a list of compositions to process. E.g. "full-length-dpm-0.6.json"')
    parser.add_argument('--compositions-dir', type=str, required=True, help="(typically 'outputs'): a list of directories, one for each composition, with the composition name.")
    parser.add_argument('--mp3-dir', type=str, required=True, help="Location of original audio files to add to videos. File names should match those listed in the batch config")
    parser.add_argument("--seed-used", type=int, required=True, help="Which seed was used for this set -- only process those")
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--recursion-depth', type=int, default=3)
    parser.add_argument('--key-fps', type=int, default=3)
    args = parser.parse_args()
    batch_config_path = Path(args.batch_config)
    mp3_path = Path(args.mp3_dir)
    compositions_path = Path(args.compositions_dir)
    recursion_depth = args.recursion_depth
    key_fps = args.key_fps
    batch_size = args.batch_size
    seed_used = args.seed_used

    # we're assuming this structure:
    # - each compositions_path directory contains a further list of directories, one for each combination, with the combination name, prefixed with "seed=xxx"
    # - those combination directories contain the actual frames for a specific composition combination. We do our good work there. 
    generate_interpolated_video_combinations( batch_config_path, compositions_path, mp3_path, seed_used, batch_size, recursion_depth, key_fps )
