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


def load_image(img_url: str):
  #print(f"load_image: {img_url}")
  """Returns an image with shape [height, width, num_channels], with pixels in [0..1] range, and type np.float32."""

  if (img_url.startswith("https")):
    user_agent = {'User-agent': 'Colab Sample (https://tensorflow.org)'}
    response = requests.get(img_url, headers=user_agent)
    image_data = response.content
  else:
    image_data = tf.io.read_file(img_url)
    #print(f"image_data for '{img_url}': {image_data}")

  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  #print(f"image_numpy for '{img_url}': {image_numpy}")
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
    frame_filenames: List[str], num_recursions: int,
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
  n = len(frame_filenames)
  for i in tqdm(range(1, n), desc="Processing frames", unit="frames"):
      frame1 = load_image(frame_filenames[i - 1])
      frame2 = load_image(frame_filenames[i])
      yield from _recursive_generator(frame1, frame2,
                                      times_to_interpolate, interpolator)
  # Separately yield the final frame.
  yield load_image(frame_filenames[-1])



def concatenate_videos(video_filenames, output_filename):
    video_clips = [VideoFileClip(video) for video in video_filenames]
    final_video = concatenate_videoclips(video_clips)
    final_video.write_videofile(output_filename)

# def generate_video_batches(frame_filenames, recursion_depth, framedir, batch_size):
#     interpolator = Interpolator()
#     num_batches = (len(frame_filenames) - 1) // batch_size + 1
#     print(f"Generating {num_batches} batches of {batch_size} frames each")
#     batch_filenames = []

#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size - 1, len(frame_filenames) - 1)
#         batch_frame_filenames = frame_filenames[start_idx:end_idx + 1]
#         frames = list(interpolate_recursively(batch_frame_filenames, recursion_depth, interpolator))
#         fps = 24
#         batch_movie_filename = framedir + f"-batch{batch_idx}-{fps}fps.mp4"
#         print(f'Creating {batch_movie_filename} with {len(frames)} frames')
#         media.write_video(batch_movie_filename, frames, fps=fps)
#         batch_filenames.append(batch_movie_filename)
    
#     return batch_filenames

def generate_video_batches(frame_filenames, recursion_depth, framedir, batch_size):
    interpolator = Interpolator()
    num_batches = (len(frame_filenames) - 1) // batch_size + 1
    print(f"Generating {num_batches} batches of {batch_size} frames each")
    batch_filenames = []

    for batch_idx in range(num_batches):
        batch_movie_filename = framedir + f"-batch{batch_idx}-{FPS}fps.mp4"
        if os.path.exists(batch_movie_filename):
           print(f"Skipping generation of interim '{batch_movie_filename}' because it already exists. Delete it if you want to regenerate it.")
        else:
          start_idx = batch_idx * batch_size
          end_idx = min((batch_idx + 1) * batch_size, len(frame_filenames) - 1)
          if batch_idx > 0:
              start_idx -= 1
          batch_frame_filenames = frame_filenames[start_idx:end_idx]
          frames = list(interpolate_recursively(batch_frame_filenames, recursion_depth, interpolator))
          if batch_idx > 0:
              frames = frames[1:]  # Exclude the first frame if it's not the first batch
          
          print(f'Creating {batch_movie_filename} with {len(frames)} frames')
          media.write_video(batch_movie_filename, frames, fps=FPS)
        batch_filenames.append(batch_movie_filename)
    
    return batch_filenames


def add_audio_to_video(input_video, input_audio, output_video):
    input_video_stream = ffmpeg.input(input_video)
    input_audio_stream = ffmpeg.input(input_audio)

    audio_stream = input_audio_stream.audio
    video_stream = input_video_stream.video

    output_stream = ffmpeg.concat(video_stream, audio_stream, v=1, a=1).output(output_video)
    output_stream.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--composition-frames-dir', type=str, required=True)
    parser.add_argument('--mp3', type=str, required=True)
    parser.add_argument('--times-to-interpolate', type=int, default=3)
    parser.add_argument('--key-fps', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()
    keyframe_dir = args.composition_frames_dir
    mp3 = args.mp3
    times_to_interpolate = args.times_to_interpolate
    key_fps = args.key_fps
    batch_size = args.batch_size

    output_filename = keyframe_dir + "-interpolated.mp4"
    output_filename_with_audio = keyframe_dir + "-interpolated-with-audio.mp4"
    if os.path.exists(output_filename):
        print(f"Output file {output_filename} already exists, skipping")
    else:
      filenames = sorted(glob.glob(f"{keyframe_dir}/*.png"))
      print(f"Found {len(filenames)} keyframes")
      batch_filenames = generate_video_batches(filenames, times_to_interpolate, keyframe_dir, batch_size)
      store_render_time( keyframe_dir)
      concatenate_videos(batch_filenames, output_filename)
      add_audio_to_video(output_filename, mp3, output_filename_with_audio)
