import argparse
import json
import math
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2


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


import re

def parse_disco(text: str) -> dict[int, float]:
    # parses animation strings of this format "0: (1.03), 30: (1.40), 31: (1.20), 32: (1.10), 33: (1.03), 52: (1.40), 53: (1.40), 54: (1.40), 55: (1.40), 56: (1.20), 57: (1.10), 58: (1.03), 71: (1.40), 72: (1.20), 73: (1.10), 74: (1.03), 77: (1.40),"
    result = {}
    pattern = re.compile(r"(\d+):\s\((\d+\.\d+)\)")

    for match in pattern.finditer(text):
        key = int(match.group(1))
        value = float(match.group(2))
        result[key] = value

    return result


KFPS = 3

def main(in_video, metadata, out_video, preview_duration=None, include_frame_count=False):
    reader = imageio.get_reader(in_video)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(out_video, fps=fps)

    with open(metadata, 'r') as f:
        data = json.load(f)
        captions_data = data["full_transcript"]["segments"]
        zoom_data = parse_disco(data["animation"]["keyframe_zoom_animations"])

    num_frames = reader.count_frames()
    print(f"Processing {num_frames} frames...")
    current_zoom_keyframe = 0
    for i, frame in tqdm(enumerate(reader), total=num_frames):
        frame_time = i / fps

        # track the zoom keyframe
        keyframe_num = math.trunc(frame_time * KFPS)
        if keyframe_num in zoom_data:
            current_zoom_keyframe = keyframe_num
        zoom_text = f"{current_zoom_keyframe}: ({zoom_data[current_zoom_keyframe]})"
        frame = add_text_to_frame(frame, zoom_text, align="top", text_size=18)

        for item in captions_data:
            start_time = item['start']
            end_time = item['end']
            if start_time <= frame_time < end_time:
                lyric_text = f"{keyframe_num}: {item['text']}"
                frame = add_text_to_frame(frame, lyric_text, align="bottom", text_size=14)
                break

        if preview_duration is not None and frame_time >= preview_duration:
            break

        writer.append_data(frame)


    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-video", help="Input video file")
    parser.add_argument("--metadata", help="Metadata file from audio analysis e.g 'full-metadata.json'")
    parser.add_argument("--out-video", help="Output video file")
    parser.add_argument("--preview", type=int, help="Preview duration in seconds (optional)")

    args = parser.parse_args()

    in_video = args.in_video
    metadata = args.metadata
    if args.preview:
        out_video = f"{in_video.split('.')[0]}-{args.preview}s-preview.mp4"
    else:
        out_video = args.out_video
    preview_duration = args.preview

    main(in_video, metadata, out_video, preview_duration)
