import argparse
import json
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


from PIL import ImageDraw, ImageFont

from PIL import ImageDraw, ImageFont

from PIL import ImageDraw, ImageFont

from PIL import ImageDraw, ImageFont

def create_text_clip(text, frame_shape):
    # Create an image with the same dimensions as the video frame
    img = Image.new('RGBA', (frame_shape[1], frame_shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

    # Calculate the text size with the black border
    padding = 20
    text_width, text_height = font.getsize(text)

    # Resize the text to fit within the frame while maintaining the aspect ratio
    aspect_ratio = float(text_height) / float(text_width)
    frame_height = frame_shape[0]
    frame_width = frame_shape[1]
    new_width = int(frame_width * aspect_ratio)
    new_height = int(frame_height * aspect_ratio)

    if new_height > frame_height:
        new_height = frame_height
        new_width = int(frame_height * aspect_ratio)
    elif new_width > frame_width:
        new_width = frame_width
        new_height = int(frame_width / aspect_ratio)

    # Calculate the position to center the text within the frame
    x = (frame_width - new_width) // 2
    y = (frame_height - new_height) // 2

    # Draw the text with the black border
    draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill='black')

    # Return the image as a numpy array
    return np.array(img)



def main(in_video, metadata, out_video, preview_duration=None, include_frame_count=False):
    reader = imageio.get_reader(in_video)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(out_video, fps=fps)

    with open(metadata, 'r') as f:
        captions_data = json.load(f)["full_transcript"]["segments"]

    for i, frame in tqdm(enumerate(reader), total=reader.count_frames()):
        frame_time = i / fps
        frame_shape = frame.shape

        for item in captions_data:
            start_time = item['start']
            end_time = item['end']
            if start_time <= frame_time < end_time:
                text = item['text']
                text_clip = create_text_clip(text, frame_shape)
                frame = np.where(text_clip[..., 3, np.newaxis] == 255, text_clip[..., :3], frame)
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
