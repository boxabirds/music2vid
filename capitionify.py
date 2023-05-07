# for debugging -- adding captions and frame numbers and other stuff

import argparse
import json
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from tqdm import tqdm

def create_text_clip(text):
    return TextClip(
        txt=text,
        size=(400, 0),
        font="Arial",
        fontsize=24,
        color="black",
        bg_color="aqua"
    )

def create_frame_number_clip(frame_number, video_duration):
    return TextClip(
        txt=str(frame_number),
        size=(50, 0),
        font="Arial",
        fontsize=12,
        color="black",
        bg_color="white"
    ).set_start(frame_number / video_duration).set_duration(1 / video_duration)

def main(in_video, captions, out_video, preview_duration=None, include_frame_count=False):
    video = VideoFileClip(in_video)
    if preview_duration is not None:
        video = video.subclip(0, preview_duration)
    video_duration = video.duration
    video_fps = video.fps

    with open(captions, 'r') as f:
        captions_data = json.load(f)["segments"]

    for item in captions_data:
        start_time = item['start']
        text = item['text']
        end_time = item['end']
        duration = end_time - start_time
        if preview_duration is not None and start_time >= preview_duration:
            continue
        text_clip = (create_text_clip(text)
                     .set_start(start_time)
                     .set_position(("right", "bottom"))
                     .set_duration(duration))
        video = CompositeVideoClip([video, text_clip])

    if include_frame_count:
        frame_clips = [create_frame_number_clip(i, video_duration) for i in tqdm(range(int(video_duration * video_fps)), desc="Processing frames") if i / video_fps < video_duration]
        video_with_frames = CompositeVideoClip([video] + frame_clips)
    else:
        video_with_frames = video

    if preview_duration is not None:
        video_with_frames = video_with_frames.subclip(0, preview_duration)

    video_with_frames.write_videofile(out_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-video", help="Input video file", default="data/seaside-clip-video.mp4")
    parser.add_argument("--captions", help="JSON file containing time and captions", default="data/seaside-clip-long.mp3-base-segments.json")
    parser.add_argument("--out-video", help="Output video file", default="data/seaside-clip-video-captioned.mp4")
    parser.add_argument("--preview", type=int, help="Preview duration in seconds (optional)")
    parser.add_argument("--include-frame-count", type=bool, help="Include frame count (optional)", default=False")

    args = parser.parse_args()

    in_video = args.in_video
    captions = args.captions
    if args.preview:
        out_video = f"{in_video.split('.')[0]}-{args.preview}s-preview.mp4"
    else:
        out_video = args.out_video
    preview_duration = args.preview

    main(in_video, captions, out_video, preview_duration, args.include_frame_count)
