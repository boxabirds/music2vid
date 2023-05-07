import argparse
import os
import subprocess
import imageio_ffmpeg as ffmpeg
import imageio.v3 as iio

def change_frame_rate_ffmpeg(input_video_path, output_video_path, output_frame_rate):
    cmd = [
        ffmpeg.get_ffmpeg_exe(),
        "-i",
        input_video_path,
        "-vf",
        f"fps={output_frame_rate}",
        "-y",
        output_video_path,
    ]

    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with return code {process.returncode} and stderr:\n{process.stderr.decode('utf-8')}")

def main():
    parser = argparse.ArgumentParser(description="Change the frame rate of a video.")
    parser.add_argument("--input", required=True, help="Path to the input video file.")
    parser.add_argument("--fps", type=int, required=True, help="New frame rate.")
    parser.add_argument("--output", help="Optional path for the output video file.")

    args = parser.parse_args()

    input_video_path = args.input
    output_frame_rate = args.fps

    if args.output:
        output_video_path = args.output
    else:
        input_name, input_ext = os.path.splitext(input_video_path)
        output_video_path = f"{input_name}-{output_frame_rate}-fps{input_ext}"

    change_frame_rate_ffmpeg(input_video_path, output_video_path, output_frame_rate)

if __name__ == "__main__":
    main()
