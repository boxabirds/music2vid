#!/bin/bash

source_video=$1
frame_number=$2
output_filename="${source_video%.*}-${frame_number}.jpg"

echo "Source video: $source_video"
echo "Frame number: $frame_number"
echo "Output filename: $output_filename"

ffmpeg -i "$source_video" -vf "select='eq(n\,$frame_number)'" -vframes 1 "$output_filename"
