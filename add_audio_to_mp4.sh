#!/bin/bash
# usage ./add_audio_to_mp4.sh --input-mp4 input.mp4 --input-mp3 audio.mp3

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input-mp4) input_mp4="$2"; shift ;;
        --input-mp3) input_mp3="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

output_mp4="${input_mp4%.mp4}-with-sound.mp4"

ffmpeg -i "$input_mp4" -i "$input_mp3" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "$output_mp4"
