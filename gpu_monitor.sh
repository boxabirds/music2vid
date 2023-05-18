#!/bin/bash

EMAIL="julian.harris@gmail.com"
THRESHOLD=80
DURATION=10

while true; do
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=\$1} END {print sum/NR}')
  if (( $(echo "$GPU_UTIL < $THRESHOLD" | bc -l) )); then
    sleep $DURATION
    GPU_UTIL_AGAIN=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=\$1} END {print sum/NR}')
    if (( $(echo "$GPU_UTIL_AGAIN < $THRESHOLD" | bc -l) )); then
      echo "GPU utilization dropped below 80% for more than 10 seconds" | mail -s "GPU Utilization Alert" $EMAIL
    fi
  fi
  sleep 5
done

