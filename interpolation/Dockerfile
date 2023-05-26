# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    software-properties-common \
    ffmpeg

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

EXPOSE 5000
CMD ["flask", "run"]
