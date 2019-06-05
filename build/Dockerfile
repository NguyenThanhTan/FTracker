FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update

RUN apt-get install -y build-essential python3-dev python3-pip libxml2 libxml2-dev zlib1g-dev

RUN apt-get install -y libxext6 libsm6 libxrender1 libfontconfig1 libglib2.0-0 ffmpeg

COPY ./requirements.txt ./requirements.txt

COPY ./requirements_2.txt ./requirements_2.txt

RUN pip3 --no-cache-dir install -r requirements.txt

RUN pip3 --no-cache-dir install -r requirements_2.txt

EXPOSE 5000

WORKDIR /workspace
