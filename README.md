# FTracker

This face tracker is proposed in a graduate thesis 
at HCM University of Science, Vietnam National University.

Full documents can be found at: (Upcoming...)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This project require installing of these tools:

- [Docker CE](https://docs.docker.com/install/) (~=18.09)
- [NVidia Docker driver](https://github.com/NVIDIA/nvidia-docker)

### Installing

This is a step by step guide to help you install this project

First, build the Docker image using this command

`docker build . -f Dockerfile --tag face`

Build a network dockernet

`docker network create -d bridge --subnet 192.168.0.0/24 --gateway 192.168.0.1 dockernet`

Second, start a Docker container using NVidia Docker driver,
replace your absolute path to cloned project with `{absolute path to face_service}`: 
i.e.,`'/home/face_service/:/workplace/`

`NV_GPU=3 nvidia-docker run -d  --name=face_dtm  -v {absolute path to face_service}:/workspace/ --net dockernet --entrypoint="tail" face:latest -f /dev/null`

Finally, you can access to the container:

`docker exec -it face_dtm /bin/bash`

Some demo:

1. Create `office3` folder: 
`mkdir data/videos/office3`

2. Decode and extract frames from video:
`ffmpeg -i {path_to_test_video} data/videos/office3/%05d.jpg -hide_banner`
. Decode with specific frames 
`ffmpeg -i in.mp4 -vf select='between(n\,55111\,55800)' -vsync 0 -start_number 55111 data/videos/office3/%05d.jpg`

3. Run the test
`python3 main.py`


## Running the tests

(Upcoming...)

## Deployment

(Upcoming...)

## Built With

* Tensorflow
* OpenCV
* imageio

## Contributing

(Upcoming...)

## Versioning

(Upcoming...) 

## Authors

* **Nguyen Thanh Tan** - [NguyenThanhTan](https://github.com/NguyenThanhTan)
* **Vong Chi Tai** - [vchitai](https://github.com/vchitai)


## License

(Upcoming...)

## Acknowledgments

(Upcoming...)
