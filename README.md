`docker build -f Dockerfile.cuda9.cudnn7 --tag face`

`NV_GPU=3 nvidia-docker run -d  --name=face_dtm  -v {absolute path to face_service}:/workspace/  --entrypoint="tail" face:latest -f /dev/null`

`docker exec -it face_dtm /bin/bash`

`mkdir data/videos/office3`

`ffmpeg -i {path_to_test_video} data/videos/office3/%05d.jpg -hide_banner`

`ffmpeg -i in.mp4 -vf select='between(n\,55111\,55800)' -vsync 0 -start_number 55111 data/videos/office3/%05d.jpg`

`python3 main.py`
