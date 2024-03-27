docker build -t mmdet .
docker run --gpus all --ipc=host -it -v /mnt/ssd8/ola/aicity:/aicity mmdet
