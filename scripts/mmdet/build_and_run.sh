docker build -t mmdet .
docker run --gpus all --ipc=host -it -v $(pwd)/../..:/aicity mmdet
