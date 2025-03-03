if [ ! -d "$(pwd)/../../3rdparty/yolov7" ]; then
    git clone https://github.com/WongKinYiu/yolov7.git $(pwd)/../../3rdparty/yolov7
fi

if [ ! -d "$(pwd)/../../3rdparty/yolov7/weights" ]; then
    mkdir -p $(pwd)/../../3rdparty/yolov7/weights
fi

if [ ! -f "$(pwd)/../../3rdparty/yolov7/weights/yolov7-e6e.pt" ]; then
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -O $(pwd)/../../3rdparty/yolov7/weights/yolov7-e6e.pt
fi

docker build -t mmdet .
docker run --gpus all --ipc=host -it -v $(pwd)/../..:/aicity mmdet
