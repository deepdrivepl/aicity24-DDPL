ARG PYTORCH="2.1.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y curl wget unzip 

    
# RUN mkdir -p /aicity/models \
#     && wget https://ia600309.us.archive.org/30/items/006-epoch-7/006_epoch_7.pth -d /aicity/models/006_epoch_7.pth

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"
    
# codetr dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir fairscale future tensorboard setuptools

# Install MMDetection
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection.git /mmdetection \
    && cd /mmdetection \
    && pip install --no-cache-dir -e .
    
# data aug dependencies
RUN pip install --no-cache-dir albumentations diffusers kornia \
    pudb==2019.2 test-tube>=0.7.5 streamlit>=0.73.1 torch-fidelity==0.3.0 \
    transformers==4.19.2 einops==0.3.0 omegaconf pytorch-lightning==1.4.2 torchmetrics==0.6.0 \
    git+https://github.com/crowsonkb/k-diffusion.git 

RUN pip install --no-cache-dir -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
RUN pip install --no-cache-dir -e git+https://github.com/openai/CLIP.git@main#egg=clip
RUN pip uninstall pycocotools -y 
RUN pip install git+https://github.com/deepdrivepl/FE8K-eval.git

WORKDIR /mmdetection

