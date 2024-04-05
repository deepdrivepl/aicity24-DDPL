from glob import glob
from tqdm import tqdm
import os
import sys
import random
import cv2
from shutil import copyfile
from collections import defaultdict


input_list = "/mnt/ssd8/ola/aicity/data/FishEye8K/train-minus-val-fractal.txt"
input_images = sorted([x.rstrip() for x in open(input_list)])

output_dir = "/mnt/ssd8/ola/aicity/data/FishEye8K/train-minus-val/aug-pix2pix-30-per-camera"
os.makedirs(output_dir, exist_ok=True)

NUM_FILES = 10
NUM_PROMPTS = 3

cam2flist = defaultdict(list)
for input_image in input_images:
    cam = os.path.basename(input_image).split('_')[0]
    cam2flist[cam].append(input_image)


prompts = [
    "evening lights",
    "make it look like it is evening",
    "night lights",
    "pitch black night",
    "make it look like from a surveillance camera at night",
    "make it look like it is a sunny morning",
    "morning",
    "dark night with bright car lights",
    "grayscale",
    "blurry night",
    "make it harder for a neural network",
    "cloudy day",
]


for cam, input_images in tqdm(cam2flist.items()):
    images = random.sample(input_images, NUM_FILES)
    
    for input_image in images:
        H,W = cv2.imread(input_image).shape[:2]
        prompts_ = random.sample(prompts, NUM_PROMPTS)
    
        for i, prompt in enumerate(prompts_):
            out_path = input_image.replace(os.path.dirname(input_image), output_dir)
            out_path = out_path.replace('.png', f'-{i}.png').replace('.jpg', f'-i.jpg')
            print(f"python edit_cli.py --input {input_image} --output {out_path} --edit '{prompt}'  --resolution {int(0.75*(min(H,W)))}")
            os.system(f"python edit_cli.py --input {input_image} --output {out_path} --edit '{prompt}' --resolution {int(0.75*(min(H,W)))}")    