import json
import argparse
import os

from tqdm import tqdm
from glob import glob
from shutil import copyfile

import cv2
import numpy as np



directories = {
    "FE8K-orig": "/aicity/data/FishEye8K/train", # original FE8K
    "FE8K-aug": "/aicity/data/augmented", # augmented FE8K
    "UAVDT-aug": "/aicity/data/UAVDT/yolo-aug-distorted", # aug + distorted UAVDT
    "VisDrone-aug": "/aicity/data/VisDrone/yolo-aug-distorted", # aug + distorted VisDrone
    "WoodScape-aug": "/aicity/data/WoodScape/yolo-aug", # aug WoodScape
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--train_list', type=str, default='/aicity/data/train-minus-val.txt')
    args = parser.parse_args()
    
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {'id': 0, 'name': 'Bus', 'supercategory': 'Bus'},
            {'id': 1, 'name': 'Bike', 'supercategory': 'Bike'},
            {'id': 2, 'name': 'Car', 'supercategory': 'Car'},
            {'id': 3, 'name': 'Pedestrian', 'supercategory': 'Pedestrian'},
            {'id': 4, 'name': 'Truck', 'supercategory': 'Truck'}
        ]
    }
    
    img_id, obj_id = 0, 0
    for dir_name, directory in directories.items():
        img_dir = f"{directory}/images"
        lbl_dir = f"{directory}/labels"
        
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            images = glob(f"{img_dir}/*.png") + glob(f"{img_dir}/*.jpg")
        else:
            images = glob(f"{directory}/*.png") + glob(f"{directory}/*.jpg")
            
        if dir_name == "FE8K-orig":
            train_fnames = [os.path.basename(x.rstrip()) for x in open(args.train_list)]
            images = [x for x in images if os.path.basename(x) in train_fnames]
            
        labels = [x.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt') for x in images]
        print(f"Found {len(images)} images in {img_dir}")
        
        
        for img_path, lbl_path in tqdm(zip(images, labels)):
            if not os.path.isfile(lbl_path):
                print(f"{lbl_path} not found.")
                continue
                
            H,W = cv2.imread(img_path).shape[:2]
            coco_annotations['images'].append({"id": img_id, "file_name": os.path.basename(img_path), "height": H, "width": W})
            out_dir = f"{args.out_dir}/images"
            os.makedirs(out_dir, exist_ok=True)
            copyfile(img_path, f"{out_dir}/{os.path.basename(img_path)}")
            
            labels = [x.rstrip().split(' ') for x in open(lbl_path)]
            
            for lbl in labels:
                cls_id, xc, yc, w,h = map(float, lbl)
                xmin, ymin = xc-0.5*w, yc-0.5*h
                xmin, w = [_*W for _ in [xmin, w]]
                ymin, h = [_*H for _ in [ymin, h]]

                coco_annotations["annotations"].append({
                    "id": obj_id,
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "bbox": [xmin, ymin, w, h],
                    "segmentation": [],
                    "area": w*h,
                    "iscrowd": 0
                })
                obj_id+=1
                
            img_id+=1
            
    with open(f"{args.out_dir}/train-augmented.json", 'w', encoding='utf-8') as f:
        json.dump(coco_annotations, f, ensure_ascii=False, indent=4)