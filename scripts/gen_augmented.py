import json
import argparse
import os

from tqdm import tqdm
from glob import glob
from shutil import copyfile

import cv2
import numpy as np
import albumentations as albu


DIRS = {
    "WoodScape": "/aicity/data/WoodScape/yolo",
    "VisDrone": "/aicity/data/VisDrone/yolo-distorted",
    "UAVDT": "/aicity/data/UAVDT/yolo-distorted",
}
NAMES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]

COLORS = {
     0: (16, 177, 244),
     1: (37, 216, 236),
     2: (241, 108, 43),
     3: (115, 252, 23),
     4: (254, 206, 98)
}

TRANSFORM = albu.Compose([

    albu.ToGray(p=0.2, always_apply=False),

    albu.OneOf([
        albu.OneOf([
            albu.Defocus(p=1),
            albu.GaussianBlur(p=1),
            albu.GlassBlur(p=1),
            albu.MedianBlur(p=1),
            albu.MotionBlur(p=1),
            albu.ZoomBlur(p=1, max_factor=1.2)],
            p=0.9
        ),
        albu.Sharpen(p=0.3)],
        p=1
    ),

    albu.OneOf([
        albu.RandomFog(p=1),
        albu.RandomShadow(p=1),
        albu.RandomSunFlare(p=0.4,src_radius=200)],
        p = 0.3
    ),
    
    albu.ImageCompression(quality_lower=70, p=0.8),  
])


def vis_bbox(img, mask, bbox, color, class_name, lw=2):
    xmin, ymin, xmax, ymax = bbox
    mask[ymin:ymax, xmin:xmax,:] = color
    img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, lw)
    
    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )
    img = cv2.rectangle(
        img, (xmin, ymin - int(1.3 * text_height)), (xmin + text_width, ymin), color, -1
    )
    img = cv2.putText(
        img,
        text=class_name,
        org=(xmin, ymin - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.45,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
    )
    return img, mask


def vis_augmented(img_out, new_labels, out_fname_vis):
    
    mask = np.zeros(img_out.shape, dtype=np.uint8)
    for lbl in new_labels:
        cls_id, xc, yc, w, h = lbl
        cls_name = NAMES[int(cls_id)]

        xc,yc,w,h = map(float, [xc,yc,w,h])
        xc,w = [_*W for _ in [xc,w]]
        yc,h = [_*H for _ in [yc,h]]
        xmin,ymin = int(xc-0.5*w), int(yc-0.5*h)
        xmax,ymax = int(xc+0.5*w), int(yc+0.5*h)

        img_out, mask = img, mask = vis_bbox(img_out, mask, [xmin,ymin,xmax,ymax], COLORS[int(cls_id)], cls_name)

    out = cv2.addWeighted(img, 0.7, mask, 0.3, 0.0)
    cv2.imwrite(out_fname_vis, out)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true', help="if True, the visualization will be saved.")
    args = parser.parse_args()
    
    for ds_name, ds_dir in DIRS.items():
        
        images = sorted(glob(f"{ds_dir}/images/*.jpg")+glob(f"{ds_dir}/images/*.png"))
        labels = [x.replace('.png', '.txt').replace('.jpg', '.txt').replace('images', 'labels') for x in images]
        print(f"{ds_name}: found {len(images)} files")
        
        for img_path, lbl_path in tqdm(zip(images, labels)):
            img = cv2.imread(img_path)
            H,W = img.shape[:2]
            transformed = TRANSFORM(image=img.copy())['image']
            out_img_path = img_path.replace('yolo', 'yolo-aug')
            out_lbl_path = lbl_path.replace('yolo', 'yolo-aug')
            out_vis_path = out_img_path.replace('images', 'vis')

            for x in [out_img_path, out_lbl_path, out_vis_path]:
                os.makedirs(os.path.dirname(x), exist_ok=True)

            cv2.imwrite(out_img_path, transformed)
            copyfile(lbl_path, out_lbl_path)
            
            if args.debug:
                labels = [list(map(float, x.rstrip().split(' '))) for x in open(lbl_path)]
                vis_augmented(transformed, labels, out_vis_path)
 