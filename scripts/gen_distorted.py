import argparse
import os

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from kornia import image_to_tensor
from kornia.augmentation import AugmentationSequential, RandomFisheye
from kornia.constants import DataKey



NAMES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]

COLORS = {
     0: (16, 177, 244),
     1: (37, 216, 236),
     2: (241, 108, 43),
     3: (115, 252, 23),
     4: (254, 206, 98)
}

AUG = RandomFisheye(
    center_x=torch.Tensor([0.0, 0.0]), 
    center_y=torch.Tensor([0.0, 0.0]), 
    gamma=torch.Tensor([5.0, 150.0]), 
    p=1.0
)


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
    
    
def get_augmented(img_orig, labels, orig_labels):
    img_tensor = (image_to_tensor(img_orig).float() / 255.0).unsqueeze(0)
    params = AUG.generate_parameters(img_tensor.shape)
    
    img_out = AUG(img_tensor, params)

    bboxes = []
    for i, label in enumerate(labels):
        mask = torch.zeros(img_tensor.shape)
        xmin,ymin,xmax,ymax = label
        mask[:,:,ymin:ymax+1, xmin:xmax+1] = 1
        mask_output = AUG(mask, params)
        mask_output = (np.transpose(mask_output.squeeze().numpy(), (1,2,0))*255).astype(np.uint8).copy()[:,:,0]
        mask_output[mask_output < 127] = 0
        mask_output[mask_output >= 127] = 255
        contours = cv2.findContours(mask_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        if len(contours) == 0:
            bboxes.append([])
            continue

        if len(contours) >= 1:
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                if w <=2 and h<=2:
                    continue
                xmax, ymax = x+w, y+h
                bboxes.append([int(x) for x in [x,y,xmax,ymax]])

    new_labels = []
    for new_bbox, old_label in zip(bboxes, orig_labels):
        if len(new_bbox) == 0:
            continue
        cls_id = int(old_label[0])
        xmin,ymin,xmax,ymax = new_bbox
        xc, yc = (xmin+xmax)/2, (ymin+ymax)/2
        w,h = xmax-xmin, ymax-ymin

        if w < 5 or h < 5:
            continue

        xc,w = [_/W for _ in [xc,w]]
        yc,h = [_/H for _ in [yc,h]]
        new_labels.append([cls_id, xc,yc,w,h])
        
    img_out = (np.transpose(img_out.squeeze().numpy(), (1,2,0))*255).astype(np.uint8).copy()
    return img_out, new_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true', help="if True, the visualization will be saved.")
    args = parser.parse_args()
    
    ignore = "WoodScape"
    images = glob("../data/**/yolo/images/*", recursive=True)
    images = [x for x in images if ignore not in x]

    
    for img_path in tqdm(images):

        img_orig = cv2.imread(img_path)
        H,W = img_orig.shape[:2]
        out_fname_vis = img_path.replace('images', 'vis').replace('yolo', 'yolo-distorted')
        out_fname_lbl = img_path.replace('images', 'labels').replace('yolo', 'yolo-distorted').replace('.jpg', '.txt').replace('.png', '.txt')
        out_fname_img = img_path.replace('images', 'images').replace('yolo', 'yolo-distorted')

        for d in [out_fname_vis, out_fname_lbl, out_fname_img]:
            os.makedirs(os.path.dirname(d), exist_ok=True)

        label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('images', 'labels')
        labels = [list(map(float, x.rstrip().split(' ')[1:])) for x in open(label_path)]
        labels = [list(map(int, [x[0]*W, x[1]*H, x[2]*W, x[3]*H])) for x in labels]

        xyxy_labels = []
        for lbl in labels:
            xc,yc,w,h = lbl
            xmin, ymin = int(xc-0.5*w), int(yc-0.5*h)
            xmax, ymax = int(xc+0.5*w), int(yc+0.5*h)

            xyxy_labels.append([xmin,ymin,xmax,ymax])
        
        orig_labels = [list(map(float, x.rstrip().split(' '))) for x in open(label_path)]
        img_out, new_labels = get_augmented(img_orig, xyxy_labels, orig_labels)
    
        with open(out_fname_lbl, 'w') as f:
            for _ in new_labels:
                f.write(f"{' '.join(list(map(str, _)))}\n")
                
        cv2.imwrite(out_fname_img, img_out)
        
        if args.debug:
            vis_augmented(img_out, new_labels, out_fname_vis)