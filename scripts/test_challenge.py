import json
import os
import warnings
warnings.simplefilter("ignore", ResourceWarning)

from tqdm import tqdm
from glob import glob


def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId



res_path = "../scripts/mmdet/006-ep7-test-challenge.bbox.json"
yolo = False

YOLO_THS_PER_CLASS = {1: 0.25, 0: 0.55, 2: 0.5, 3: 0.3, 4: 0.3}
CODETR_THS_PER_CLASS = {1: 0.4, 0: 0.5, 2: 0.5, 3: 0.4, 4: 0.45}
ths_per_class = YOLO_THS_PER_CLASS if yolo else CODETR_THS_PER_CLASS


res = json.load(open(res_path))
res = [x for x in res if x['score'] >= ths_per_class[x['category_id']]]

mapped = []
for det_obj in tqdm(res):
    im_id = det_obj['image_id'] if not yolo else get_image_Id(os.path.basename(det_obj['image_id']))
    mapped.append({
        'image_id': im_id,
        'category_id': det_obj['category_id'],
        'bbox': det_obj['bbox'],
        'score': det_obj['score']
    })

with open(res_path.replace('.json', '_mapped.json'), 'w', encoding='utf-8') as f:
    json.dump(mapped, f, ensure_ascii=False, indent=4)