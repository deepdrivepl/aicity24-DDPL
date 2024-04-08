# aicity24-DDPL
![TOP6](images/res-val-TOP-6.png)

## Run inference using a trained Co-DETR model

### get data 

### run inference
```
cd scripts/mmdet
./build_and_run.sh

cd /aicity/scripts/mmdet/
./test.sh
```

### filter detections & evaluate
For evaluation we use a [modified pycocotools library](https://github.com/deepdrivepl/FE8K-eval).\
Our reposotory is a pip-installable version of https://github.com/MoyoG/FishEye8K/tree/main/evaluation_Linux.

The provided script will filter detections according to class-dependent confidence thresholds. These thresholds were chosen experimentally to maximize the F1 score on the test set (original validation) and are as follows: 
```
{'Bike': 0.4, 'Bus': 0.5, 'Car': 0.5, 'Pedestrian': 0.4, 'Truck': 0.45}
```
If `--split` is different from "test-challenge", and `--skip_metrics` is not specified, the script will also calcuate metrics for the specified detection file `--detections`.

```
cd /aicity/scripts
python run_filtering_and_eval.py --detections mmdet/006-ep7-val.bbox.json
python run_filtering_and_eval.py --detections mmdet/006-ep7-test-challenge.bbox.json --split test-challenge
```

## Run training

### Download augmented dataset
### Generate augmented dataset

1. Download non-augmented datasets
2. Run distortion (VisDrone, UAVDT)
   ```
   cd /aicity/scripts
   python gen_distorted.py --debug
   ```
4. Run pixel-level data augmentation (WoodScape, VisDrone, UAVDT)
   ```
   cd /aicity/scripts
   python gen_augmented.py --debug
   ```
5. Run GAN-based data augmentation

   pix2pix
   ```
   cd /aicity/3rdparty/instruct-pix2pix
   python generate-FE8K.py
   ```

   style transfer
   ```
   cd /aicity/3rdparty/pytorch-neural-style-transfer
   python style_transfer_FE8K.py
   ```
   
### Run training


## Acknowledgements
- 
