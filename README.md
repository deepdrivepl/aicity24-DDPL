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

```
cd /aicity/scripts
python run_filtering_and_eval.py --detections mmdet/006-ep7-val.bbox.json
python run_filtering_and_eval.py --detections mmdet/006-ep7-test-challenge.bbox.json --split test-challenge
```

## Run training

### Download augmented dataset
### Generate augmented dataset
### Run training
