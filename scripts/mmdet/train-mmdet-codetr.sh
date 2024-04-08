MKL_NUM_THREADS=40 ./dist_train.sh "/aicity/scripts/mmdet/configs/004-960.py" 2 --amp  --work-dir "/aicity/scripts/mmdet/training/004-960-FishEye8k_VD_UAVDT_WD_aug-co_dino-swinl-16-detr-bs4"

MKL_NUM_THREADS=40 ./dist_train.sh "/aicity/scripts/mmdet/configs/005-960.py" 2 --amp  --work-dir "/aicity/scripts/mmdet/training/005-960-cd-004-ep3-FE8K-train-test"

MKL_NUM_THREADS=40 ./dist_train.sh "/aicity/scripts/mmdet/configs/006-960.py" 2 --amp  --work-dir "/aicity/scripts/mmdet/training/006-960-cd-005-ep3-FE8K-train-test"

