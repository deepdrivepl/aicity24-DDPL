# 006 ep7 val & test-challenge
CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/006-960-val.py' \
"/aicity/models/006_epoch_7.pth" \
--cfg-options test_evaluator.outfile_prefix="006-ep7-val" test_dataloader.batch_size=8 \
--work-dir "/aicity/scripts/mmdet/inference"

CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/006-960-test-challenge.py' \
"/aicity/models/006_epoch_7.pth" \
--cfg-options test_evaluator.outfile_prefix="006-ep7-test-challenge" test_dataloader.batch_size=8 \
--work-dir "/aicity/scripts/mmdet/inference"