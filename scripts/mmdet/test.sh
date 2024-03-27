# 004 ep3 val, test & test-challenge
CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/004-960-val.py' \
"/aicity/results/FishEye8K/mmdet/codetr/004-960-FishEye8k_VD_UAVDT_WD_aug-co_dino-swinl-16-detr-bs4/epoch_3.pth" \
--cfg-options test_evaluator.outfile_prefix="004-ep3-val" test_dataloader.batch_size=8 \
--work-dir "/aicity/results/FishEye8K/mmdet/inference"

CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/004-960-test.py' \
"/aicity/results/FishEye8K/mmdet/codetr/004-960-FishEye8k_VD_UAVDT_WD_aug-co_dino-swinl-16-detr-bs4/epoch_3.pth" \
--cfg-options test_evaluator.outfile_prefix="004-ep3-test" test_dataloader.batch_size=8 \
--work-dir "/aicity/results/FishEye8K/mmdet/inference"


CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/004-960-test-challenge.py' \
"/aicity/results/FishEye8K/mmdet/codetr/004-960-FishEye8k_VD_UAVDT_WD_aug-co_dino-swinl-16-detr-bs4/epoch_3.pth" \
--cfg-options test_evaluator.outfile_prefix="004-ep3-test-challenge" test_dataloader.batch_size=8 \
--work-dir "/aicity/results/FishEye8K/mmdet/inference"


# 005 ep5 val & test-challenge
CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/005-960-val.py' \
"/aicity/results/FishEye8K/mmdet/codetr/005-960-cd-004-ep3-FE8K-train-test/epoch_5.pth" \
--cfg-options test_evaluator.outfile_prefix="005-ep5-val" test_dataloader.batch_size=8 \
--work-dir "/aicity/results/FishEye8K/mmdet/inference"


CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/005-960-test-challenge.py' \
"/aicity/results/FishEye8K/mmdet/codetr/005-960-cd-004-ep3-FE8K-train-test/epoch_5.pth" \
--cfg-options test_evaluator.outfile_prefix="005-ep5-test-challenge" test_dataloader.batch_size=8 \
--work-dir "/aicity/results/FishEye8K/mmdet/inference"


# 006 ep7 val & test-challenge
CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/006-960-val.py' \
"/aicity/results/FishEye8K/mmdet/codetr/006-960-cd-005-ep3-FE8K-train-test/epoch_7.pth" \
--cfg-options test_evaluator.outfile_prefix="006-ep7-val" test_dataloader.batch_size=8 \
--work-dir "/aicity/results/FishEye8K/mmdet/inference"

CUDA_VISIBLE_DEVICES=0 python /mmdetection/tools/test.py \
'/aicity/scripts/mmdet/configs/006-960-test-challenge.py' \
"/aicity/results/FishEye8K/mmdet/codetr/006-960-cd-005-ep3-FE8K-train-test/epoch_7.pth" \
--cfg-options test_evaluator.outfile_prefix="006-ep7-test-challenge" test_dataloader.batch_size=8 \
--work-dir "/aicity/results/FishEye8K/mmdet/inference"