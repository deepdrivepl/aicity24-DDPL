# baseline
python /aicity/3rdparty/yolov7/train_aux.py \
--workers 8 --device 0 --sync-bn --batch-size 10 \
--img-size 1280 1280 \
--hyp /aicity/scripts/yolo/hyp.focal.scratch.custom.yaml \
--data /aicity/scripts/yolo/FishEye8K-train.yaml \
--cfg /aicity/scripts/yolo/yolov7-e6e.yaml \
--weights /aicity/3rdparty/yolov7/weights/yolov7-e6e.pt \
--project /aicity/results/yolo \
--name yolov7e6e-1280-baseline --epochs 300 --save_period 1

python /aicity/3rdparty/yolov7/test.py \
--img 1280 --conf 0.001 --iou 0.65 --batch 32 --device 0 \
--data /aicity/scripts/yolo/FishEye/FishEye8K-train.yaml \
--weights /aicity/results/yolo/yolov7e6e-1280-baseline/weights/best.pt \
--project /aicity/results/yolo \
--name yolov7e6e-1280-baseline/test \
--task test --save-json --exist-ok


# advanced aug
python /aicity/3rdparty/yolov7/train_aux.py \
--workers 8 --device 0 --sync-bn --batch-size 12 \
--img-size 1280 1280 \
--hyp /aicity/scripts/yolo/hyp.focal.scratch.custom.yaml \
--data /aicity/scripts/yolo/combo-no-crops-aug.yaml \
--cfg /aicity/scripts/yolo/yolov7-e6e.yaml \
--weights /aicity/3rdparty/yolov7/weights/yolov7-e6e.pt \
--project /aicity/results/yolo \
--name yolov7e6e-1280-advanced_aug --epochs 300 --autobalance

python /aicity/3rdparty/yolov7/test.py \
--img 1280 --conf 0.001 --iou 0.65 --batch 12 --device 0 \
--data /aicity/scripts/yolo/combo-no-crops-aug.yaml \
--weights /aicity/results/yolo/yolov7e6e-1280-advanced_aug/weights/best.pt \
--project /aicity/results/yolo \
--name yolov7e6e-1280-advanced_aug/test \
--task test --save-json --exist-ok