#!/bin/bash

OUT_DIR="/aicity/data"

mkdir -p $OUT_DIR

wget -q --show-progress https://ia800309.us.archive.org/30/items/006-epoch-7/mmdet-coco.zip -P $OUT_DIR
unzip $OUT_DIR/mmdet-coco.zip -d $OUT_DIR
rm $OUT_DIR/mmdet-coco.zip