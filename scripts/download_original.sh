#!/bin/bash

OUT_DIR="/aicity/data-new"

mkdir -p $OUT_DIR

wget -q --show-progress https://ia800309.us.archive.org/30/items/006-epoch-7/UAVDT.zip -P $OUT_DIR
wget -q --show-progress https://ia800309.us.archive.org/30/items/006-epoch-7/VisDrone.zip -P $OUT_DIR
wget -q --show-progress https://ia800309.us.archive.org/30/items/006-epoch-7/WoodScape.zip -P $OUT_DIR

unzip $OUT_DIR/UAVDT.zip -d $OUT_DIR
unzip $OUT_DIR/VisDrone.zip -d $OUT_DIR
unzip $OUT_DIR/WoodScape.zip -d $OUT_DIR

rm $OUT_DIR/UAVDT.zip $OUT_DIR/VisDrone.zip $OUT_DIR/WoodScape.zip