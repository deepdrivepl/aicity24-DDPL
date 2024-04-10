#!/bin/bash

OUT_DIR="/aicity/data"

mkdir -p $OUT_DIR

wget -q --show-progress https://ia800309.us.archive.org/30/items/006-epoch-7/FishEye8K.zip -P $OUT_DIR
unzip $OUT_DIR/FishEye8K.zip -d $OUT_DIR
rm $OUT_DIR/FishEye8K.zip 