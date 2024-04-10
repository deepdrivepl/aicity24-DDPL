#!/bin/bash

OUT_DIR="/aicity/data"

mkdir -p $OUT_DIR

echo "Downloading model weights"
wget -q --show-progress https://ia600309.us.archive.org/30/items/006-epoch-7/006_epoch_7.pth -P /aicity/models

echo "Downloading dataset"
wget -q --show-progress https://ia800309.us.archive.org/30/items/006-epoch-7/FishEye8K.zip -P $OUT_DIR
unzip $OUT_DIR/FishEye8K.zip -d $OUT_DIR
rm $OUT_DIR/FishEye8K.zip 