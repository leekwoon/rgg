#! /bin/bash
set -e

mkdir -p $(pwd)/logs

echo Downloading the pretrained gap predictor ...
gdown https://drive.google.com/uc?id=1wCCLmp0XDwh247H-TUKgnKbGKnn4bMxN

tar -xvf gap_predictor.tar
rsync -av gap_predictor/ logs/gap_predictor
rm gap_predictor.tar 
rm gap_predictor

echo 
echo Preparation complete!