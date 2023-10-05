#! /bin/bash
set -e

mkdir -p $(pwd)/logs

echo Downloading the pretrained gap predictor ...
gdown https://drive.google.com/uc?id=1ElB0klbW3eDPehiRbdGd0fTmsulWGP8D

tar -xvf gap_predictor.tar
rsync -av gap_predictor/ logs/gap_predictor
rm gap_predictor.tar

echo 
echo Preparation complete!