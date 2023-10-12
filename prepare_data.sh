#! /bin/bash
set -e

mkdir -p $(pwd)/logs

echo Downloading the generated synthetic data ...
gdown https://drive.google.com/uc?id=1Ely2qBsG5xPVDnJJlo-si4pDp_S1T-BO

tar -xvf data.tar
rsync -av data/ logs/data
rm data.tar
rm data

echo 
echo Preparation complete!