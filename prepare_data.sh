#! /bin/bash
set -e

mkdir -p $(pwd)/logs

echo Downloading the generated synthetic data ...
gdown https://drive.google.com/uc?id=1bkL_LRKgPPJXNjvieR6ocGXn-KcQbD4Z

tar -xvf data.tar
rsync -av data/ logs/data
rm data.tar
rm data

echo 
echo Preparation complete!