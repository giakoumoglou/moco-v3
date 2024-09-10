#!/bin/bash
#PBS -lselect=1:ncpus=64:mem=256gb:ngpus=8
#PBS -lwalltime=72:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate torch

python main_moco.py --lr=.3 --wd=1.5e-6 --epochs=1000 --moco-m=0.996 --moco-m-cos --crop-min=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/

python main_lincls.py -a resnet50 --lr 0.1 --batch-size 1024 --pretrained ./checkpoint_0099.pth.tar --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/