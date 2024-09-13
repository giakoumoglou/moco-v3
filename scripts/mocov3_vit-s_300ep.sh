#!/bin/bash
#PBS -lselect=1:ncpus=64:mem=256gb:ngpus=8
#PBS -lwalltime=72:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate torch

python main_moco.py -a vit_small -b 1024 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --epochs=300 --warmup-epochs=40 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/

python main_lincls.py -a vit_small --lr 3 --pretrained ./checkpoint_0299.pth.tar --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/