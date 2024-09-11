## MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers

This is a PyTorch implementation of the [MoCo v3 paper](https://arxiv.org/abs/2104.02057):

```
@misc{chen2021empiricalstudytrainingselfsupervised,
      title={An Empirical Study of Training Self-Supervised Vision Transformers}, 
      author={Xinlei Chen and Saining Xie and Kaiming He},
      year={2021},
      eprint={2104.02057},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2104.02057}, 
}
```

### Introduction

This is a PyTorch implementation of [MoCo v3](https://arxiv.org/abs/2104.02057) for self-supervised ResNet and ViT. The original MoCo v3 was implemented in Tensorflow and run in TPUs. This repo re-implements in PyTorch and GPUs. Despite the library and numerical differences, this repo reproduces the results and observations in the paper. 

### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

#### ResNet-50, pretrain

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run on the first node:

```
python main_moco.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```

On the second node, run the same command with `--rank 1`.

#### ViT-S, pretrain

To do unsupervised pre-training of a ViT-Small model on ImageNet in an 8-gpu machine, run:

```bash
python main_moco.py \
  -a vit_small -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

#### ViT-B, pretrain
To do unsupervised pre-training of a ViT-Base model on ImageNet on 8 nodes, run:

```bash
python main_moco.py \
  -a vit_base \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 8 --rank 0 \
  [your imagenet-folder with train and val folders]
```

On other nodes, run the same command with `--rank 1`, ..., `--rank 7` respectively.

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:

```bash
python main_lincls.py \
  -a [architecture] --lr [learning rate] \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```

### End-to-end fine-tuning ViT

To fine-tune ViT end-to-end:

1. Convert the pre-trained ViT checkpoint to DeiT format:
```bash
python convert_to_deit.py \
  --input [your checkpoint path]/[your checkpoint file].pth.tar \
  --output [target checkpoint file].pth
```

2. Run the training (in the DeiT repo) with the converted checkpoint:
```bash
python $DEIT_DIR/main.py \
  --resume [target checkpoint file].pth \
  --epochs 150
```

### Transfer Learning

See [./transfer](transfer).

### Models

#### ResNet-50, linear classification

| Pretrain epochs | Pretrain crops | Linear acc |
|-----------------|----------------|------------|
| 100             | 2x224          | 68.9       |
| 300             | 2x224          | 72.8       |
| 1000            | 2x224          | 74.6       |

#### ViT, linear classification

| Model      | Pretrain epochs | Pretrain crops | Linear acc |
|------------|-----------------|----------------|------------|
| ViT-Small  | 300             | 2x224          | 73.2       |
| ViT-Base   | 300             | 2x224          | 76.7       |

#### ViT, end-to-end fine-tuning

| Model      | Pretrain epochs | Pretrain crops | E2E acc |
|------------|-----------------|----------------|---------|
| ViT-Small  | 300             | 2x224          | 81.4    |
| ViT-Base   | 300             | 2x224          | 83.2    |

**Note:** End-to-end fine-tuning results are obtained using the [DeiT](https://github.com/facebookresearch/deit) repository with default DeiT configs. ViT-B is fine-tuned for 150 epochs.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

