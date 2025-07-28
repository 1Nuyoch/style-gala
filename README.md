# style-gala

This repo contains the code for our paper **Enhancing Chinese Character Restoration: A Style-Integrated and Feature-Aggregated Approach**.

## Contents

1. [Setup Instructions](#1-setup-instructions)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Training and Evaluation](#3-training-and-evaluation)

## 1. Setup Instructions

- Clone the repo:

    ```.bash
    git clone [https://github.com/1Nuyoch/style-gala.git]
    cd style-gala
    ```

- Create a conda environment:

    ```.bash
    conda create --name stygan python=3.7
    ```

- Install [Pytorch 1.7.1](https://pytorch.org/get-started/previous-versions/) and other dependencies:

    ```.bash
    pip3 install -r requirements.txt
    export TORCH_HOME=$(pwd) && export PYTHONPATH=.
    ```

- Download the models for the high receptive perceptual loss:

    ```.bash
    mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
    ```

## 2. Dataset Preparation

###  Chinese Calligraphy Styles dataset

#### Training Data

    ```.bash
    mkdir -p datasets/
    # unzip & split into train/test/visualization
    bash tools/prepare_celebahq.sh
    ```

#### Evaluation Data

- Generate 2k `(image, mask)` pairs to be used for evaluation.

    ```.bash
    bash tools/prepare_celebahq_evaluation.sh
    ```

#### Evaluation Data

##### Irregular Mask Strategy

- Generate 30k `(image, mask)` pairs to be used for evaluation.

    ```.bash
    bash tools/prepare_places_evaluation.sh
    ```

##### Segmentation Mask strategy

    ```.bash
    python -m pip install detectron2==0.5 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
    ```

- Download networks for segmentation masks:

    ```.bash
    mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
    ```

- Generate `(image, mask)` pairs to be used for segmentation mask based evaluation.

    ```.bash
    bash tools/prepare_places_segm_evaluation.sh
    ```

> Note: The pairs are only generated for images with detected instances.

## 3. Training and Evaluation

![places](docs/places_qual.svg)

### Training on 256x256

    ```.bash
    python train.py \
        --outdir=training-runs-inp \
        --img_data=datasets/Chinese/train \
        --gpus 1 \
        --kimg 500 \
        --gamma 10 \
        --aug 'noaug' \
        --metrics True \
        --eval_img_data datasets/Chinese/evaluation/random_segm_256
        --batch 4
    ```

> Note: If the process hangs on `Setting up PyTorch plugin ...`, refer to [this issue](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/41).

### Evaluation

- Run the following command to calculate the metric scores (fid, ssim ，psnr and lpips) using 1 gpus:

    ```.bash
    python evaluate.py \
        --img_data=datasets/Chinese/evaluation/random_segm_256 \
        --network=[path-to-checkpoint] \
        --num_gpus=8
    ```

