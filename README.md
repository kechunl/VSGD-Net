# VSGD-Net
This is the official PyTorch implementation of VSGD-Net: Virtual Staining Guided Melanocyte Detection on Histopathological Images

## Prerequisites
- Linux or macOS
- Python 2 or 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
- **Please install the env ```/projects/patho4/shared_env/Pix2Pix_clone.tar.gz``` following ```README.md``` under the same folder.**
- ~~Install PyTorch and dependencies from http://pytorch.org~~
- ~~Install python libraries [dominate](https://github.com/Knio/dominate).~~
- Clone this repo:
```bash
git clone https://github.com/kechunl/VSGD-Net.git
cd VSGD-Net
```

### Quick Inference
- A few example H&E skin biopsy images are included in the ```datasets/test_A``` folder.
- ~~Please download the pre-trained Melanocyte model from [here](https://drive.google.com/file/d/1nVftbE-h8t7OVcTmFR-9FQqRnIkbZIY9/view?usp=sharing) (google drive link), and unzip it under ```./checkpoints/```.~~
- Test the model (```bash ./scripts/test_melanocyte.sh```)

The test results will be saved to a html file here: ```./results/melanocyte/test_latest/index.html```

### Training
- An example training script is provided (```./scripts/train_melanocyte.sh```):
```
# Multi-GPU, use decoder feature in FPN, use attention module
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 28501 train.py --name Melanocyte_Attn_DecoderFeat --dataroot DATA_PATH --resize_or_crop none --gpu_ids 0,1,2,3 --batchSize 2 --no_instance --loadSize 256 --ngf 32 --has_real_image --save_epoch_freq 5 --use_resnet_as_backbone --use_UNet_skip --fpn_feature decoder --niter_decay 200
```
Note: please specify the data path as explained in [Training with your own dataset](#Training-with-your-own-dataset).

#### Training with your own dataset
- If you want to train with your own dataset, please generate the corresponding image patches and name the folders as ```train_A``` and ```train_B```. For detection purpose, you should also name the mask folder as ```train_mask```. In our paper, we use 256x256 patched in 10x magnification. Please refer to our paper for the preprocessing steps.
- After preparing the image folders, you can generate the json file using ```./data/get_coco_anno.py```. Please take a look at the argument and modify accordingly.
- In VSGD-Net's dataloader, the default setting for preprocessing is `none` which will do nothing other than making sure the image is divisible by 32. If you want a different setting, please change it by using the `--resize_or_crop` option. For example, `scale_width_and_crop` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`. `crop` skips the resizing step and only performs random cropping. `scale_width` scales the width of all training images to `opt.loadSize` (256) while keeping the aspect ratio. 

#### More Training/Test Details
- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.

#### Check your training/testing progress
When training the model, eveything will be stored in ``./checkpoints/[experiment_name]``. You can easily check the output images in ``index.html`` under ``/web_train/`` and ``/web_val/``. The command I use to visualize the folder in web server is:

	python -m http.server 8888 -b patholin.cs.washington.edu


## Dataset used in VSGD-Net
- Path: ```/projects/patho4/Kechun/melanocyte_detection/melanocyte_data/datasets/melanocyte_10x_256```
- Subsets: train, test, valid
- ```xxx_A```: H&E;
  ```xxx_B```: Sox10;
  ```xxx_mask```: melanocyte mask.

## Acknowledgement

This project is based on [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD).