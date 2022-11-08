# Multi-GPU, use decoder feature in FPN, use attention module
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 28501 train.py --name Melanocyte_Attn_DecoderFeat --dataroot DATA_PATH --resize_or_crop none --gpu_ids 0,1,2,3 --batchSize 2 --no_instance --loadSize 256 --ngf 32 --has_real_image --save_epoch_freq 5 --use_resnet_as_backbone --use_UNet_skip --fpn_feature decoder --niter_decay 200