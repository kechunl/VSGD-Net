import argparse
import os
from util import util
import torch
from util.detection_utils import setup_for_distributed
# import yaml

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='Detection', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='detection', help='which model to use [pix2pixHD | detection]')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/melanocyte/') 
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')       
        self.parser.add_argument('--niter_fix_GAN', type=int, default=200, help='number of epochs that we only train the detection branch') 
        self.parser.add_argument('--use_resnet_as_backbone', action='store_true', help='if true, use resnet as backbone in generator')
        self.parser.add_argument('--use_UNet_skip', action='store_true', help='if true, use UNet Skip Connection design in decoder of generator')      
        self.parser.add_argument('--fpn_feature', type=str, default='decoder', choices=['encoder', 'decoder'], help='specify which features to use in FPN')
        self.parser.add_argument('--cascaded', action='store_true', help='if true, use cascaded design for detection')  
        self.parser.add_argument('--use_attention', action='store_true', help='if true, use Attention in generator. Currently support Attenion G with skip connection and resnet backbone. You can choose encoder or decoder feature for detection branch.')  
        self.parser.add_argument('--attention_layers', type=str, default='0,1,2,3,4', help='specify which layers to use CBAM')  

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.parser.add_argument('--feat_num', type=int, default=1, help='vector length for encoded features')        
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')       

        # for region proposal network
        self.parser.add_argument('--anchor_sizes', type=str, default='4,8,16,32,64', help='anchor box sizes, correspond to feature maps from decoder')
        self.parser.add_argument('--aspect_ratios', type=str, default='0.5,1.0,2.0', help='aspect ratios for anchors')
        self.parser.add_argument('--rpn_pre_nms_top_n_train', type=int, default=2000, help='number of top scoring RPN proposals to keep before applying NMS, training')
        self.parser.add_argument('--rpn_pre_nms_top_n_test', type=int, default=1000, help='number of top scoring RPN proposals to keep before applying NMS, testing')
        self.parser.add_argument('--rpn_post_nms_top_n_train', type=int, default=2000, help='number of top scoring RPN proposals to keep after applying NMS, training')
        self.parser.add_argument('--rpn_post_nms_top_n_test', type=int, default=1000, help='number of top scoring RPN proposals to keep after applying NMS, testing')
        self.parser.add_argument('--rpn_fg_iou_thresh', type=float, default=0.7, help='minimum overlap required between an anchor and groundtruth box to be a positive example')
        self.parser.add_argument('--rpn_bg_iou_thresh', type=float, default=0.3, help='maximum overlap allowed between an anchor and groundtruth box to be a negative example')
        self.parser.add_argument('--rpn_batch_size_per_image', type=int, default=256, help='number of regions per image used to train RPN')
        self.parser.add_argument('--rpn_positive_fraction', type=float, default=0.5, help='target fraction of foreground (positive) examples per RPN minibatch')
        self.parser.add_argument('--rpn_nms_thresh', type=float, default=0.7, help='NMS threshold used on RPN proposals')
        self.parser.add_argument('--rpn_score_thresh', type=float, default=0.0, help='minimum score threshold (assuming scores in a [0, 1] range); a value chosen to balance obtaining high recall with not having too many low precision detections that will slow down inference post processing steps (like NMS)')

        # for roi heads
        self.parser.add_argument('--num_classes', type=int, default=2, help='number of classes to detect')
        self.parser.add_argument('--box_fg_iou_thresh', type=float, default=0.5, help='overlap threshold for an RoI to be considered foreground')
        self.parser.add_argument('--box_bg_iou_thresh', type=float, default=0.5, help='overlap threshold for an RoI to be considered background')
        self.parser.add_argument('--box_score_thresh', type=float, default=0.05, help='minimum score threshold (assuming scores in a [0, 1] range)')
        self.parser.add_argument('--box_nms_thresh', type=float, default=0.1, help='overlap threshold used for non-maximum suppression')
        self.parser.add_argument('--box_detections_per_img', type=int, default=1000, help='maximum number of detections to return per image during inference')
        self.parser.add_argument('--box_batch_size_per_image', type=int, default=512, help='RoI minibatch size *per image* (number of regions of interest [ROIs]) during training')
        self.parser.add_argument('--box_positive_fraction', type=float, default=0.25, help='target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)')
        self.parser.add_argument('--bbox_reg_weights', default=None, help='Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets. These are empirically chosen to approximately lead to unit variance targets') 
        self.parser.add_argument('--score_thresh_test', type=float, default=0.5, help='minimum score threshold (assuming scores in a [0, 1] range) for inference')

        self.initialized = True

    def parse(self, save=True, set_GPU=True):
        if not self.initialized:
            self.initialize()
        # self.opt = self.parser.parse_args()
        self.opt, unknown = self.parser.parse_known_args()
        # print("==================Opt early", self.opt)
        self.opt.isTrain = self.isTrain   # train or test

        anchor_sizes = self.opt.anchor_sizes.split(',')
        self.opt.anchor_sizes = []
        for anchor_size in anchor_sizes:
            self.opt.anchor_sizes.append(int(anchor_size))

        aspect_ratios = self.opt.aspect_ratios.split(',')
        self.opt.aspect_ratios = []
        for aspect_ratio in aspect_ratios:
            self.opt.aspect_ratios.append(float(aspect_ratio))

        attention_layers = self.opt.attention_layers.split(',')
        self.opt.attention_layers = []
        for attention_layer in attention_layers:
            self.opt.attention_layers.append(attention_layer)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.local_rank)
            self.opt.gpu_device = torch.device(self.opt.local_rank)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0 and set_GPU:
            torch.distributed.init_process_group(backend='nccl')
            torch.distributed.barrier()
            setup_for_distributed(self.opt.local_rank == 0)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.yaml')
            with open(file_name, 'wt') as opt_file:
                # yaml.dump(args, opt_file)
                # opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                # opt_file.write('-------------- End ----------------\n')

        # print("opt", self.opt)

        return self.opt
