import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import collections
import pdb

from typing import Dict
from collections import OrderedDict

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .rpn import RegionProposalNetwork_FocalLoss
# from torchvision.models.detection.roi_heads import RoIHeads
from .roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models import resnet50

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm='instance', gpu_device=torch.device("cuda:0")):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    # print(netG)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())   
    #     # netG.cuda(gpu_ids[0])
    netG.to(gpu_device)
    netG.apply(weights_init)
    return netG

def define_G_ResNet(netG, skipCon, return_feature=True, use_Attn=False, attn_layers=['0','1','2','3','4'], fpn_feature='decoder', norm='instance', gpu_device=torch.device("cuda:0")):
    norm_layer = get_norm_layer(norm_type=norm)
    if use_Attn:
        if len(attn_layers) == 5:
            netG = AttGenerator_ResNet50_SkipCon(norm_layer, fpn_feature)
        else:
            netG = AttGenerator(norm_layer, fpn_feature, attn_layers)
    elif return_feature:
        if netG == 'global' and not skipCon:
            if fpn_feature == 'encoder':
                netG = GlobalGenerator_ResNet50_encoderFeature(norm_layer, fpn_feature)
            elif fpn_feature == 'decoder':
                netG = GlobalGenerator_ResNet50_decoderFeature(norm_layer, fpn_feature)
        elif netG == 'global' and skipCon:
            netG = GlobalGenerator_ResNet50_SkipCon(norm_layer, fpn_feature)
        else:
            raise('generator not implemented!')
    else:
        if netG == 'global':
            netG = GlobalGenerator_ResNet50(norm_layer)
        else:
            raise('generator not implemented!')
    netG.to(gpu_device)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_device=torch.device("cuda:0")):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    # print(netD)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     netD.cuda(gpu_ids[0])
    netD.to(gpu_device)
    netD.apply(weights_init)
    return netD

def define_backbone(opt, extra_blocks=None):
    channels_list = [256, 512, 1024, 2048]
    return_layers = {f"layer{k}": str(v) for v, k in enumerate([1,2,3,4])}
    backbone = BackboneWithFPN(resnet50(pretrained=True), return_layers=return_layers, in_channels_list=channels_list, out_channels=opt.ngf, extra_blocks=extra_blocks)
    return backbone

def define_FPN(opt, extra_blocks=None):
    if opt.use_resnet_as_backbone:
        channels_list = [256, 512, 1024, 2048]
        fpn = FeaturePyramidNetwork(channels_list, opt.ngf, extra_blocks=extra_blocks)
    else:
        if opt.netG == 'global':
            channels_list = [opt.ngf*(2**i) for i in range(opt.n_downsample_global+1)]
            fpn = FeaturePyramidNetwork(channels_list, opt.ngf)
        elif opt.netG == 'local':
            channels_list = []
            for i in range(1, opt.n_local_enhancers+1):
                channels_list.append(opt.ngf*(2**i))
                # channels_list.append(opt.ngf*(2**(i+1)))
            channels_list += [opt.ngf*(2**i) for i in range(opt.n_local_enhancers+1, opt.n_downsample_global+opt.n_local_enhancers+1)]
            fpn = FeaturePyramidNetwork(channels_list, opt.ngf*2)
    return fpn

def define_RPN(opt):
    anchor_sizes = tuple((s,) for s in opt.anchor_sizes)
    aspect_ratios = (tuple(opt.aspect_ratios),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    ngf = opt.ngf if opt.netG == 'global' else opt.ngf*2
    rpn_head = RPNHead(ngf, rpn_anchor_generator.num_anchors_per_location()[0])

    rpn_pre_nms_top_n = dict(training=opt.rpn_pre_nms_top_n_train, testing=opt.rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=opt.rpn_post_nms_top_n_train, testing=opt.rpn_post_nms_top_n_test)

    if opt.use_focal_loss:
        rpn = RegionProposalNetwork_FocalLoss(rpn_anchor_generator, rpn_head, 
                                    opt.rpn_fg_iou_thresh, opt.rpn_bg_iou_thresh, 
                                    opt.rpn_batch_size_per_image, opt.rpn_positive_fraction,
                                    rpn_pre_nms_top_n, rpn_post_nms_top_n, opt.rpn_nms_thresh,
                                    score_thresh=opt.rpn_score_thresh, focal_loss_gamma=opt.focal_loss_gamma)
    else:
        rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head, 
                                    opt.rpn_fg_iou_thresh, opt.rpn_bg_iou_thresh, 
                                    opt.rpn_batch_size_per_image, opt.rpn_positive_fraction,
                                    rpn_pre_nms_top_n, rpn_post_nms_top_n, opt.rpn_nms_thresh,
                                    score_thresh=opt.rpn_score_thresh)
    return rpn

def define_RoIHeads(opt):
    if opt.netG == 'global':
        box_roi_pool = MultiScaleRoIAlign(featmap_names=['feat'+str(i) for i in reversed(range(opt.n_downsample_global+1))] + ['pool']*opt.use_resnet_as_backbone, output_size=7, sampling_ratio=2)
        ngf = opt.ngf
    elif opt.netG == 'local':
        box_roi_pool = MultiScaleRoIAlign(featmap_names=['feat'+str(i) for i in reversed(range(opt.n_downsample_global+opt.n_local_enhancers))] + ['pool']*opt.use_resnet_as_backbone, output_size=7, sampling_ratio=2)
        ngf = opt.ngf * (2**(opt.n_local_enhancers))
    resolution = box_roi_pool.output_size[0]
    box_head = TwoMLPHead(ngf * resolution ** 2, representation_size=1024)
    box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=opt.num_classes)
    if opt.netG == 'global':
        mask_roi_pool = MultiScaleRoIAlign(featmap_names=['feat'+str(i) for i in reversed(range(opt.n_downsample_global+1))] + ['pool']*opt.use_resnet_as_backbone, output_size=14, sampling_ratio=2)
    elif opt.netG == 'local':
        mask_roi_pool = MultiScaleRoIAlign(featmap_names=['feat'+str(i) for i in reversed(range(opt.n_downsample_global+opt.n_local_enhancers))] + ['pool']*opt.use_resnet_as_backbone, output_size=14, sampling_ratio=2)
    mask_head = MaskRCNNHeads(ngf, layers=(256, 256, 256, 256), dilation=1)
    mask_predictor_in_channels = 256 # == mask_layers[-1]
    mask_dim_reduced = 256
    mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, opt.num_classes)
    roi_heads = RoIHeads(
                        # Box
                        box_roi_pool, box_head, box_predictor,
                        # Faster RCNN training
                        opt.box_fg_iou_thresh, opt.box_bg_iou_thresh,
                        opt.box_batch_size_per_image, opt.box_positive_fraction,
                        opt.bbox_reg_weights,
                        # Faster RCNN inference
                        opt.box_score_thresh, opt.box_nms_thresh, opt.box_detections_per_img,
                        # Mask
                        mask_roi_pool, mask_head, mask_predictor,
                        # Loss
                        use_focal=opt.focal_loss_gamma if opt.use_focal_loss else 0)
    return roi_heads

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        self.n_downsample_global = n_downsample_global
        self.n_blocks_global = n_blocks_global
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            if n != n_local_enhancers:
                model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_global), nn.ReLU(True)]                      
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))
        
        ### final convolution
        if n == n_local_enhancers:                
            self.final_upsample = nn.Sequential(*[
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_global), nn.ReLU(True), 
                nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()
                ])
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        # output_prev = self.model(input_downsampled[-1]) 
        output_prev, tempfeat = self.get_output_and_features(input_downsampled[-1])       
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]     
            # tempfeat['feat'+str(self.n_downsample_global+2*n_local_enhancers-1)] = model_downsample(input_i) + output_prev  
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
            tempfeat['feat'+str(self.n_downsample_global+n_local_enhancers-1)] = output_prev
        output_prev = self.final_upsample(output_prev)
        features = collections.OrderedDict(reversed(list(tempfeat.items())))
        return output_prev, features

    def get_output_and_features(self, input):
        # Feature maps in decoder, n_downsampling+1 layers
        features = dict()
        x = input
        for i in range(len(self.model)):
            x = self.model[i](x)
            if i >= 3+3*self.n_downsample_global+self.n_blocks_global and (i-3-3*self.n_downsample_global-self.n_blocks_global) % 3 == 0:
                features['feat'+str((i-3-3*self.n_downsample_global-self.n_blocks_global) // 3)] = x
        return x, features  

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        
        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        # Feature maps in decoder, n_downsampling+1 layers
        tempfeat = dict()
        x = input
        for i in range(len(self.model)):
            if i >= 4+3*self.n_downsampling+self.n_blocks and (i-4-3*self.n_downsampling-self.n_blocks) % 3 == 0:
                tempfeat['feat'+str((i-4-3*self.n_downsampling-self.n_blocks) // 3)] = x
            x = self.model[i](x)
        features = collections.OrderedDict(reversed(list(tempfeat.items())))
        # features = collections.OrderedDict(list(tempfeat.items()))
        # print(features.keys(), features['feat0'].shape)
        return x, features       

class GlobalGenerator_ResNet50_encoderFeature(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, fpn_feature='encoder'):
        super(GlobalGenerator_ResNet50_encoderFeature, self).__init__()        

        backbone = resnet50(pretrained=True)
        self.model = IntermediateLayerGetter(backbone, return_layers={"layer1": 'feat0', "layer2": 'feat1', "layer3": 'feat2', "layer4": 'feat3'})
        
        decoder = []
        n_downsampling = 5
        ngf = 64
        activation = nn.ReLU(True)
        output_nc = 3
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)
            
    def forward(self, input):
        x, features = self.model(input)
        x = self.decoder(x)
        return x, features              

class GlobalGenerator_ResNet50_decoderFeature(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, fpn_feature='decoder'):
        super(GlobalGenerator_ResNet50_decoderFeature, self).__init__()        

        backbone = resnet50(pretrained=True)
        self.encoder = IntermediateLayerGetter(backbone, return_layers={"layer1": 'feat0', "layer2": 'feat1', "layer3": 'feat2', "layer4": 'feat3'})
        
        assert fpn_feature == 'encoder' or fpn_feature == 'decoder'
        self.fpn_feature = fpn_feature

        decoder = []
        self.n_downsampling = 5
        self.ngf = 64
        activation = nn.ReLU(True)
        output_nc = 3
        for i in range(self.n_downsampling-1):
            mult = 2**(self.n_downsampling - i)
            setattr(self, 'upsample_'+str(i), nn.Sequential(nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(self.ngf * mult / 2)), activation))
        self.final_upsample = nn.Sequential(nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=3, stride=2, padding=1, output_padding=1), 
            norm_layer(self.ngf), nn.ReLU(True), nn.ReflectionPad2d(3), nn.Conv2d(self.ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())
            
    def forward(self, input):
        x, features = self.encoder(input) # features: dict{'feat0': 64x64x256, 'feat1': 32x32x512, 'feat2': 16x16x1024, 'feat3': 8x8x2048} x: 8x8x2048
        for i in range(self.n_downsampling-1):
            model = getattr(self, 'upsample_'+str(i))
            x = model(x)
            if self.fpn_feature == 'decoder' and self.n_downsampling-i-3>=0:
                feature_key = 'feat' + str(self.n_downsampling-i-3)
                features[feature_key] = x
        x = self.final_upsample(x)
        return x, features            

class GlobalGenerator_ResNet50(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(GlobalGenerator_ResNet50, self).__init__()        

        backbone = resnet50(pretrained=True)
        self.encoder = IntermediateLayerGetter(backbone, return_layers={"layer1": 'feat0', "layer2": 'feat1', "layer3": 'feat2', "layer4": 'feat3'})

        self.n_downsampling = 5
        self.ngf = 64
        activation = nn.ReLU(True)
        output_nc = 3
        for i in range(self.n_downsampling-1):
            mult = 2**(self.n_downsampling - i)
            setattr(self, 'upsample_'+str(i), nn.Sequential(nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(self.ngf * mult / 2)), activation))
        self.final_upsample = nn.Sequential(nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=3, stride=2, padding=1, output_padding=1), 
            norm_layer(self.ngf), nn.ReLU(True), nn.ReflectionPad2d(3), nn.Conv2d(self.ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())
            
    def forward(self, input):
        x, _ = self.encoder(input) # features: dict{'feat0': 64x64x256, 'feat1': 32x32x512, 'feat2': 16x16x1024, 'feat3': 8x8x2048} x: 8x8x2048
        for i in range(self.n_downsampling-1):
            model = getattr(self, 'upsample_'+str(i))
            x = model(x)
        x = self.final_upsample(x)
        return x, None           

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return x, out

class GlobalGenerator_ResNet50_SkipCon(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, fpn_feature='decoder'):
        super(GlobalGenerator_ResNet50_SkipCon, self).__init__()        

        backbone = resnet50(pretrained=True)
        self.encoder = IntermediateLayerGetter(backbone, return_layers={"conv1": 'feat', "layer1": 'feat0', "layer2": 'feat1', "layer3": 'feat2', "layer4": 'feat3'})

        assert fpn_feature == 'encoder' or fpn_feature == 'decoder'
        self.fpn_feature = fpn_feature
        self.ngf = 64
        self.n_downsampling = 5
        self.output_nc = 3
        for i in range(self.n_downsampling - 1):
            mult = 2**(self.n_downsampling - i)
            setattr(self, 'upsample_'+str(i), Up(self.ngf * mult, int(self.ngf * mult / 2) if i!=self.n_downsampling-2 else self.ngf))
        
        self.final_upsample = nn.Sequential(nn.ConvTranspose2d(self.ngf, self.ngf // 2, kernel_size=3, stride=2, padding=1, output_padding=1), 
            norm_layer(self.ngf // 2), nn.ReLU(True), nn.ReflectionPad2d(3), 
            nn.Conv2d(self.ngf // 2, self.output_nc, kernel_size=7, padding=0), nn.Tanh())
            
    def forward(self, input):
        x, features = self.encoder(input) # features: dict{'feat': 128x128x64, feat0': 64x64x256, 'feat1': 32x32x512, 'feat2': 16x16x1024, 'feat3': 8x8x2048} x: 8x8x2048
        for i in range(self.n_downsampling - 1):
            model = getattr(self, 'upsample_'+str(i))
            feature_key = 'feat' + (str(self.n_downsampling-i-3) if self.n_downsampling-i-3>=0 else '')
            x = model(x, features[feature_key])
            if self.fpn_feature == 'decoder':
                features[feature_key] = x
        x = self.final_upsample(x)
        del features['feat']
        return x, features    

class AttGenerator_ResNet50_SkipCon(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, fpn_feature='decoder'):
        super(AttGenerator_ResNet50_SkipCon, self).__init__()        

        backbone = resnet50(pretrained=True)
        self.encoder = IntermediateLayerGetter(backbone, return_layers={"conv1": 'feat', "layer1": 'feat0', "layer2": 'feat1', "layer3": 'feat2', "layer4": 'feat3'})

        assert fpn_feature == 'encoder' or fpn_feature == 'decoder'
        self.fpn_feature = fpn_feature
        self.ngf = 64
        self.n_downsampling = 5
        self.output_nc = 3
        for i in range(self.n_downsampling - 1):
            mult = 2**(self.n_downsampling - i)
            setattr(self, 'upsample_'+str(i), Up(self.ngf * mult, int(self.ngf * mult / 2) if i!=self.n_downsampling-2 else self.ngf))
            setattr(self, 'cbam_'+str(i), CBAM(self.ngf * mult))
        setattr(self, 'cbam_'+str(self.n_downsampling-1), CBAM(self.ngf))
        
        self.final_upsample = nn.Sequential(nn.ConvTranspose2d(self.ngf, self.ngf // 2, kernel_size=3, stride=2, padding=1, output_padding=1), 
            norm_layer(self.ngf // 2), nn.ReLU(True), nn.ReflectionPad2d(3), 
            nn.Conv2d(self.ngf // 2, self.output_nc, kernel_size=7, padding=0), nn.Tanh())
            
    def forward(self, input):
        x, features = self.encoder(input) # features: dict{'feat': 128x128x64, feat0': 64x64x256, 'feat1': 32x32x512, 'feat2': 16x16x1024, 'feat3': 8x8x2048} x: 8x8x2048
        cbam = getattr(self, 'cbam_0') # 2048
        x = cbam(x)
        for i in range(self.n_downsampling - 1):
            model = getattr(self, 'upsample_'+str(i))
            feature_key = 'feat' + (str(self.n_downsampling-i-3) if self.n_downsampling-i-3>=0 else '')
            cbam = getattr(self, 'cbam_'+str(i+1))
            feat = cbam(features[feature_key])
            x = model(x, feat)
            # features for detection. encoder: use features after cbam.
            if self.fpn_feature == 'decoder':
                features[feature_key] = x
            elif self.fpn_feature == 'encoder':
                features[feature_key] = feat
        x = self.final_upsample(x)
        del features['feat']
        return x, features 

class AttGenerator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, fpn_feature='decoder', attention_layers=['0','1','2','3','4']):
        super(AttGenerator, self).__init__()        

        backbone = resnet50(pretrained=True)
        self.encoder = IntermediateLayerGetter(backbone, return_layers={"conv1": 'feat', "layer1": 'feat0', "layer2": 'feat1', "layer3": 'feat2', "layer4": 'feat3'})

        assert fpn_feature == 'encoder' or fpn_feature == 'decoder'
        self.fpn_feature = fpn_feature
        self.ngf = 64
        self.n_downsampling = 5
        self.output_nc = 3
        self.attention_layers = attention_layers
        for i in range(self.n_downsampling - 1):
            mult = 2**(self.n_downsampling - i)
            setattr(self, 'upsample_'+str(i), Up(self.ngf * mult, int(self.ngf * mult / 2) if i!=self.n_downsampling-2 else self.ngf))
        for i in range(self.n_downsampling):
            if str(i) not in self.attention_layers:
                continue
            if i != self.n_downsampling - 1:
                mult = 2**(self.n_downsampling - i)
            else:
                mult = 1
            setattr(self, 'cbam_'+str(i), CBAM(self.ngf * mult))
        
        self.final_upsample = nn.Sequential(nn.ConvTranspose2d(self.ngf, self.ngf // 2, kernel_size=3, stride=2, padding=1, output_padding=1), 
            norm_layer(self.ngf // 2), nn.ReLU(True), nn.ReflectionPad2d(3), 
            nn.Conv2d(self.ngf // 2, self.output_nc, kernel_size=7, padding=0), nn.Tanh())
            
    def forward(self, input):
        x, features = self.encoder(input) # features: dict{'feat': 128x128x64, feat0': 64x64x256, 'feat1': 32x32x512, 'feat2': 16x16x1024, 'feat3': 8x8x2048} x: 8x8x2048
        if '0' in self.attention_layers:
            cbam = getattr(self, 'cbam_0') # 2048
            x = cbam(x)
        for i in range(self.n_downsampling - 1):
            model = getattr(self, 'upsample_'+str(i))
            feature_key = 'feat' + (str(self.n_downsampling-i-3) if self.n_downsampling-i-3>=0 else '')
            feat = features[feature_key]
            if str(i+1) in self.attention_layers:
                cbam = getattr(self, 'cbam_'+str(i+1))
                feat = cbam(feat)
            x = model(x, feat)
            # features for detection. encoder: use features after cbam.
            if self.fpn_feature == 'decoder':
                features[feature_key] = x
            elif self.fpn_feature == 'encoder':
                features[feature_key] = feat
        x = self.final_upsample(x)
        del features['feat']
        return x, features  

# Define CBAM block
class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    Flatten(),
                    nn.Linear(input_channels, input_channels//reduction_ratio),
                    nn.ReLU(),
                    nn.Linear(input_channels//reduction_ratio, input_channels))

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.mlp(avg_values) + self.mlp(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.bn(self.conv(out))
        scale = x * torch.sigmoid(out)
        return scale

# Define an Attention block in Attention U-Net
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x*psi


# Define a Unet Up-sample block
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(Up, self).__init__()
        activation = nn.ReLU(True)
        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(out_channels), activation)
        self.conv = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1), norm_layer(out_channels), activation, nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), norm_layer(out_channels), activation)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = self.conv(torch.cat([x2, x1], dim=1))
        return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc    
        self.n_downsampling = n_downsampling    

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        # outputs = self.model(input)
        # Feature maps in decoder, n_downsampling+1 layers
        features = collections.OrderedDict()
        outputs = input
        for i in range(len(self.model)):
            if i >= 4+3*self.n_downsampling and (i-4-3*self.n_downsampling) % 3 == 0:
                features['feat'+str((i-4-3*self.n_downsampling) // 3)] = outputs
            outputs = self.model[i](outputs)   

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean, features

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
