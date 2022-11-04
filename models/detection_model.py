import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from .transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

import pdb

class Detection_Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Detection_Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, detect, recon):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake, detect, recon),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        if opt.use_resnet_as_backbone or opt.use_attention:
            self.netG = networks.define_G_ResNet(opt.netG, skipCon=opt.use_UNet_skip, return_feature=not opt.cascaded, use_Attn=opt.use_attention, attn_layers=opt.attention_layers, fpn_feature=opt.fpn_feature, norm=opt.norm, gpu_device=self.opt.gpu_device)
        else:
            netG_input_nc = input_nc
            if not opt.no_instance:
                netG_input_nc += 1
            if self.use_features:
                netG_input_nc += opt.feat_num                  
            self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                          opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                          opt.n_blocks_local, opt.norm, gpu_device=self.opt.gpu_device)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_device=self.opt.gpu_device)

        # Feature pyramid network
        if opt.use_resnet_as_backbone:
            extra_blocks = LastLevelMaxPool()
        else:
            extra_blocks = None

        if opt.cascaded:
            self.fpn = networks.define_backbone(opt, extra_blocks=extra_blocks)
        else:
            self.fpn = networks.define_FPN(opt, extra_blocks=extra_blocks)

        # Region proposal network
        self.rpn = networks.define_RPN(opt)

        # ROI heads
        self.roi_heads = networks.define_RoIHeads(opt)
        self.transform_detection = GeneralizedRCNNTransform(opt.loadSize/2, opt.loadSize*2, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]) # the parameters won't be used, so set them arbitrarily

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_device=self.opt.gpu_device)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.netG, opt.which_epoch, pretrained_path)
            # self.load_network(self.netG, 'G', 'global', opt.which_epoch, pretrained_path)     
            self.load_network(self.fpn, 'FPN', opt.netG, opt.which_epoch, pretrained_path)  
            self.load_network(self.rpn, 'RPN', opt.netG, opt.which_epoch, pretrained_path)  
            self.load_network(self.roi_heads, 'RoIheads', opt.netG, opt.which_epoch, pretrained_path)       
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.netG, opt.which_epoch, pretrained_path)  
                # self.load_network(self.netD, 'D', 'global', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.netG, opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionRecon = torch.nn.L1Loss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'Detect', 'Recon')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params_G = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params_G += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))  
            else:
                params_G = list(self.netG.parameters())
            if self.gen_features:              
                params_G += list(self.netE.parameters())  
            pnames_G = [n for (n, p) in self.netG.named_parameters()]
            self.optimizer_G = torch.optim.Adam(params_G, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params_D = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params_D, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer RPN + RoI heads
            if opt.niter_fix_global > 0:                
                params_detect = self.fpn.inner_blocks[-2:] + self.fpn.layer_blocks[-2,:]
                print('------------- Only training the fpn layers in local network (for %d epochs) ------------' % opt.niter_fix_global)                       
            else:
                params_detect = list(self.fpn.parameters()) + list(self.rpn.parameters()) + list(self.roi_heads.parameters())
            pnames_detect = [n for (n,p) in self.fpn.named_parameters()] +[n for (n,p) in self.rpn.named_parameters()] + [n for (n,p) in self.roi_heads.named_parameters()]

            detect_lr_milestone = [50, 80, 110, 140, 170, 200, 240, 270]
            if opt.start_epoch != 1:
                decay = detect_lr_milestone.index([m for m in detect_lr_milestone if m>opt.start_epoch][0])
                cur_lr_detect = opt.lr_detect * (0.5 ** decay)
            else:
                cur_lr_detect = opt.lr_detect
            self.optimizer_detect = torch.optim.SGD([{'params': params_detect, 'initial_lr': opt.lr_detect}], lr=cur_lr_detect)
            self.scheduler_detect = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_detect, milestones=detect_lr_milestone, gamma=0.5, last_epoch=opt.start_epoch-1)

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, target=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.to(self.opt.gpu_device)
            # input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().to(self.opt.gpu_device), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        inst_map = inst_map.data.to(self.opt.gpu_device)
        if not self.opt.no_instance:
            # inst_map = inst_map.data.to(self.opt.gpu_device)
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)         
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data).to(self.opt.gpu_device)

        # instance map for feature encoding
        feat_map = feat_map.data.to(self.opt.gpu_device)
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data).to(self.opt.gpu_device)
            if self.opt.label_feat:
                inst_map = label_map.to(self.opt.gpu_device)
        if target is not None:
            target = [{k: v.to(self.opt.gpu_device) for k, v in t.items()} for t in target]
        image_sizes = [(input_label.shape[2], input_label.shape[3])] * input_label.shape[0]

        return input_label, inst_map, real_image, feat_map, target, (image_sizes)

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, targets, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map, targets, image_sizes = self.encode_input(label, inst, image, feat, targets)  

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)                     
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label
        fake_image, features = self.netG.forward(input_concat)

        # Detection Branch
        # features = {k: v.detach() for k, v in features.items()}
        if self.opt.cascaded:
            features = self.fpn(fake_image)
        else:
            features = self.fpn(features)
        proposals, proposal_losses = self.rpn(ImageList(input_label.detach(), image_sizes), features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
        outputs = self.transform_detection.postprocess(detections, image_sizes, image_sizes)
        outputs = [{k: v.to(torch.device("cpu")).detach() for k, v in t.items()} for t in outputs]

        losses_detect = {}
        losses_detect.update(proposal_losses)
        losses_detect.update(detector_losses)

        # Reconstruction Loss
        loss_recon = self.criterionRecon(fake_image, real_image)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, losses_detect, loss_recon ), None if not infer else fake_image.detach(), None if not infer else outputs ]

    def inference(self, label, inst, image=None, feat=None, targets=None):
        # Encode Inputs        
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _, targets, image_sizes = self.encode_input(Variable(label), Variable(inst), image, feat, targets, infer=True)
        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features             
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label        
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image, features = self.netG.forward(input_concat)
        else:
            fake_image, features = self.netG.forward(input_concat)

        # Detection
        if self.opt.cascaded:
            features = self.fpn(fake_image)
        else:
            features = self.fpn(features) # B x 32(ngf*2) x (16x16, 32x32, 64x64, 128x128, 256x256)
        proposals, proposal_losses = self.rpn(ImageList(input_label, image_sizes), features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
        outputs = self.transform_detection.postprocess(detections, image_sizes, image_sizes)

        # Reconstruction loss
        losses = {}
        if self.training:
            loss_recon = self.criterionRecon(fake_image, real_image) if image is not None else 0
            losses.update(proposal_losses)
            losses.update(detector_losses)
            losses.update({'loss_recon': loss_recon})

        return [fake_image, outputs, losses]

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.to(self.opt.gpu_device), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.to(self.opt.gpu_device))
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch, prefix):
        self.save_network(self.netG, prefix + 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, prefix + 'D', which_epoch, self.gpu_ids)
        self.save_network(self.fpn, prefix + 'FPN', which_epoch, self.gpu_ids)
        self.save_network(self.rpn, prefix + 'RPN', which_epoch, self.gpu_ids)
        self.save_network(self.roi_heads, prefix+ 'RoIheads', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, prefix + 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        params = list(self.fpn.parameters()) + list(self.rpn.parameters()) + list(self.roi_heads.parameters())
        self.optimizer_detect = torch.optim.SGD(params, lr=opt.lr_detect)
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd        
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            if self.opt.verbose:
                print('update learning rate: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr
        self.scheduler_detect.step()

class Detection_InferenceModel(Detection_Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
