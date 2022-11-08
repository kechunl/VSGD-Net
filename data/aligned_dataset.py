import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from pycocotools.mask import decode as decode_mask
import torch
import numpy as np
import json
import copy
from torchvision import transforms
import pdb

class AlignedDataset_Detection(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A), reverse=True)

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image or opt.has_real_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B), reverse=True)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst), reverse=True)

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat), reverse=True)

        ### detection targets:
        if opt.isTrain or opt.has_real_image:
            json_file = os.path.join(opt.dataroot, opt.phase+opt.data_suffix+'.json')
            with open(json_file) as f:
                detect_dicts = json.load(f)
            self.detect_dicts = sorted(detect_dicts, key=lambda d: os.path.basename(d["file_name"]).split('.')[0], reverse=True)

        # filter empty patches
        if self.opt.filter_empty:
            keep_indices = []
            for i in range(len(self.A_paths)):
                if len(keep_indices) > opt.max_dataset_size: # to make a small subset
                    break
                if len(self.detect_dicts[i]['target']['labels']) > 0:
                    keep_indices.append(i)
            keep_indices = np.array(keep_indices)
            self.A_paths = list(np.array(self.A_paths)[keep_indices])
            if hasattr(self, 'B_paths'):
                self.B_paths = list(np.array(self.B_paths)[keep_indices])
            if hasattr(self, 'inst_paths'):
                self.inst_paths = list(np.array(self.inst_paths)[keep_indices])
            if hasattr(self, 'feat_paths'):
                self.feat_paths = list(np.array(self.feat_paths)[keep_indices])
            if hasattr(self, 'detect_dicts'):
                self.detect_dicts = list(np.array(self.detect_dicts)[keep_indices])

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image or self.opt.has_real_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            transform_inst = get_transform(self.opt, params, normalize=False)
            inst_tensor = transform_inst(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))      

        ### detection
        target = {}
        if self.opt.isTrain or self.opt.has_real_image:
            target = copy.deepcopy(self.detect_dicts[index]["target"])
            height, width = self.detect_dicts[index]["height"], self.detect_dicts[index]["width"]
            target = self.transform_det(params, target, height, width, index)

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path, 'target': target}

        return input_dict

    def transform_det(self, params, target, height, width, idx):
        if len(target["boxes"]) > 0:
            target["masks"] = np.transpose(decode_mask(target["masks"]), (2,0,1)) # Nx256x256
            if 'resize' in self.opt.resize_or_crop:
                osize = self.opt.loadSize
                h_scale, w_scale = osize / height, osize / width
                target["boxes"] = np.array(target["boxes"]) * np.array([h_scale, w_scale, h_scale, w_scale])
                target["masks"] = transforms.Resize([osize, osize])(torch.as_tensor(target["masks"])).numpy()
                # target["masks"] = np.array([np.array(Image.fromarray(mask).resize((osize, osize))) for mask in target["masks"]])
                # target["masks"] = np.array([cv2.resize(np.squeeze(mask), (osize, osize)) for mask in target["masks"]])
                # target["masks"] = ndimage.zoom(target["masks"], (1,h_scale,w_scale)) # too slow
            elif 'scale_width' in self.opt.resize_or_crop:
                osize = self.opt.loadSize
                h_scale, w_scale = osize / width, osize / width
                target["boxes"] = np.array(target["boxes"]) * np.array([h_scale, w_scale, h_scale, w_scale])
                target["masks"] = np.resize(np.array(target["masks"]), (target["boxes"].shape[0], h_scale*height, osize))

            if 'crop' in self.opt.resize_or_crop:
                # TODO: implement crop for detections
                raise NotImplementedError
                # x1, y1 = params['crop_pos']
                # x2, y2 = x1 + opt.fineSize, y1 + opt.fineSize
                # for i in range(len(target['boxes'])):

            if self.opt.resize_or_crop == 'none':
                base = float(2 ** self.opt.n_downsample_global)
                osize = self.opt.loadSize
                if self.opt.netG == 'local':
                    base *= (2 ** self.opt.n_local_enhancers)
                h_scale, w_scale = int(round(height / base) * base) / height, int(round(width / base) * base) / width 
                target["boxes"] = np.array(target["boxes"]) * np.array([h_scale, w_scale, h_scale, w_scale])
                target["masks"] = np.resize(np.array(target["masks"]), (target["boxes"].shape[0], osize, osize))

            if self.opt.isTrain and not self.opt.no_flip:
                if params['flip']:
                    boxes_pos = np.array(target['boxes'])
                    target['boxes'] = np.hstack((self.opt.loadSize - boxes_pos[:,2][:,np.newaxis], boxes_pos[:,1][:,np.newaxis], self.opt.loadSize - boxes_pos[:,0][:,np.newaxis], boxes_pos[:,3][:,np.newaxis]))
                    target["masks"] = np.flip(np.array(target["masks"]), 2)
        else:
            target["boxes"] = np.zeros([0, 4], dtype=np.float32)
            target["masks"] = np.zeros([0, self.opt.loadSize, self.opt.loadSize], dtype=np.uint8)

        target["boxes"] = torch.as_tensor(np.array(target["boxes"]), dtype=torch.float32)
        target["labels"] = torch.as_tensor(np.array(target["labels"]), dtype=torch.int64)
        target["masks"] = torch.as_tensor(np.array(target["masks"]), dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx])
        boxes = target["boxes"]
        if len(boxes) != 0:
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            target["area"] = torch.as_tensor(np.array([]), dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(np.array([]), dtype=torch.int64)

        return target


    def __len__(self):
        num = len(self.opt.gpu_ids) if len(self.opt.gpu_ids) != 0  else 1
        return len(self.A_paths) // (self.opt.batchSize*num) * self.opt.batchSize * num

    def name(self):
        return 'AlignedDataset_Detection'

def collate_func(batch_list):
    data = {}
    data['label'] = torch.stack([b['label'] for b in batch_list], dim=0)
    if not isinstance(batch_list[0]['image'], int):
        data['image'] = torch.stack([b['image'] for b in batch_list], dim=0)
        data['target'] = [b['target'] for b in batch_list]
    else:
        data['image'] = torch.as_tensor([b['feat'] for b in batch_list])
        data['target'] = None
    data['feat'] = torch.as_tensor([b['feat'] for b in batch_list])
    data['inst'] = torch.as_tensor([b['inst'] for b in batch_list])
    data['path'] = [b['path'] for b in batch_list]
    return data


