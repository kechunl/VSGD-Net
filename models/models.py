import torch
import os

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'detection':
        from .detection_model import Detection_Pix2PixHDModel, Detection_InferenceModel
        if opt.isTrain:
            model = Detection_Pix2PixHDModel()
        else:
            model = Detection_InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.to(opt.gpu_device))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank])
        # model.to(opt.gpu_device)
    else:
        model.to(opt.gpu_device)
        # model = model.cuda()

    return model
