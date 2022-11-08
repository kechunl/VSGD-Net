import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util.visualize_detection import display_instances
from util import html
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.distributed as dist

import warnings
warnings.filterwarnings('ignore')

import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.no_html = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt, opt.phase)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx


@torch.no_grad()
def evaluate(model, dataloader, opt, visualizer, webpage):
    model.eval()
    cpu_device = torch.device("cpu")

    for i, data in tqdm(enumerate(dataloader.load_data()), total=len(dataloader)):
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx:
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                              opt.export_onnx, verbose=True)
            exit(0)
        minibatch = 1 
        if opt.engine:
            generated, detections, losses = run_trt_engine(opt.engine, minibatch, [data['label'], data['feat'], data['inst']])
        elif opt.onnx:
            generated, detections, losses = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['feat'], data['inst']])
        else:
            generated, detections, losses = model.inference(data['label'], data['inst'], data['image'], data['feat'], data['target'])
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in detections]

        mask = np.array(Image.fromarray(get_mask(outputs[0], mask_thresh=opt.mask_thresh)).resize((256,256)), dtype=np.uint8)
        assert mask.shape == (256,256)

        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('detection', mask*255)])
        img_path = data['path']
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()


def get_mask(outputs, mask_thresh, conf_thresh=0):
    # N * 1 * 256 * 256 --> 256 * 256
    if conf_thresh==0:
        masks = outputs['masks'] > mask_thresh
    else:
        index = outputs['scores'] >= conf_thresh
        masks = outputs['masks'][index,:,:,:] > mask_thresh
    mask = np.array(torch.squeeze(torch.sum(masks.int(), dim=0)), dtype=np.uint8)
    return mask

evaluate(model, data_loader, opt, visualizer=visualizer, webpage=webpage)