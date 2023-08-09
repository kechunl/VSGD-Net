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
import pdb
import pickle
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import torch.distributed as dist
import csv
import sys
sys.path.append('..')
from util.fast_nuclei_Evaluation import compute_nuclei_metric, compute_nuclei_metric_average

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

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
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%f' % (opt.phase, opt.which_epoch, opt.mask_thresh))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
last_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)

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

epoch = int(opt.which_epoch) if opt.which_epoch != 'latest' else last_epoch


@torch.no_grad()
def evaluate(model, dataloader, opt, visualizer, webpage):
    model.eval()
    cpu_device = torch.device("cpu")

    if opt.average_metric:
        results = pd.DataFrame(columns=["Image", "TP", "FP", "FN", "Dice", "IoU", "C", "U", "count", "IOU_thresh"])
    else:
        results = pd.DataFrame(columns=["Image", "pixel_TP", "pixel_TN", "pixel_FP", "pixel_FN", "TP", "FP", "FN", "Dice", "IoU", "C", "U", "count", "conf_thresh"])
    auc = []

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

        if not opt.has_real_image:
            mask = np.array(Image.fromarray(get_mask(outputs[0], mask_thresh=opt.mask_thresh)).resize((256,256)), dtype=np.uint8)
            assert mask.shape == (256,256)
            # Image.fromarray(np.array(mask*255, dtype=np.uint8)).save(os.path.join(web_dir, images, os.path.basename(data['path'][0])))
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('detection', mask*255)])
            img_path = data['path']
            visualizer.save_images(webpage, visuals, img_path)
            continue

        # AUC score
        # fpr, tpr, thresholds = metrics.roc_curve(np.squeeze(np.sum(data['target'][0]['masks'].numpy(), axis=0)).flatten(), np.squeeze(np.sum(outputs[0]['masks'].numpy(), axis=0)).flatten())
        # auc.append(metrics.auc(fpr, tpr))

        # Evaluate using nuclei metric
        gt_mask = np.array(Image.fromarray(get_mask(data['target'][0], mask_thresh=0.5)).resize((256,256)), dtype=np.uint8)
        assert gt_mask.shape == (256,256)
        # for conf_thresh in np.arange(0.0, 1, 0.05):
        for conf_thresh in [0.1]:
            mask = np.array(Image.fromarray(get_mask(outputs[0], mask_thresh=opt.mask_thresh, conf_thresh=conf_thresh)).resize((256,256)), dtype=np.uint8)

            # # exclude patches without melanocytes
            # if np.max(gt_mask) == 0:
            #     continue
            assert mask.shape == (256,256)
            if opt.average_metric:
                results = compute_nuclei_metric_average(gt_mask, mask, results, data['target'][0]["image_id"])
            else:
                results = compute_nuclei_metric(gt_mask, mask, results, data['target'][0]["image_id"], conf_thresh, opt.nuclei_thresh)

            if i < opt.how_many and conf_thresh==opt.mask_thresh:
                # visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0])),
                                       ('detection', mask*255),
                                       ('gt', gt_mask*255)])
                img_path = data['path']
                visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
    
    if not opt.has_real_image:
        return

    auc = [x for x in auc if np.isnan(x)==False]
    print('AUC Score: %f\t Std: %f\n' % (np.mean(auc), np.std(auc)))

    # plot PR-curve
    AP = []
    AR = []
    for threshold in np.arange(0, 1, 0.05):
        performance = results.loc[results['conf_thresh'] == threshold].sum(axis=0)
        AP.append(performance["TP"] / (performance["TP"] + performance["FP"] + 1e-10))
        AR.append(performance["TP"] / (performance["TP"] + performance["FN"] + 1e-10))

    result_dict = {"AP": AP, "AR": AR}
    with open(os.path.join(web_dir, 'pr.pickle'), 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure()
    plt.plot(AR, AP)
    plt.savefig(os.path.join(web_dir, 'pr.png'))
    plt.close()
    pdb.set_trace()

    if opt.average_metric:
        pdb.set_trace()
        result_summary_average(results)
    else:
        result_summary(results, opt.nuclei_thresh)
    return results


def get_mask(outputs, mask_thresh, conf_thresh=0):
    # N * 1 * 256 * 256 --> 256 * 256
    if conf_thresh==0:
        masks = outputs['masks'] > mask_thresh
    else:
        index = outputs['scores'] >= conf_thresh
        masks = outputs['masks'][index,:,:,:] > mask_thresh
    mask = np.array(torch.squeeze(torch.sum(masks.int(), dim=0)), dtype=np.uint8)
    return mask

def result_summary_average(results):
    results.to_csv(os.path.join(web_dir, 'test_results_details.csv'), index=False)
    # AP50, AP75, mAP, AR50, AR75, mAR
    # threshold=0.5
    performance = results.loc[results['IOU_thresh'] == 0.5].sum(axis=0)
    AP50 = performance["TP"] / (performance["TP"] + performance["FP"] + 1e-10)
    AR50 = performance["TP"] / (performance["TP"] + performance["FN"] + 1e-10)
    # threshold=0.75
    performance = results.loc[results['IOU_thresh'] == 0.75].sum(axis=0)
    AP75 = performance["TP"] / (performance["TP"] + performance["FP"] + 1e-10)
    AR75 = performance["TP"] / (performance["TP"] + performance["FN"] + 1e-10)
    # mAP, mAR
    AP = []
    AR = []
    for threshold in np.arange(0.5, 0.95, 0.05):
        performance = results.loc[results['IOU_thresh'] == threshold].sum(axis=0)
        AP.append(performance["TP"] / (performance["TP"] + performance["FP"] + 1e-10))
        AR.append(performance["TP"] / (performance["TP"] + performance["FN"] + 1e-10))
    average_precision = np.mean(AP)
    average_recall = np.mean(AR)

    test_result = {'AP50': AP50, 'AR50': AR50, 'AP75': AP75, 'AR75': AR75, 'mAP': average_precision, 'mAR': average_recall}
    print("****** Average Nuclei Result ******\n", test_result)

    with open(os.path.join(web_dir, 'testResult_average.csv'), 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, test_result.keys())
        w.writeheader()
        w.writerow(test_result)


def result_summary(results, nuclei_thresh):
    average_performance = results.sum(axis=0)
    # Calculate pixel-level metrics
    tp, tn, fp, fn = average_performance["pixel_TP"], average_performance["pixel_TN"], average_performance["pixel_FP"], average_performance["pixel_FN"]
    pixel_precision = tp / (tp + fp + 1e-10)
    pixel_recall = tp / (tp + fn + 1e-10)
    pixel_F1 = 2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-10)

    pixel_acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    pixel_performance = (pixel_recall + tn/(tn+fp+1e-10)) / 2
    pixel_iou = tp / (tp+fp+fn+1e-10)
    # Calculate object-level metrics
    TP, FP, FN, dice, iou, C, U, count = average_performance["TP"], average_performance["FP"], average_performance["FN"], average_performance["Dice"], average_performance["IoU"], average_performance["C"], average_performance["U"], average_performance["count"]

    object_recall = TP / (TP + FN + 1e-10)
    object_precision = TP / (TP + FP + 1e-10)
    object_F1 = 2 * TP / (2 * TP + FP + FN + 1e-10)

    object_dice = dice / count if count!=0 else count
    object_iou = iou / count if count!=0 else count
    # object_haus = haus / count
    object_AJI = float(C) / U if U!=0 else U

    # test_result = [pixel_precision, pixel_recall, pixel_F1, pixel_acc, pixel_performance, pixel_iou, object_recall, object_precision, object_F1, object_dice, object_iou, object_AJI]
    test_result = {'pixel_precision': pixel_precision, "pixel_recall": pixel_recall, "pixel_F1": pixel_F1, "pixel_acc": pixel_acc, "pixel_performance": pixel_performance, "pixel_iou": pixel_iou, "object_recall": object_recall, "object_precision": object_precision, "object_F1": object_F1, "object_dice": object_dice, "object_iou": object_iou, "object_AJI": object_AJI}
    print("****** Nuclei Result ******\n", test_result, '\n')

    # save to csv
    with open(os.path.join(web_dir, 'testResult_{}.csv'.format(nuclei_thresh)), 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, test_result.keys())
        w.writeheader()
        w.writerow(test_result)

evaluate(model, data_loader, opt, visualizer=visualizer, webpage=webpage)

dist.destroy_process_group()