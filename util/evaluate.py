from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from .detection_utils import MetricLogger, is_dist_avail_and_initialized

import sys
sys.path.append('..')
from util.fast_nuclei_Evaluation import compute_nuclei_metric
from .visualize_detection import display_instances

import torch
from tqdm import tqdm
import pdb
import time
import os
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch.distributed as dist
import sys
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

import util.util as util

@torch.no_grad()
def evaluate(model, dataloader, opt, coco_only=True, visualizer=None, epoch=None, total_steps=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    model.eval()
    # coco = get_coco_api_from_dataset(dataloader.dataset)
    # iou_types = ["bbox", "segm"]
    # coco_evaluator = CocoEvaluator(coco, iou_types)
    # metric_logger = MetricLogger(delimiter="  ")
    # header = 'Test:'
    cpu_device = torch.device("cpu")

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'eval.txt')
    savedStdout = sys.stdout
    sys.stdout = open(log_name, 'a')

    if epoch != None:
        print('\nEvaluation Epoch %d' % epoch, '\n')

    results = pd.DataFrame(columns=["Image", "pixel_TP", "pixel_TN", "pixel_FP", "pixel_FN", "TP", "FP", "FN", "Dice", "IoU", "C", "U", "count"])
    auc = []
    vis_index = np.random.randint(len(dataloader))
    print('Going to visualize image %d/%d' % (vis_index, len(dataloader)))

    for i, data in tqdm(enumerate(dataloader.load_data()), total=opt.how_many):
        if i >= opt.how_many:
            break
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
        model_time = time.time()
        if opt.engine:
            generated, detections, losses = run_trt_engine(opt.engine, minibatch, [data['label'], data['feat'], data['inst']])
        elif opt.onnx:
            generated, detections, losses = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['feat'], data['inst']])
        else:
            generated, detections, losses = model.module.inference(data['label'], data['inst'], data['image'], data['feat'], data['target'])
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in detections]
        model_time = time.time() - model_time

        if i == vis_index:
            image_drawn1 = display_instances(data['label'], outputs[0]['boxes'], scores=outputs[0]['scores'], masks=outputs[0]['masks'], mask_thresh=0.5)
            image_drawn1 = image_drawn1.permute(1,2,0).numpy()
            image_drawn2 = display_instances(data['label'], outputs[0]['boxes'], scores=outputs[0]['scores'], masks=outputs[0]['masks'], mask_thresh=0.1)
            image_drawn2 = image_drawn2.permute(1,2,0).numpy()
            gt_drawn = display_instances(data['label'], data['target'][0]['boxes'], scores=None, masks=data['target'][0]['masks'], mask_thresh=0.5)
            gt_drawn = gt_drawn.permute(1,2,0).numpy()
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0])),
                                   ('detection_0.5', image_drawn1),
                                   ('detection_0.1', image_drawn2),
                                   ('gt', gt_drawn)])
            if visualizer is not None:
                print('[EVAL] visualize epoch %d' % epoch)
                visualizer.display_current_results(visuals, epoch, total_steps)

        # AUC score
        fpr, tpr, thresholds = metrics.roc_curve(np.squeeze((np.sum(data['target'][0]['masks'].numpy(), axis=0)>0).astype(int)).flatten(), np.squeeze(np.sum(outputs[0]['masks'].numpy(), axis=0)).flatten())
        auc.append(metrics.auc(fpr, tpr))

        # Evaluate using nuclei metric
        if not coco_only:
            mask = np.array(Image.fromarray(get_mask(outputs[0], mask_thresh=0.1)).resize((256,256)), dtype=np.uint8)
            gt_mask = np.array(Image.fromarray(get_mask(data['target'][0], mask_thresh=0.5)).resize((256,256)), dtype=np.uint8)
            results = compute_nuclei_metric(gt_mask, mask, results, data['target'][0]["image_id"])

        # res = {target["image_id"].item(): output for target, output in zip(data['target'], outputs)}
        # evaluator_time = time.time()
        # coco_evaluator.update(res)
        # evaluator_time = time.time() - evaluator_time
        # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    torch.cuda.synchronize()
    auc = [x for x in auc if np.isnan(x)==False]
    print('AUC Score: %f\t Std: %f\n' % (np.mean(auc), np.std(auc)))
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # coco_evaluator.synchronize_between_processes()

    if not coco_only:
        result_summary(results)

    # accumulate predictions from all images
    # coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    sys.stdout = savedStdout
    return results


def get_mask(outputs, mask_thresh=0.1):
    mask = np.array(torch.squeeze(torch.sum((outputs['masks'] > mask_thresh).int(), dim=0)), dtype=np.uint8)
    return mask


def result_summary(results):
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

    test_result = [pixel_precision, pixel_recall, pixel_F1, pixel_acc, pixel_performance, pixel_iou, object_recall, object_precision, object_F1, object_dice, object_iou, object_AJI]
    print("****** Nuclei Result ******\n", test_result, '\n')
