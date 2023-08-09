import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff as hausdorff
import pdb
import skimage.morphology


def intersection_over_union(ground_truth, prediction):
    
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    # import pdb;pdb.set_trace()
    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union
    
    return IOU
    


def measures_at(threshold, IOU):
    
    matches = IOU > threshold
    
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    f1 = 2*TP / (2*TP + FP + FN + 1e-9)
    
    return f1, TP, FP, FN

def compute_nuclei_metric(ground_truth, prediction, results, image_name, conf_thresh, nuclei_thresh=3):
    ## Pixel Level
    gt = np.where(ground_truth>0, 1, 0)
    pred = np.where(prediction>0, 1, 0)
    pixel_tp = np.sum(pred * gt)    # true postives
    pixel_tn = np.sum((1-pred) * (1-gt))  # true negatives
    pixel_fp = np.sum(pred * (1-gt))  # false postives
    pixel_fn = np.sum((1-pred) * gt)  # false negatives

    ## Object Level 
    # Count objects
    gt_labeled = skimage.morphology.label(ground_truth>0)
    pred_labeled = skimage.morphology.label(prediction>0)
    true_objects = len(np.unique(gt_labeled))
    pred_objects = len(np.unique(pred_labeled))
    # import pdb;pdb.set_trace()
    # Compute intersection
    h = np.histogram2d(gt_labeled.flatten(), pred_labeled.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(gt_labeled, bins=true_objects)[0]
    area_pred = np.histogram(pred_labeled, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1) # 1 x (N(true_objects)+1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude tiny instances
    intersection[:, area_pred[0,:]<nuclei_thresh] = 0
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union # N(true_objects) x N(pred_objects)

    # Threshold
    # portion_gt = intersection / area_true[1:, :]
    # intersection = (portion_gt > 0.3) * intersection
    # IOU = (portion_gt > 0.3) * IOU

    # Object level metrics
    TP = 0
    FP = len(np.unique(pred_labeled)) - 1
    FN = 0
    dice, iou, C, U, count = 0, 0, 0, 0, 0
    # haus = 0

    if len(np.unique(gt_labeled)) > 1:
        if np.sum(np.where(np.sum(intersection, axis=1) > 0, 1, 0)) > 0:
            # unique matching version:
            # while intersection.any() != 0:
            #       find unique matches
            #       set the rows and columns of the matches in intersection to 0
            intersect = intersection.copy() # N(true_objects) x N(pred_objects)
            while intersect.any() != 0:
                gt_index = np.array(np.where(np.sum(intersect, axis=1) > 0))
                gt_index = gt_index.reshape([gt_index.size])
                pred_index = np.argmax(intersect[gt_index, :], axis=1)
                gt_matches = list(zip(gt_index, pred_index))

                pred_index = np.array(np.where(np.sum(intersect, axis=0) > 0))
                pred_index = pred_index.reshape([pred_index.size])
                gt_index = np.argmax(intersect[:, pred_index], axis=0)
                pred_matches = list(zip(gt_index, pred_index))

                matches = list(set(gt_matches) & set(pred_matches))
                TP += len(matches)
                index = list(zip(*matches))
                gt_index = np.array(index[0])
                pred_index = np.array(index[1])

                overlap_area = intersect[gt_index, pred_index]
                # pdb.set_trace()
                dice += np.sum(2 * overlap_area / (area_true[gt_index+1, 0] + area_pred[0, pred_index+1]))
                iou += np.sum(overlap_area / (area_true[gt_index+1, 0] + area_pred[0, pred_index+1] - overlap_area))
                # haus += np.sum([max(hausdorff(np.argwhere(ground_truth==gt_index[i]+1), np.argwhere(prediction==pred_index[i]+1))[0], hausdorff(np.argwhere(prediction==pred_index[i]+1), np.argwhere(ground_truth==gt_index[i]+1))[0]) for i in range(len(gt_index))])
                C += np.sum(overlap_area)
                count += TP

                intersect[gt_index, :] = 0
                intersect[:, pred_index] = 0


    FN = len(np.unique(gt_labeled)) - 1 - TP
    FP = len(np.unique(pred_labeled)) - 1 - TP
    # C = np.sum(np.max(intersection, axis=1))
    U = np.sum(gt) + np.sum(pred) - C

    # y_true += [1.] * FN
    # y_scores += [0.] * FP

    # Log result
    res = {"Image": image_name, "pixel_TP": pixel_tp, "pixel_TN": pixel_tn, "pixel_FP": pixel_fp, "pixel_FN": pixel_fn, "TP": TP, "FP": FP, "FN": FN, "Dice": dice, "IoU": iou, "C": C, "U": U, "count": count, "conf_thresh": conf_thresh}
    # res = {"Image": image_name, "pixel_TP": pixel_tp, "pixel_TN": pixel_tn, "pixel_FP": pixel_fp, "pixel_FN": pixel_fn, "TP": TP, "FP": FP, "FN": FN, "Dice": dice, "IoU": iou, "Haus": haus, "C": C, "U": U, "count": count}
    row = len(results)
    results.loc[row] = res

    return results

# Compute Average Metrics for all IoU thresholds
def compute_nuclei_metric_average(ground_truth, prediction, results, image_name):
    # ## Pixel Level
    gt = np.where(ground_truth>0, 1, 0)
    pred = np.where(prediction>0, 1, 0)
    # pixel_tp = np.sum(pred * gt)    # true postives
    # pixel_tn = np.sum((1-pred) * (1-gt))  # true negatives
    # pixel_fp = np.sum(pred * (1-gt))  # false postives
    # pixel_fn = np.sum((1-pred) * gt)  # false negatives

    ## Object Level 
    # Count objects
    gt_labeled = skimage.morphology.label(ground_truth>0)
    pred_labeled = skimage.morphology.label(prediction>0)
    true_objects = len(np.unique(gt_labeled))
    pred_objects = len(np.unique(pred_labeled))
    # import pdb;pdb.set_trace()
    # Compute intersection
    h = np.histogram2d(gt_labeled.flatten(), pred_labeled.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(gt_labeled, bins=true_objects)[0]
    area_pred = np.histogram(pred_labeled, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1) # 1 x (N(true_objects)+1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union # N(true_objects) x N(pred_objects)

    # Threshold
    for threshold in np.arange(0.5, 0.95, 0.05):
        intersection_thresh = (IOU>threshold) * intersection
        IOU_thresh = (IOU>threshold) * IOU

        # Object level metrics
        TP = 0
        FP = len(np.unique(pred_labeled)) - 1
        FN = 0
        dice, iou, C, U, count = 0, 0, 0, 0, 0
        # haus = 0

        if len(np.unique(gt_labeled)) > 1:
            if np.sum(np.where(np.sum(intersection_thresh, axis=1) > 0, 1, 0)) > 0:
                intersect = intersection_thresh.copy() # N(true_objects) x N(pred_objects)
                while intersect.any() != 0:
                    gt_index = np.array(np.where(np.sum(intersect, axis=1) > 0))
                    gt_index = gt_index.reshape([gt_index.size])
                    pred_index = np.argmax(intersect[gt_index, :], axis=1)
                    gt_matches = list(zip(gt_index, pred_index))

                    pred_index = np.array(np.where(np.sum(intersect, axis=0) > 0))
                    pred_index = pred_index.reshape([pred_index.size])
                    gt_index = np.argmax(intersect[:, pred_index], axis=0)
                    pred_matches = list(zip(gt_index, pred_index))

                    matches = list(set(gt_matches) & set(pred_matches))
                    TP += len(matches)
                    index = list(zip(*matches))
                    gt_index = np.array(index[0])
                    pred_index = np.array(index[1])

                    overlap_area = intersect[gt_index, pred_index]
                    # pdb.set_trace()
                    dice += np.sum(2 * overlap_area / (area_true[gt_index+1, 0] + area_pred[0, pred_index+1]))
                    iou += np.sum(overlap_area / (area_true[gt_index+1, 0] + area_pred[0, pred_index+1] - overlap_area))
                    # haus += np.sum([max(hausdorff(np.argwhere(ground_truth==gt_index[i]+1), np.argwhere(prediction==pred_index[i]+1))[0], hausdorff(np.argwhere(prediction==pred_index[i]+1), np.argwhere(ground_truth==gt_index[i]+1))[0]) for i in range(len(gt_index))])
                    C += np.sum(overlap_area)
                    count += TP

                    intersect[gt_index, :] = 0
                    intersect[:, pred_index] = 0
                # count = TP
        FN = len(np.unique(gt_labeled)) - 1 - TP
        FP = len(np.unique(pred_labeled)) - 1 - TP
        # C = np.sum(np.max(intersection, axis=1))
        U = np.sum(gt) + np.sum(pred) - C

        # Log result
        res = {"Image": image_name, "TP": TP, "FP": FP, "FN": FN, "Dice": dice, "IoU": iou, "C": C, "U": U, "count": count, "IOU_thresh": threshold}
        # res = {"Image": image_name, "pixel_TP": pixel_tp, "pixel_TN": pixel_tn, "pixel_FP": pixel_fp, "pixel_FN": pixel_fn, "TP": TP, "FP": FP, "FN": FN, "Dice": dice, "IoU": iou, "Haus": haus, "C": C, "U": U, "count": count}
        row = len(results)
        results.loc[row] = res

    return results


# Compute Average Precision for all IoU thresholds

def compute_af1_results(ground_truth, prediction, results, image_name):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    if IOU.shape[0] > 0:
        jaccard = np.max(IOU, axis=0).mean()
    else:
        jaccard = 0.0
    
    # Calculate F1 score at all thresholds
    for t in np.arange(0.5, 0.95, 0.05):
        f1, tp, fp, fn = measures_at(t, IOU)
        res = {"Image": image_name, "Threshold": t, "F1": f1, "Jaccard": jaccard, "TP": tp, "FP": fp, "FN": fn}
        row = len(results)
        results.loc[row] = res
        
    return results

# Count number of False Negatives at 0.7 IoU

def get_false_negatives(ground_truth, prediction, results, image_name, threshold=0.7):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    
    true_objects = len(np.unique(ground_truth))
    if true_objects <= 1:
        return results
        
    area_true = np.histogram(ground_truth, bins=true_objects)[0][1:]
    true_objects -= 1
    
    # Identify False Negatives
    matches = IOU > threshold
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    data = np.asarray([ 
        area_true.copy(), 
        np.array(false_negatives, dtype=np.int32)
    ])

    results = pd.concat([results, pd.DataFrame(data=data.T, columns=["Area", "False_Negative"])], sort=False)
        
    return results

# Count the number of splits and merges

def get_splits_and_merges(ground_truth, prediction, results, image_name):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    
    matches = IOU > 0.1
    merges = np.sum(matches, axis=0) > 1
    splits = np.sum(matches, axis=1) > 1
    r = {"Image_Name":image_name, "Merges":np.sum(merges), "Splits":np.sum(splits)}
    results.loc[len(results)+1] = r
    return results

