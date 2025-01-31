import pandas as pd
import numpy as np
import numba
import re
from glob import glob
import cv2
import ast
import matplotlib.pyplot as plt

from numba import jit
from typing import List, Union, Tuple
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import numba
import re
import cv2
import ast
import matplotlib.pyplot as plt

from numba import jit
from typing import List, Union, Tuple


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision

def show_result(sample_id, preds, gt_boxes):
    sample = cv2.imread(f'{TRAIN_ROOT_PATH}/{sample_id}.jpg', cv2.IMREAD_COLOR)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for pred_box in preds:
        cv2.rectangle(
            sample,
            (pred_box[0], pred_box[1]),
            (pred_box[2], pred_box[3]),
            (220, 0, 0), 2
        )

    for gt_box in gt_boxes:    
        cv2.rectangle(
            sample,
            (gt_box[0], gt_box[1]),
            (gt_box[2], gt_box[3]),
            (0, 0, 220), 2
        )

    ax.set_axis_off()
    ax.imshow(sample)
    ax.set_title("RED: Predicted | BLUE - Ground-truth")
    
# Numba typed list!
iou_thresholds = numba.typed.List()

for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    iou_thresholds.append(x)

def calculate_final_score(all_predictions, score_threshold):
    final_scores = []
    for i in range(len(all_predictions)):
        gt_boxes = all_predictions[i]['gt_boxes'].copy()
        pred_boxes = all_predictions[i]['pred_boxes'].copy()
        scores = all_predictions[i]['scores'].copy()
        image_id = all_predictions[i]['image_id']
        labels = all_predictions[i]['labels']

        indexes = np.where(scores>score_threshold)
        #print(indexes)
        #print(labels)
        pred_boxes = pred_boxes[indexes]
        scores = scores[indexes]
        #print(len(scores))
        
        if labels[0] == 2: # no ground truth boxes
            if len(scores) == 0:
                image_precision = 1.
            else:
                image_precision = 0.
        else:
            image_precision = calculate_image_precision(gt_boxes, pred_boxes,thresholds=iou_thresholds,form='pascal_voc')
        
        final_scores.append(image_precision)

    return np.mean(final_scores)

#############################################################
import torch

def predict_eval_set(model, validation_loader, imsize=512):
    all_predictions = []

    for images, targets, image_ids in tqdm(validation_loader, total=len(validation_loader)):
        with torch.no_grad():
            images = torch.stack(images)
            images = images.cuda().float()
            det = model(images, torch.tensor([1]*images.shape[0]).float().cuda())

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                
                labels = targets[i]['labels'].cpu().numpy()
                gt_boxes = (targets[i]['boxes'].cpu().numpy()*(1024//imsize)).clip(min=0, max=1023).astype(int)
                if hasattr(validation_loader.dataset, 'yxyx') and validation_loader.dataset.yxyx is True:
                    gt_boxes[:, [0,1,2,3]] = gt_boxes[:, [1,0,3,2]]
                
                all_predictions.append({
                    'pred_boxes': (boxes*(1024//imsize)).clip(min=0, max=1023).astype(int),
                    'scores': scores,
                    'gt_boxes': gt_boxes,
                    'image_id': image_ids[i],
                    'labels': labels,
                    'source': targets[i]['source']
                })
    return all_predictions

def find_best_metrics(all_predictions):
    metrics = {}
    score_thresholds = [0.3, 0.35, 0.37, 0.4, 0.42, 0.45, 0.47, 0.5, 0.55, 0.6]
    best_final_score, best_score_threshold = 0, 0
    #for score_threshold in tqdm(np.arange(0, 1, 0.01), total=np.arange(0, 1, 0.01).shape[0]):
    for score_threshold in np.arange(0, 1, 0.01):
        final_score = calculate_final_score(all_predictions, score_threshold)
        if score_threshold in score_thresholds: 
            metrics[score_threshold] = round(final_score, 5)
        if final_score > best_final_score:
            best_final_score = final_score
            best_score_threshold = score_threshold

        metrics['best_score'] = round(best_final_score, 5)
        metrics['best_threshold'] = round(best_score_threshold, 5)
    return metrics

#def eval_metrics(model, validation_loader, imsize=512):
#    all_predictions = predict_eval_set(model, validation_loader, imsize)
    
#    return find_best_metrics(all_predictions)

def eval_metrics(model, validation_loader, imsize=512, by_source=False):
    all_predictions = predict_eval_set(model, validation_loader, imsize)
    all_metrics = find_best_metrics(all_predictions)
    
    if not by_source:
        return all_metrics
    
    metrics = {}
    metrics.update(all_metrics)
    def get_preds_by_source(preds, src):
        results = []
        for p in preds:
            if p['source'] == src:
                results.append(p)
        return results
    
    for source in ['arvalis_1', 'arvalis_2', 'arvalis_3', 'ethz_1', 'rres_1', 'usask_1', 'inrae_1', 'unknown']:
        preds = get_preds_by_source(all_predictions, source)
        source_metrics = find_best_metrics(preds)
        metrics[source] = {
            'score': round(calculate_final_score(preds, all_metrics['best_threshold']), 5), #find_best_metrics(preds)
            'num': len(preds),
            'best_score': source_metrics['best_score'],
            'best_threshold': source_metrics['best_threshold']
        }

    return metrics