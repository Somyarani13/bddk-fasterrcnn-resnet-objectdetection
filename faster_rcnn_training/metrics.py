import os
import json
import numpy as np
import mmcv
from collections import defaultdict
from mmdet.apis import init_detector, inference_detector

# Load Faster R-CNN model
config_file = './inference_gt_app/configs/fast_rcnn_bdd100/faster_rcnn_r50_fpn_1x_det_bdd100k.py'
checkpoint_file = './inference_gt_app/configs/fast_rcnn_bdd100/faster_rcnn_r50_fpn_1x_det_bdd100k.pth'
device = 'cpu'  # Use 'cuda:0' if you have GPU, otherwise use 'cpu'

model = init_detector(config_file, checkpoint_file, device=device)

# Category to class index mapping, based on the categories in the dataset
category_to_class_index = {
    "bus": 5,
    "traffic light": 9,
    "traffic sign": 9,
    "car": 2,
    "person": 0,
    "bike": 3,
    "motor": 3,
    "rider": 3,
    "truck": 7,
    "train": 6,
}

# Helper function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

# Matching predictions to ground truth using IoU threshold
def match_predictions_to_ground_truth(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    for i, pred_box in enumerate(pred_boxes):
        pred_label = pred_labels[i]
        max_iou = 0
        best_match = -1

        for j, gt_box in enumerate(gt_boxes):
            if gt_labels[j] == pred_label:
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match = j

        if max_iou >= iou_threshold and best_match != -1 and not gt_matched[best_match]:
            tp[pred_label] += 1
            gt_matched[best_match] = True
        else:
            fp[pred_label] += 1

    for i, matched in enumerate(gt_matched):
        if not matched:
            fn[gt_labels[i]] += 1

    return tp, fp, fn

def calculate_overall_metrics(tp, fp, fn):
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    # Calculate overall precision, recall, and F1
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    return overall_precision, overall_recall, overall_f1

# Function to calculate precision, recall, and F1 scores
def calculate_metrics(tp, fp, fn):
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}

    for class_id in range(10):  # Assuming 10 classes
        true_pos = tp[class_id]
        false_pos = fp[class_id]
        false_neg = fn[class_id]

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_dict[class_id] = precision
        recall_dict[class_id] = recall
        f1_dict[class_id] = f1

    return precision_dict, recall_dict, f1_dict

# Parsing ground truth annotations from the JSON file
def parse_annotations(json_data, img_name):
    gt_boxes = []
    gt_labels = []

    for annotation in json_data:
        if annotation['name'] == img_name:
            for label in annotation['labels']:
                if 'box2d' in label:

                    box = label['box2d']
                    category = label['category']
                    if category in category_to_class_index:
                        gt_boxes.append([box['x1'], box['y1'], box['x2'], box['y2']])
                        gt_labels.append(category_to_class_index[category])  # passing category to above dict
                    else:
                        print(f"Warning: Category {category} not in mapping")

    return gt_boxes, gt_labels

# Load validation dataset
def evaluate_on_dataset(val_images_path, val_annotations_path, prob_threshold=0.3, iou_threshold=0.5):
    all_tp, all_fp, all_fn = defaultdict(int), defaultdict(int), defaultdict(int)

    # Load the entire annotations JSON once
    with open(val_annotations_path, 'r') as f:
        val_annotations = json.load(f)

    # Loop through the validation images and annotations
    for img_name in os.listdir(val_images_path)[:10]: #Evaluate first 10 images
        img_path = os.path.join(val_images_path, img_name)

        img = mmcv.imread(img_path)
        result = inference_detector(model, img)

        # Extract predicted boxes, labels, and scores
        pred_boxes, pred_scores, pred_labels = [], [], []
        for class_idx, class_results in enumerate(result):
            if len(class_results) == 0:
                continue  # Skip empty predictions
            for bbox in class_results:
                if bbox[4] >= prob_threshold:  # Use the score threshold to filter predictions
                    pred_boxes.append(bbox[:4])
                    pred_labels.append(class_idx)  # Class index
                    pred_scores.append(bbox[4])

        gt_boxes, gt_labels = parse_annotations(val_annotations, img_name)

        # print(f"Predicted labels for {img_name}: {pred_labels}")
        # print(f"Ground Truth Label for {img_name}: {gt_labels}\n")

        # Match predictions to ground truth
        tp, fp, fn = match_predictions_to_ground_truth(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold)

        # Accumulate metrics
        for class_id in tp.keys():
            all_tp[class_id] += tp[class_id]
            all_fp[class_id] += fp[class_id]
            all_fn[class_id] += fn[class_id]

    # Class Level results
    precision_dict, recall_dict, f1_dict = calculate_metrics(all_tp, all_fp, all_fn)

    overall_precision, overall_recall, overall_f1 = calculate_overall_metrics(all_tp, all_fp, all_fn)
    return overall_precision, overall_recall, overall_f1


# Set validation dataset paths
val_images_path = './assignment_data_bdd/bdd100k_images_100k/val'
val_annotations_path = './assignment_data_bdd/bdd100k_labels_release/bdd100k_labels_images_val.json'

# Run evaluation
precision, recall, f1 = evaluate_on_dataset(val_images_path, val_annotations_path)

# Output the results
print("IOU: 0.5, Thres: 0.4")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall: {recall:.4f}")
print(f"Overall F1-Score: {f1:.4f}")
