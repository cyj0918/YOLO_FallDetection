#!/usr/bin/env python3
"""
exp_statistic.py

Evaluate YOLO TXT predictions vs ground truth (TXT only),
compute detection and localization metrics, print and save as JSON.
Now includes per-image detailed statistics and per-detection data.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO TXT predictions vs ground truth (TXT only)"
    )
    parser.add_argument("--gt-dir", required=True,
                        help="Folder with ground-truth YOLO TXT files")
    parser.add_argument("--pred-dir", required=True,
                        help="Folder with prediction YOLO TXT files")
    parser.add_argument("--iou-thres", type=float, default=0.7,
                        help="IoU threshold for matching TP/FP")
    parser.add_argument("--output", "-o", type=str, default="statistics.json",
                        help="Output JSON file path")
    return parser.parse_args()

def load_txt(path):
    """Load YOLO format TXT file, return classes, boxes and confidences"""
    cls, boxes, confs = [], [], []
    if Path(path).exists():
        for line in open(path):
            parts = line.strip().split()
            cls.append(int(parts[0]))
            boxes.append(list(map(float, parts[1:5])))
            # Handle presence/absence of confidence score
            confs.append(float(parts[5]) if len(parts) >= 6 else 1.0)
    return cls, boxes, confs

def xywh_to_xyxy_norm(box):
    """Convert normalized XYWH to normalized XYXY format"""
    x, y, w, h = box
    return [x - w/2, y - h/2, x + w/2, y + h/2]

def compute_iou(a, b):
    """Calculate Intersection over Union for two boxes in XYXY format"""
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    
    # Calculate intersection coordinates
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    
    # Calculate areas
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    
    # Avoid division by zero
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def evaluate_detection(gt_dir, pred_dir, iou_thres):
    """Evaluate detections, returning per-image statistics and overall metrics"""
    gt_files = sorted(Path(gt_dir).glob("*.txt"))
    pred_files = sorted(Path(pred_dir).glob("*.txt"))
    
    # Per-image statistics storage
    per_image_stats = []
    overall_tp = overall_fp = overall_fn = 0
    
    for gf, pf in zip(gt_files, pred_files):
        image_name = gf.stem
        gcls, gboxes, _ = load_txt(gf)
        pcls, pboxes, pconfs = load_txt(pf)
        
        # Convert all boxes to XYXY for IoU calculation
        gboxes_xyxy = [xywh_to_xyxy_norm(b) for b in gboxes]
        pboxes_xyxy = [xywh_to_xyxy_norm(b) for b in pboxes]
        
        # Per-image metrics
        img_tp = img_fp = img_fn = 0
        matched_gt_indices = set()
        detections = []
        
        # Process each prediction
        for i, (pc, pb, conf) in enumerate(zip(pcls, pboxes, pconfs)):
            pxy = pboxes_xyxy[i]
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching GT
            for j, (gc, gxy) in enumerate(zip(gcls, gboxes_xyxy)):
                if gc == pc:
                    iou = compute_iou(pxy, gxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            # Determine detection status
            status = "FP"
            if best_iou >= iou_thres and best_gt_idx != -1:
                if best_gt_idx not in matched_gt_indices:
                    status = "TP"
                    matched_gt_indices.add(best_gt_idx)
                    img_tp += 1
                else:
                    img_fp += 1
            else:
                img_fp += 1
            
            # Record detection details
            detections.append({
                "class": pc,
                "confidence": conf,
                "bbox": pb,
                "status": status,
                "iou": best_iou
            })
        
        # Identify false negatives (unmatched ground truths)
        for j in range(len(gboxes)):
            if j not in matched_gt_indices:
                img_fn += 1
                detections.append({
                    "class": gcls[j],
                    "confidence": 1.0,  # Ground truth confidence is always 1
                    "bbox": gboxes[j],
                    "status": "FN",
                    "iou": 0.0
                })
        
        # Update overall counts
        overall_tp += img_tp
        overall_fp += img_fp
        overall_fn += img_fn
        
        # Store per-image statistics
        per_image_stats.append({
            "image": image_name,
            "TP": img_tp,
            "FP": img_fp,
            "FN": img_fn,
            "detections": detections
        })
    
    # Calculate overall metrics
    prec = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    rec = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return per_image_stats, {
        "TP": overall_tp,
        "FP": overall_fp,
        "FN": overall_fn,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1
    }

def evaluate_localization(gt_dir, pred_dir, iou_thres):
    """Evaluate localization metrics with per-class statistics"""
    gt_files = sorted(Path(gt_dir).glob("*.txt"))
    pred_files = sorted(Path(pred_dir).glob("*.txt"))
    
    class_stats = {}
    per_image_localization = []
    
    for gf, pf in zip(gt_files, pred_files):
        image_name = gf.stem
        gcls, gboxes, _ = load_txt(gf)
        pcls, pboxes, pconfs = load_txt(pf)
        
        # Convert boxes for calculations
        gboxes_xyxy = [xywh_to_xyxy_norm(b) for b in gboxes]
        pboxes_xyxy = [xywh_to_xyxy_norm(b) for b in pboxes]
        
        img_localization = []
        matched_gt_indices = set()
        
        # Process each prediction
        for pc, pb, conf in zip(pcls, pboxes, pconfs):
            pxy = xywh_to_xyxy_norm(pb)
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching GT
            for j, (gc, gxy) in enumerate(zip(gcls, gboxes_xyxy)):
                if gc == pc:
                    iou = compute_iou(pxy, gxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            # Record if TP
            if best_iou >= iou_thres and best_gt_idx != -1 and best_gt_idx not in matched_gt_indices:
                matched_gt_indices.add(best_gt_idx)
                
                # Initialize class stats if needed
                if pc not in class_stats:
                    class_stats[pc] = {"ious_tp": [], "mses": []}
                
                # Calculate MSE
                gb = gboxes_xyxy[best_gt_idx]
                mse = np.mean((np.array(pxy) - np.array(gb))**2)
                
                # Update class stats
                class_stats[pc]["ious_tp"].append(best_iou)
                class_stats[pc]["mses"].append(mse)
                
                # Record per-detection localization
                img_localization.append({
                    "class": pc,
                    "pred_bbox": pb,
                    "gt_bbox": gboxes[best_gt_idx],
                    "iou": best_iou,
                    "mse": mse,
                    "status": "TP"
                })
        
        # Store per-image localization data
        per_image_localization.append({
            "image": image_name,
            "localization": img_localization
        })
    
    # Build localization table
    loc_table = []
    for cid, stats in class_stats.items():
        ious = np.array(stats["ious_tp"]) if stats["ious_tp"] else np.array([0.0])
        mses = np.array(stats["mses"]) if stats["mses"] else np.array([0.0])
        
        loc_table.append({
            "Class_ID": cid,
            "Avg_IoU_TP": float(ious.mean()),
            "IoU_Std": float(ious.std()),
            "Dice_Coefficient": float((2 * ious / (1 + ious)).mean()),
            "Box_MSE": float(mses.mean()),
            "Num_TP": len(ious)
        })
    
    return loc_table, per_image_localization

def main():
    args = parse_args()
    
    # Run detection evaluation
    per_image_stats, overall_metrics = evaluate_detection(
        args.gt_dir, args.pred_dir, args.iou_thres
    )
    
    # Run localization evaluation
    loc_table, per_image_localization = evaluate_localization(
        args.gt_dir, args.pred_dir, args.iou_thres
    )
    
    # Compile comprehensive output
    output = {
        "overview": {
            "gt_directory": str(args.gt_dir),
            "pred_directory": str(args.pred_dir),
            "iou_threshold": args.iou_thres
        },
        "performance_metrics": overall_metrics,
        "localization_table": loc_table,
        "per_image_detection_stats": per_image_stats,
        "per_image_localization_data": per_image_localization
    }
    
    # Print to stdout
    print(json.dumps(output, indent=2))
    
    # Save to file
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved comprehensive statistics to {out_path}")

if __name__ == "__main__":
    main()