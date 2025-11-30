#!/usr/bin/env python3
"""
yolo_predict_stream_eval.py

Run YOLO inference using Ultralytics built-in predict mode with streaming,
measure timing and system metrics, and compute ground-truth metrics by reading
predict and ground-truth TXT files. Output a comprehensive JSON summary with
per-image detailed data.
"""

import argparse
import sys
import time
import torch
import numpy as np
import psutil
import json
import yaml
from pathlib import Path
from datetime import datetime
import thop
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO streaming inference + metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="Image/video directory for inference")
    parser.add_argument("--dataset-config", "-d", type=str, default=None,
                        help="YAML dataset config for GT metrics (unused for path)")
    parser.add_argument("--conf", "-c", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2,
                        help="IoU threshold for NMS and GT matching")
    parser.add_argument("--imgsz", "--img-size", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps"], help="Compute device")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated results")
    parser.add_argument("--save-txt", action="store_true", default=True,
                        help="Save predictions in YOLO TXT format")
    parser.add_argument("--save-conf", action="store_true", default=True,
                        help="Include confidence in TXT")
    parser.add_argument("--enable-metrics", action="store_true", default=True,
                        help="Compute ground-truth metrics")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Ultralytics project folder")
    parser.add_argument("--name", type=str, default="exp",
                        help="Ultralytics run name")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def measure_model_info(model_path, device):
    model = YOLO(model_path)
    model.model = model.model.to(device)
    dummy = torch.randn(1, 3, 640, 640).to(device)
    flops, params = thop.profile(model.model, inputs=(dummy,), verbose=False)
    return {
        "model_path": model_path,
        "parameters_M": params / 1e6,
        "flops_G": flops / 1e9,
        "device": str(device),
        "model_size_mb": Path(model_path).stat().st_size / (1024 * 1024),
        "timestamp": datetime.now().isoformat()
    }


def measure_system_metrics(device):
    mem = psutil.virtual_memory()
    mem_mb = (mem.total - mem.available) / (1024 * 1024)
    mps_mb = 0
    if device.type == "mps":
        try:
            mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        except:
            mps_mb = mem_mb
    return mem_mb, mps_mb


def load_txt(path):
    cls_list, boxes, confs = [], [], []
    if Path(path).exists():
        for line in open(path):
            parts = line.split()
            cls_list.append(int(parts[0]))
            box = list(map(float, parts[1:5]))
            boxes.append(box)
            confs.append(float(parts[5]) if len(parts) > 5 else 1.0)
    return cls_list, boxes, confs


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x - w/2, y - h/2, x + w/2, y + h/2]


def compute_iou(a, b):
    xa1, ya1, xa2, ya2 = xywh_to_xyxy(a)
    xb1, yb1, xb2, yb2 = xywh_to_xyxy(b)
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0


def load_gt(txt_path):
    return load_txt(txt_path)[:2]  # cls, boxes


def evaluate_gt_per_image(pred_txt, gt_txt, iou_thres):
    pcls, pboxes, pconfs = load_txt(pred_txt)
    gcls, gboxes = load_gt(gt_txt)
    
    detections = []
    matched_gt_indices = set()
    
    # Process each prediction
    for i, (c, box, conf) in enumerate(zip(pcls, pboxes, pconfs)):
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find best matching GT
        for j, (gc, gbox) in enumerate(zip(gcls, gboxes)):
            if gc == c:
                iou = compute_iou(box, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        # Determine detection status
        status = "TP" if best_iou >= iou_thres and best_gt_idx not in matched_gt_indices else "FP"
        if status == "TP":
            matched_gt_indices.add(best_gt_idx)
        
        detections.append({
            "class": c,
            "confidence": conf,
            "bbox": box,
            "iou": best_iou,
            "status": status
        })
    
    # Identify false negatives
    for j in range(len(gboxes)):
        if j not in matched_gt_indices:
            detections.append({
                "class": gcls[j],
                "confidence": 0.0,
                "bbox": gboxes[j],
                "iou": 0.0,
                "status": "FN"
            })
    
    return detections


def main():
    args = parse_args()
    device = (torch.device(args.device))
    model_info = measure_model_info(args.model, device)
    model = YOLO(args.model)

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    lat, pre, post, mem, mps = [], [], [], [], []
    detailed_results = []
    print("Running streaming inference...")
    start_all = time.perf_counter()
    
    for res in model(
        args.source,
        conf=args.conf,
        iou=args.iou,
        device=device,
        imgsz=args.imgsz,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        verbose=False,
        stream=True
    ):
        s = res.speed
        pre_time = s.get("preprocess", 0)
        inf_time = s.get("inference", 0)
        post_time = s.get("postprocess", 0)
        mu, mm = measure_system_metrics(device)
        
        # Store per-image metrics
        img_data = {
            "image_path": str(res.path),
            "preprocess_time_ms": pre_time,
            "inference_time_ms": inf_time,
            "postprocess_time_ms": post_time,
            "memory_usage_mb": mu,
            "mps_memory_mb": mm,
            "num_detections": len(res.boxes) if res.boxes else 0
        }
        
        # Store detection details if available
        detections = []
        if res.boxes:
            for box in res.boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                bbox = box.xywhn[0].tolist()  # normalized xywh
                detections.append({
                    "class": cls_id,
                    "confidence": conf,
                    "bbox": bbox
                })
        img_data["detections"] = detections
        
        detailed_results.append(img_data)
        pre.append(pre_time)
        lat.append(inf_time)
        post.append(post_time)
        mem.append(mu)
        mps.append(mm)
        total += 1
    
    end_all = time.perf_counter()

    # Timing metrics
    tm = {
        "total_images": total,
        "avg_inference_time_ms": float(np.mean(lat)) if lat else 0.0,
        "std_inference_time_ms": float(np.std(lat)) if lat else 0.0,
        "median_inference_time_ms": float(np.median(lat)) if lat else 0.0,
        "min_inference_time_ms": float(np.min(lat)) if lat else 0.0,
        "max_inference_time_ms": float(np.max(lat)) if lat else 0.0,
        "fps": (1000/np.mean(lat)) if lat else 0.0,
        "avg_preprocess_time_ms": float(np.mean(pre)) if pre else 0.0,
        "avg_postprocess_time_ms": float(np.mean(post)) if post else 0.0,
        "total_time_ms": (end_all - start_all) * 1000
    }

    # System metrics
    sm = {
        "avg_memory_usage_mb": float(np.mean(mem)) if mem else 0.0,
        "max_memory_usage_mb": float(np.max(mem)) if mem else 0.0,
        "avg_mps_memory_mb": float(np.mean(mps)) if mps else 0.0,
        "max_mps_memory_mb": float(np.max(mps)) if mps else 0.0
    }

    # Ground truth evaluation
    gt_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_dir = save_dir / "labels"
    src_parts = Path(args.source).parts
    gt_dir = Path(*src_parts[:src_parts.index("images")], "labels", *src_parts[src_parts.index("images")+1:]) if "images" in src_parts else None

    if args.enable_metrics and gt_dir and gt_dir.exists():
        print("Computing ground-truth metrics...")
        all_detections = []
        pfs = sorted(Path(pred_dir).glob("*.txt"))
        gfs = sorted(gt_dir.glob("*.txt"))
        
        for pf, gf in zip(pfs, gfs):
            base_name = pf.stem
            img_detections = evaluate_gt_per_image(pf, gf, args.iou)
            all_detections.append({
                "image": base_name,
                "detections": img_detections
            })
        
        # Calculate overall metrics
        tp = sum(1 for img in all_detections for d in img["detections"] if d["status"] == "TP")
        fp = sum(1 for img in all_detections for d in img["detections"] if d["status"] == "FP")
        fn = sum(1 for img in all_detections for d in img["detections"] if d["status"] == "FN")
        
        prec = tp/(tp+fp) if tp+fp > 0 else 0.0
        rec = tp/(tp+fn) if tp+fn > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
        
        gt_metrics = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
    else:
        all_detections = []
        print(f"Warning: GT evaluation skipped for {gt_dir}")

    # Compile final output
    output = {
        "model_info": model_info,
        "timing_metrics": tm,
        "system_metrics": sm,
        "ground_truth_metrics": gt_metrics,
        "per_image_metrics": detailed_results,
        "per_image_detections": all_detections
    }

    # Save comprehensive results
    out_file = save_dir / "evaluation_summary.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved detailed results to {out_file}")

    print("\n=== Performance Summary ===")
    print(f"Images processed: {total}")
    print(f"Avg inference: {tm['avg_inference_time_ms']:.2f} ms, FPS: {tm['fps']:.2f}")
    if gt_metrics["precision"] > 0:
        print(f"Precision: {gt_metrics['precision']:.4f}, Recall: {gt_metrics['recall']:.4f}, F1: {gt_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()