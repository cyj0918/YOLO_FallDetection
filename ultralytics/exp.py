# yolo train model=yolo11n.pt data=dataset/FallDataset/data.yaml epochs=10 batch=16 device=mps
# yolo train model=ultralytics/cfg/models/modified/exp1.yaml data=dataset/FallDataset/data.yaml epochs=100 imgsz=224 batch=4 device=mps

# yolo predict model=runs/detect/train4/weights/best.pt source=/Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/images/test/
# yolo detect val data=yolo11n.pt data=dataset/FallDataset/data.yaml model=runs\detect\train4\weights\best.pt batch=16 device=0

# Train: yolo detect train data=dataset/FallDataset/data.yaml model=ultralytics/ultralytics/cfg/models/modified/exp0.yaml epochs=100 batch=16 lr0=0.01 optimizer=SGD device=mps plots=True save=True project=exp0 name=exp0
# Detect: python3 tests/exp_evaluate.py --source /Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/images/test/  --dataset-config dataset/FallDataset/data.yaml --model /Users/jhen/Documents/CUHK-Project/ultralytics/exp10/exp10/weights/best.pt --project exp10 --name test10-1
# Statistic: python3 tests/exp_statistic.py --gt-dir /Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/labels/test/ --pred-dir /Users/jhen/Documents/CUHK-Project/ultralytics/exp6/test6-12/labels --output /Users/jhen/Documents/CUHK-Project/ultralytics/exp6/test6-1/statistics.json

#!/usr/bin/env python3
"""
exp.py

Automatically run YOLO training and testing experiments starting from exp5.
For each experiment:
1. Train the model
2. Run 3 tests (test{N}-1, test{N}-2, test{N}-3)
3. Generate statistics for each test

Usage: python3 exp.py --start-exp 5 --end-exp 10
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO experiments automatically")
    parser.add_argument("--start-exp", type=int, default=5, help="Starting experiment number")
    parser.add_argument("--end-exp", type=int, default=10, help="Ending experiment number")
    parser.add_argument("--num-tests", type=int, default=3, help="Number of tests per experiment")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()

def run_command(cmd, dry_run=False):
    """Execute a shell command"""
    print(f"Running: {cmd}")
    if dry_run:
        print("[DRY RUN] Command would be executed")
        return True
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def train_experiment(exp_num, dry_run=False):
    """Train a single experiment"""
    print(f"\n{'='*60}")
    print(f"TRAINING EXPERIMENT {exp_num}")
    print(f"{'='*60}")
    
    cmd = (f"yolo detect train "
           f"data=ultralytics/cfg/datasets/fall.yaml "
           f"model=ultralytics/cfg/models/modified/exp{exp_num}.yaml "
           f"epochs=100 batch=16 lr0=0.01 optimizer=SGD device=mps "
           f"plots=True save=True project=exp{exp_num} name=exp{exp_num}")
    
    return run_command(cmd, dry_run)

def test_experiment(exp_num, test_num, dry_run=False):
    """Run a single test for an experiment"""
    print(f"\n{'-'*40}")
    print(f"TESTING EXP{exp_num} - TEST{test_num}")
    print(f"{'-'*40}")
    
    # Test command
    test_cmd = (f"python3 /tests/exp_evaluate.py "
                f"--source /dataset/FallDataset/images/test/ "
                f"--dataset-config /ultralytics/cfg/datasets/fall.yaml "
                f"--model /ultralytics/exp{exp_num}/exp{exp_num}/weights/best.pt "
                f"--project exp{exp_num} --name test{exp_num}-{test_num}")
    
    if not run_command(test_cmd, dry_run):
        return False
    
    # Statistics command
    stats_cmd = (f"python3 /tests/exp_statistic.py "
                 f"--gt-dir /dataset/FallDataset/labels/test/ "
                 f"--pred-dir /ultralytics/exp{exp_num}/test{exp_num}-{test_num}/labels "
                 f"--output /ultralytics/exp{exp_num}/test{exp_num}-{test_num}/statistics.json")
    
    return run_command(stats_cmd, dry_run)

def main():
    args = parse_args()
    
    print(f"Starting experiments from exp{args.start_exp} to exp{args.end_exp}")
    print(f"Each experiment will have {args.num_tests} tests")
    if args.dry_run:
        print("DRY RUN MODE: Commands will be printed but not executed")
    
    # Phase 1: Train all experiments
    print(f"\n{'#'*80}")
    print("PHASE 1: TRAINING ALL EXPERIMENTS")
    print(f"{'#'*80}")
    
    failed_trains = []
    for exp_num in range(args.start_exp, args.end_exp + 1):
        success = train_experiment(exp_num, args.dry_run)
        if not success:
            failed_trains.append(exp_num)
            print(f"⚠️  exp{exp_num} training failed, but continuing...")
    
    if failed_trains:
        print(f"\n⚠️  Failed training experiments: {failed_trains}")
        response = input("Continue with testing? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Phase 2: Test all experiments
    print(f"\n{'#'*80}")
    print("PHASE 2: TESTING ALL EXPERIMENTS")
    print(f"{'#'*80}")
    
    failed_tests = []
    for exp_num in range(args.start_exp, args.end_exp + 1):
        if exp_num in failed_trains:
            print(f"Skipping tests for exp{exp_num} (training failed)")
            continue
            
        print(f"\n{'='*60}")
        print(f"TESTING EXPERIMENT {exp_num}")
        print(f"{'='*60}")
        
        for test_num in range(1, args.num_tests + 1):
            success = test_experiment(exp_num, test_num, args.dry_run)
            if not success:
                failed_tests.append(f"exp{exp_num}-test{test_num}")
    
    # Summary
    print(f"\n{'#'*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'#'*80}")
    
    total_exps = args.end_exp - args.start_exp + 1
    successful_trains = total_exps - len(failed_trains)
    total_tests = successful_trains * args.num_tests
    successful_tests = total_tests - len(failed_tests)
    
    print(f"Training: {successful_trains}/{total_exps} successful")
    print(f"Testing: {successful_tests}/{total_tests} successful")
    
    if failed_trains:
        print(f"Failed training: {failed_trains}")
    if failed_tests:
        print(f"Failed tests: {failed_tests}")
    
    if not failed_trains and not failed_tests:
        print("🎉 All experiments completed successfully!")
    else:
        print("⚠️  Some experiments failed. Check the output above.")

if __name__ == "__main__":
    main()