#!/usr/bin/env python3
"""
train_experiments.py

自動執行YOLO訓練實驗
從指定的實驗編號開始訓練多個實驗

Usage: python3 train_experiments.py --start-exp 4 --end-exp 10
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser(description="自動執行YOLO訓練實驗")
    parser.add_argument("--start-exp", type=int, default=5, help="起始實驗編號")
    parser.add_argument("--end-exp", type=int, default=10, help="結束實驗編號")
    parser.add_argument("--dry-run", action="store_true", help="預覽指令而不實際執行")
    return parser.parse_args()


def run_command(cmd, dry_run=False):
    """執行shell指令"""
    print(f"執行中: {cmd}")
    if dry_run:
        print("[預覽模式] 指令將被執行")
        return True

    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ 指令執行成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 指令執行失敗，返回碼 {e.returncode}")
        print(f"錯誤輸出: {e.stderr}")
        return False


def train_experiment(exp_num, dry_run=False):
    """訓練單一實驗"""
    print(f"\n{'='*60}")
    print(f"訓練實驗 {exp_num}")
    print(f"{'='*60}")

    cmd = (f"yolo detect train "
           f"data=ultralytics/cfg/datasets/mf.yaml "
           f"model=ultralytics/cfg/models/modified/exp{exp_num}.yaml "
           f"epochs=100 batch=16 lr0=0.01 optimizer=SGD device=mps "
           f"plots=True save=True project=exp{exp_num} name=exp{exp_num}")

    return run_command(cmd, dry_run)


def main():
    args = parse_args()

    print(f"開始訓練實驗 exp{args.start_exp} 到 exp{args.end_exp}")
    if args.dry_run:
        print("預覽模式: 指令將被顯示但不執行")

    print(f"\n{'#'*80}")
    print("開始訓練所有實驗")
    print(f"{'#'*80}")

    failed_trains = []
    successful_trains = []

    for exp_num in range(args.start_exp, args.end_exp + 1):
        success = train_experiment(exp_num, args.dry_run)
        if success:
            successful_trains.append(exp_num)
            print(f"✓ exp{exp_num} 訓練成功")
        else:
            failed_trains.append(exp_num)
            print(f"✗ exp{exp_num} 訓練失敗")

    # 總結
    print(f"\n{'#'*80}")
    print("訓練總結")
    print(f"{'#'*80}")

    total_exps = args.end_exp - args.start_exp + 1
    print(f"總實驗數: {total_exps}")
    print(f"成功訓練: {len(successful_trains)} 個實驗")
    print(f"失敗訓練: {len(failed_trains)} 個實驗")

    if successful_trains:
        print(f"成功的實驗: {successful_trains}")
    if failed_trains:
        print(f"失敗的實驗: {failed_trains}")

    if not failed_trains:
        print("🎉 所有訓練實驗都成功完成！")
        print("現在可以執行測試腳本進行推理測試")
    else:
        print("⚠️  部分訓練實驗失敗，請檢查上方輸出")


if __name__ == "__main__":
    main()
