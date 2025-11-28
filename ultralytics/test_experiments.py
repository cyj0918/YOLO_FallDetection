#!/usr/bin/env python3
"""
test_experiments.py

自動執行YOLO測試實驗，包含記憶體清理以確保準確的推理速度測試
針對已訓練的模型執行多輪測試

Usage: python3 test_experiments.py --start-exp 2 --end-exp 10
"""

import subprocess
import os
import sys
import argparse
import gc
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="自動執行YOLO測試實驗")
    parser.add_argument("--start-exp", type=int, default=5, help="起始實驗編號")
    parser.add_argument("--end-exp", type=int, default=10, help="結束實驗編號")
    parser.add_argument("--num-tests", type=int, default=3, help="每個實驗的測試次數")
    parser.add_argument("--dry-run", action="store_true", help="預覽指令而不實際執行")
    parser.add_argument("--memory-wait", type=int, default=5, help="記憶體清理後等待秒數")
    return parser.parse_args()


def clean_memory():
    """清理記憶體以確保測試準確性"""
    print("🧹 清理記憶體中...")

    # Python垃圾回收
    collected = gc.collect()
    print(f"   - Python GC回收了 {collected} 個物件")

    # 強制進行額外的垃圾回收
    for i in range(3):
        gc.collect()

    # 嘗試清理GPU記憶體 (如果使用CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("   - CUDA記憶體已清空")
    except ImportError:
        pass

    # 嘗試清理MPS記憶體 (如果使用Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("   - MPS記憶體已清空")
    except (ImportError, AttributeError):
        pass

    print("✓ 記憶體清理完成")


def wait_for_system_settle(wait_time):
    """等待系統穩定"""
    if wait_time > 0:
        print(f"⏳ 等待系統穩定 ({wait_time} 秒)...")
        time.sleep(wait_time)


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


def check_model_exists(exp_num):
    """檢查模型檔案是否存在"""
    model_path = f"/Users/jhen/Documents/CUHK-Project/ultralytics/exp{exp_num}/exp{exp_num}/weights/best.pt"
    exists = os.path.exists(model_path)
    if not exists:
        print(f"⚠️  模型檔案不存在: {model_path}")
    return exists


def test_experiment(exp_num, test_num, memory_wait, dry_run=False):
    """執行單一測試"""
    print(f"\n{'-'*50}")
    print(f"測試 EXP{exp_num} - TEST{test_num}")
    print(f"{'-'*50}")

    # 檢查模型是否存在
    if not dry_run and not check_model_exists(exp_num):
        return False

    # 記憶體清理和等待
    print("🔄 準備測試環境...")
    clean_memory()
    wait_for_system_settle(memory_wait)

    # 測試指令
    test_cmd = (f"python3 tests/exp_evaluate.py "
                f"--source /Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/images/test/ "
                f"--dataset-config ultralytics/cfg/datasets/mf.yaml "
                f"--model /Users/jhen/Documents/CUHK-Project/ultralytics/exp{exp_num}/exp{exp_num}/weights/best.pt "
                f"--project exp{exp_num} --name test{exp_num}-{test_num}")

    print("🚀 開始推理測試...")
    if not run_command(test_cmd, dry_run):
        return False

    print("📊 生成統計資料...")
    # 統計指令
    stats_cmd = (f"python3 tests/exp_statistic.py "
                 f"--gt-dir /Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/labels/test/ "
                 f"--pred-dir /Users/jhen/Documents/CUHK-Project/ultralytics/exp{exp_num}/test{exp_num}-{test_num}2/labels "
                 f"--output /Users/jhen/Documents/CUHK-Project/ultralytics/exp{exp_num}/test{exp_num}-{test_num}/statistics.json")

    return run_command(stats_cmd, dry_run)


def main():
    args = parse_args()

    print(f"開始測試實驗 exp{args.start_exp} 到 exp{args.end_exp}")
    print(f"每個實驗將執行 {args.num_tests} 次測試")
    print(f"記憶體清理等待時間: {args.memory_wait} 秒")
    if args.dry_run:
        print("預覽模式: 指令將被顯示但不執行")

    print(f"\n{'#'*80}")
    print("開始測試所有實驗")
    print(f"{'#'*80}")

    failed_tests = []
    successful_tests = []
    skipped_exps = []

    for exp_num in range(args.start_exp, args.end_exp + 1):
        print(f"\n{'='*60}")
        print(f"測試實驗 {exp_num}")
        print(f"{'='*60}")

        # 檢查模型是否存在（在dry-run模式下跳過）
        if not args.dry_run and not check_model_exists(exp_num):
            print(f"跳過 exp{exp_num} (模型檔案不存在)")
            skipped_exps.append(exp_num)
            continue

        exp_success_count = 0
        for test_num in range(1, args.num_tests + 1):
            test_name = f"exp{exp_num}-test{test_num}"
            success = test_experiment(exp_num, test_num, args.memory_wait, args.dry_run)

            if success:
                successful_tests.append(test_name)
                exp_success_count += 1
                print(f"✓ {test_name} 完成")
            else:
                failed_tests.append(test_name)
                print(f"✗ {test_name} 失敗")

        print(f"實驗 {exp_num} 完成: {exp_success_count}/{args.num_tests} 測試成功")

    # 總結
    print(f"\n{'#'*80}")
    print("測試總結")
    print(f"{'#'*80}")

    total_exps = args.end_exp - args.start_exp + 1
    tested_exps = total_exps - len(skipped_exps)
    total_tests = tested_exps * args.num_tests

    print(f"總實驗數: {total_exps}")
    print(f"已測試實驗: {tested_exps}")
    print(f"跳過實驗: {len(skipped_exps)}")
    print(f"總測試數: {total_tests}")
    print(f"成功測試: {len(successful_tests)}")
    print(f"失敗測試: {len(failed_tests)}")

    if skipped_exps:
        print(f"跳過的實驗: {skipped_exps}")
    if successful_tests:
        print(f"成功的測試: {successful_tests}")
    if failed_tests:
        print(f"失敗的測試: {failed_tests}")

    if not failed_tests and not skipped_exps:
        print("🎉 所有測試都成功完成！")
    elif not failed_tests:
        print("✅ 所有可測試的實驗都成功完成！")
    else:
        print("⚠️  部分測試失敗，請檢查上方輸出")


if __name__ == "__main__":
    main()
