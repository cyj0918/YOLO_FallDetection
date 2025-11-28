import os

base_gt_dir = "/Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/labels/test"
base_pred_dir_pattern = "/Users/jhen/Documents/CUHK-Project/ultralytics/exp{exp}/test{exp}-{test}2/labels"
base_output_pattern = "/Users/jhen/Documents/CUHK-Project/ultralytics/exp{exp}/test{exp}-{test}/statistics.json"

for exp in range(11):  # 0 到 10
    for test in range(1, 4):  # 1 到 3
        gt_dir = base_gt_dir
        pred_dir = base_pred_dir_pattern.format(exp=exp, test=test)
        output = base_output_pattern.format(exp=exp, test=test)
        cmd = f"python3 tests/exp_statistic.py --gt-dir {gt_dir} --pred-dir {pred_dir} --output {output}"
        print(f"執行：{cmd}")
        os.system(cmd)
