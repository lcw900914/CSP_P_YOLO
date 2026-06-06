"""
訓練啟動器 — 用 subprocess.Popen 讓 stdout 正確寫入 log 檔案
使用方式：python launch_train.py
"""
import subprocess, sys, os
from pathlib import Path

LOG_FILE = "/home/lcw/CSPPartialYOLO/checkpoints/train_log.txt"
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-u",
    str(Path(__file__).parent / "train.py"),
    "--train_dir", str(Path(__file__).parent / "datasets/dota/dota/train"),
    "--val_dir",   str(Path(__file__).parent / "datasets/dota/dota/val"),
    "--output",    str(Path(__file__).parent / "checkpoints"),
    "--epochs",    "300",
    "--batch",     "28",
    "--lr",        "0.010",
    "--warmup",    "10",
    "--workers",   "4",
    "--dota_only_val",
]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(LOG_FILE, "a", encoding="utf-8") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)

print(f"Training started  PID={proc.pid}")
print(f"Log → {LOG_FILE}")
print("此視窗可以關閉，訓練在背景繼續執行。")
print("查看進度：tail -f", LOG_FILE)
