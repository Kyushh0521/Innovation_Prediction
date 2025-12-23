import os
import sys
import yaml
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict


def now_str():
    """返回当前时间的字符串表示，格式为 YYYYmmdd_HHMMSS。用于命名输出目录等。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(output_dir: str, log_name: str = "train.log"):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_name)

    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"日志已初始化：{log_path}")
    return log_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
