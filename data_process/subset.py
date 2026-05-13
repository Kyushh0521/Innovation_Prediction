import argparse
import json
import random
from typing import List

from matplotlib.dates import num2date


def sample_items(items: List, k: int, seed: int | None = None) -> List:
    rnd = random.Random(seed)
    if k >= len(items):
        return list(items)
    return rnd.sample(items, k)


def main():
    input = "datasets/medical/med_test.json"
    output = "datasets/medical/med_test_subset.json"
    num = None
    frac = 1
    seed = 42

    with open(input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("json 文件应包含最外层数组以便抽样")

    total = len(data)
    if frac is not None:
        k = int(total * frac)
    elif num is not None:
        k = num
    else:
        raise SystemExit("请指定 num 或 frac")

    sample = sample_items(data, k, seed)
    # 打印原数据集与子集条数
    print(f"原数据集条数: {total}")
    print(f"子集条数: {len(sample)}")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
