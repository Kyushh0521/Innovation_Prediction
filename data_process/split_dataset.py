import json
import random
from pathlib import Path

# ------------------ 配置区（仅在此处修改） ------------------
INPUT_PATH = Path("data_process_outputs/sample_sft.json")
OUTPUT_TRAIN = Path("data_process_outputs/sample_sft_train.json")
OUTPUT_VAL = Path("data_process_outputs/sample_sft_val.json")
OUTPUT_TEST = Path("data_process_outputs/sample_sft_test.json")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
# -----------------------------------------------------------


def load_items(p: Path):
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def save_json(items, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def split_and_save():
    items = load_items(INPUT_PATH)
    rnd = random.Random(SEED)
    rnd.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:n_train + n_val + n_test]
    save_json(train, OUTPUT_TRAIN)
    save_json(val, OUTPUT_VAL)
    save_json(test, OUTPUT_TEST)
    print(f"加载 {n} 条记录。训练集: {len(train)}。验证集: {len(val)}。测试集: {len(test)}")


if __name__ == "__main__":
    split_and_save()
