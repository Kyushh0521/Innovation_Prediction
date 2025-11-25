import json
import random
from pathlib import Path

# ------------------ 配置区（仅在此处修改） ------------------
INPUT_PATH = Path("data_process_outputs/sample_preference_dataset.json")
OUTPUT_TRAIN = Path("data_process_outputs/sample_train_dataset.train.json")
OUTPUT_VAL = Path("data_process_outputs/sample_val_dataset.json")
RATIO = 0.9  # 训练集比例
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
    cut = int(len(items) * RATIO)
    train = items[:cut]
    val = items[cut:]
    save_json(train, OUTPUT_TRAIN)
    save_json(val, OUTPUT_VAL)
    print(f"加载 {len(items)} 条记录。训练集: {len(train)}。验证集: {len(val)}")


if __name__ == "__main__":
    split_and_save()
