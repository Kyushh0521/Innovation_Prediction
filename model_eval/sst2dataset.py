import logging
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd

def load_parquet(path: str) -> List[Dict]:
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")

class SST2ClassificationDataset(Dataset):
    """
    针对 SST2 Parquet 数据的分类任务 Dataset
    """
    def __init__(self, path: str, tokenizer, cfg: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.max_length = cfg.get("max_length", 128)
        
        # 加载 Parquet 数据
        self.items = load_parquet(path)
        logging.info(f"已加载 Parquet 分类数据集：{path}，总样本={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        # SST2 标准字段：sentence (文本), label (0 或 1)
        text = item["sentence"]
        label = int(item["label"])

        # 序列分类不需要对话模板，直接 Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False, 
            return_tensors=None
        )

        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def collate_fn_cls(batch, tokenizer):
    """分类任务专用的动态 Padding"""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels
    }

def get_dataloader(ds, tokenizer, cfg, shuffle=True):
    batch_size = cfg.get("batch_size", 16)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=partial(collate_fn_cls, tokenizer=tokenizer),
        drop_last=True,
    )
    return dl