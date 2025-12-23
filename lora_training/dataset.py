import logging
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial

from utils import load_json


class SFTDataset(Dataset):
    """
    Qwen3 官方风格 SFT Dataset

    核心特性：
    1. 使用 apply_chat_template 构造完整对话
    2. 单次 tokenize
    3. 仅 assistant 部分参与 loss
    4. pad_token = eos_token
    """

    def __init__(self, path: str, tokenizer, cfg: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.cfg = cfg

        self.max_length = cfg.get("max_length", 4096)
        self.ignore_index = -100

        all_items = load_json(path)
        self.items = []
        
        # 简单过滤：确保 output 不为空，避免显而易见的无效数据
        # (深度过滤需要 tokenize，会拖慢启动速度，此处做基础过滤即可)
        for rec in all_items:
            if rec.get("output", "").strip():
                self.items.append(rec)

        # === Qwen 官方推荐：pad_token 使用 eos_token ===
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logging.info(f"已加载 SFT 数据集：{path}，总样本={len(all_items)}，有效样本={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def _build_messages(self, rec: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        构建 Qwen 对话消息
        """
        system = rec.get("system", "You’re a professional AI assistant, delivering accurate, comprehensive, consistent responses, adhering to constraints with neutral tone and clear structure.")
        instruction = rec.get("instruction", "")
        inp = rec.get("input", "")
        output = rec.get("output", "")

        user_content = instruction
        if inp:
            user_content = instruction + "\n" + inp

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        return messages

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        messages = self._build_messages(rec)

        # === 1. 构造完整对话文本（包含 assistant 回复）===
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # SFT 时必须为 False
        )

        # === 2. 单次 tokenize ===
        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # === 3. 构建 labels：只训练 assistant ===
        labels = input_ids.clone()

        # 重新构造「不含 assistant 的 prompt」，用于定位 assistant 起点
        prompt_messages = messages[:-1]  # 去掉 assistant
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,  # 关键：生成 <|im_start|>assistant\n
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]

        # 如果 prompt 本身就超过了 input_ids 的长度（说明整个输入都被截断了，或者 prompt 比 max_length 还长）
        # 我们必须 clamp prompt_len，否则 labels[:prompt_len] 会报错
        prompt_len = len(prompt_ids)
        if prompt_len > len(input_ids):
            prompt_len = len(input_ids)

        # prompt 部分全部 ignore
        labels[:prompt_len] = self.ignore_index

        # === 4. 极端情况处理：assistant 被完全截断 ===
        if (labels != self.ignore_index).sum().item() == 0:
            # 返回 None，由 collate_fn 过滤
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def collate_fn(batch, tokenizer):
    """Qwen SFT 专用 collate_fn

    - 自动过滤 None
    - 动态 padding
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return {}

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0,
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def get_dataloader(ds, tokenizer, cfg, shuffle=True):
    batch_size = cfg.get("batch_size", 8)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        drop_last=True,
    )
    return dl