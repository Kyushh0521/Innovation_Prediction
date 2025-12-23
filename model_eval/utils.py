# utils.py
import yaml
import json
import logging
import os
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def init_logging(out_dir: str, log_name: str = "evaluator.log"):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, log_name)
    
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"日志初始化完成: {log_path}")

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(cfg_model: Dict):
    """
    统一加载 LLM 模型和 Tokenizer
    """
    model_path = cfg_model["model_name_or_path"]
    adapter_path = cfg_model.get("adapter_name_or_path")
    cache_dir = cfg_model.get("model_cache")

    logging.info(f"加载基座模型: {model_path}")
    if adapter_path:
        logging.info(f"检测到 LoRA 适配器: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
        attn_implementation="sdpa"  # 使用更高效的注意力实现
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, device_map="auto")

    model.eval()
    return model, tokenizer

def get_generate_fn(model, tokenizer, cutoff_len: int = 2048):
    """
    返回一个闭包函数用于文本生成（接受 system_prompt、instruction、input_text）。
    """
    def generate(system_prompt: str, instruction: str, input_text: str) -> str:
        if input_text:
            user_msg = f"{instruction}\n\n输入：\n{input_text}"
        else:
            user_msg = instruction

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("当前 tokenizer 不支持对话模板（apply_chat_template）。")

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=cutoff_len
        )

        try:
            first_param = next(model.parameters())
            device = first_param.device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_inputs = model_inputs.to(device)

        with torch.no_grad():
            generated = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=False, # 保持确定性
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = model_inputs.input_ids.shape[1]
        if generated.shape[1] <= input_len:
            return ""

        output_ids = generated[0][input_len:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        return response.strip()

    return generate