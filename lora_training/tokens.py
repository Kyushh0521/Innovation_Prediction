import json
from transformers import AutoTokenizer

# 1. 手动设置路径
model_path = "Qwen/Qwen2.5-0.5B-Instruct" 
data_path = "datasets\medical\med_train.json"

# 2. 加载工具
tokenizer = AutoTokenizer.from_pretrained(model_path)
data = json.load(open(data_path, "r", encoding="utf-8"))

# 3. 统计长度
lengths = []
for item in data:
    # 模拟简单的拼接逻辑
    content = item.get("instruction", "") + item.get("input", "") + item.get("output", "")
    lengths.append(len(tokenizer.encode(content)))

# 4. 打印结果
print(f"总计样本: {len(lengths)}")
print(f"最小长度: {min(lengths)}")
print(f"最大长度: {max(lengths)}")
print(f"平均长度: {sum(lengths) / len(lengths):.1f}")