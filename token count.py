import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    r"XXX",
    use_fast=True
)

BATCH_SIZE = 2000

total_tokens = 0
count = 0
texts = []

with open(r"XXX.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = item.get("text") or item.get("content") or ""
        texts.append(text)

        if len(texts) >= BATCH_SIZE:
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_length=True,
                padding=False,
                truncation=False
            )

            total_tokens += sum(encoded["length"])
            count += len(texts)
            texts.clear()

    # 处理剩余部分
    if texts:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_length=True,
            padding=False,
            truncation=False
        )

        total_tokens += sum(encoded["length"])
        count += len(texts)

print(f"样本数: {count:,}")
print(f"总 Token 数: {total_tokens:,}")
print(f"平均 Token 数: {total_tokens / count:.1f}")