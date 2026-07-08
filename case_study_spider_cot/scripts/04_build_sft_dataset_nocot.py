"""
构建 No-CoT 对照实验的 SFT 训练数据。
模型直接学习 (question + schema) -> SQL，不包含 CoT 推理过程。
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "processed_data"
OUTPUT_DIR = PROJECT_DIR / "sft_data"

MAX_SCHEMA_CHARS = 6000
TEST_SCHEMA_CHARS = 12000

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural language question, "
    "generate the correct SQL query."
)


def truncate_text(text: str, max_chars: int, suffix: str = "... (truncated)") -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    return text[: max_chars - len(suffix)] + suffix


def truncate_schema(schema: str, max_chars: int = None) -> str:
    limit = max_chars if max_chars is not None else MAX_SCHEMA_CHARS
    if len(schema) <= limit:
        return schema
    lines = schema.split("\n")
    short = [l for l in lines if not l.strip().startswith("Sample(")]
    result = "\n".join(short)
    return truncate_text(result, limit, "\n... (schema truncated)")


def build_user_prompt(item: dict, max_schema_chars: int = None) -> str:
    schema = truncate_schema(item.get("db_schema", ""), max_chars=max_schema_chars or MAX_SCHEMA_CHARS)
    return (
        f"[Database Schema]\n{schema}\n\n"
        f"[User Question]\n{item['question']}"
    )


def build_sharegpt_format(data: list) -> list:
    conversations = []
    for item in data:
        user_msg = build_user_prompt(item, max_schema_chars=MAX_SCHEMA_CHARS)
        conv = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": item["gold_sql"]},
            ]
        }
        conversations.append(conv)
    return conversations


def build_test_dataset(data: list) -> list:
    test_items = []
    for item in data:
        test_items.append({
            "id": item["id"],
            "db_id": item["db_id"],
            "question": item["question"],
            "db_schema": item["db_schema"],
            "gold_sql": item["gold_sql"],
            "input_prompt": build_user_prompt(item, max_schema_chars=TEST_SCHEMA_CHARS),
        })
    return test_items


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "train_for_cot.json"
    if not train_path.exists():
        print(f"[ERROR] {train_path} not found")
        return

    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded train: {len(data)} samples")

    sharegpt = build_sharegpt_format(data)

    sg_path = OUTPUT_DIR / "train_sharegpt_nocot.json"
    with open(sg_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt, f, indent=2, ensure_ascii=False)
    print(f"  ShareGPT (json) -> {sg_path}  ({len(sharegpt)} samples)")

    test_cot_path = PROCESSED_DIR / "test_for_cot.json"
    if test_cot_path.exists():
        with open(test_cot_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        test_set = build_test_dataset(test_data)
        out_path = OUTPUT_DIR / "test_eval_nocot.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2, ensure_ascii=False)
        print(f"  Test eval set -> {out_path}  ({len(test_set)} samples)")

    print("\nNo-CoT SFT dataset construction complete!")


if __name__ == "__main__":
    main()
