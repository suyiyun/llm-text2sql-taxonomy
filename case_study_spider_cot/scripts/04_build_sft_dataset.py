"""
将CoT数据构建为Qwen3-8B SFT训练所需的对话格式（ShareGPT）。
同时生成测试集（不含CoT，仅question+schema作为输入）。

训练集schema做截断（避免超长token），测试集schema放宽。
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "processed_data"
OUTPUT_DIR = PROJECT_DIR / "sft_data"

MAX_SCHEMA_CHARS = 6000
MAX_COT_CHARS = 4000
TEST_SCHEMA_CHARS = 12000

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural language question, "
    "generate the correct SQL query using step-by-step Chain of Thought reasoning.\n\n"
    "Follow these steps:\n"
    "Step 1: Schema Linking - Identify relevant tables and columns.\n"
    "Step 2: Logic Decomposition - Break down the question into logical operations.\n"
    "Step 3: Draft SQL - Write an initial SQL query.\n"
    "Step 4: SQL Revision - Verify and refine the query.\n"
    "Step 5: Final SQL - Output the final SQL query."
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
            ]
        }
        if "cot" in item:
            cot = truncate_text(item["cot"], MAX_COT_CHARS)
            conv["conversations"].append({"from": "gpt", "value": cot})
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

    cot_path = PROCESSED_DIR / "train_with_cot.json"
    if cot_path.exists():
        with open(cot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded train: {len(data)} samples")

        sharegpt = build_sharegpt_format(data)

        sg_path = OUTPUT_DIR / "train_sharegpt.json"
        with open(sg_path, "w", encoding="utf-8") as f:
            json.dump(sharegpt, f, indent=2, ensure_ascii=False)
        print(f"  ShareGPT (json) -> {sg_path}")

        jsonl_path = OUTPUT_DIR / "train_sharegpt.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in sharegpt:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  ShareGPT (jsonl) -> {jsonl_path}")
    else:
        print(f"[SKIP] {cot_path} not found (run 03_generate_cot.py first)")

    test_path = PROCESSED_DIR / "test_for_cot.json"
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        test_set = build_test_dataset(test_data)
        out_path = OUTPUT_DIR / "test_eval.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2, ensure_ascii=False)
        print(f"Test eval set: {len(test_set)} samples -> {out_path}")

    print("\nSFT dataset construction complete!")


if __name__ == "__main__":
    main()
