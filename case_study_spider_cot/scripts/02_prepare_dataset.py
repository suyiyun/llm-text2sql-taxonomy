"""
处理Spider数据集，关联schema，使用官方划分：
  - 训练集：train_spider.json + train_others.json（Spider官方训练集）
  - 测试集：test.json（Spider官方测试集，2147条）
不做任何自行拆分，严格按照官方划分。
"""

import json
from pathlib import Path

SPIDER_DIR = Path(__file__).resolve().parent.parent / "spider_data"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "processed_data" / "db_schemas.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "processed_data"


def load_spider_train():
    """加载Spider官方训练集（train_spider.json + train_others.json）。"""
    samples = []
    for name in ["train_spider.json", "train_others.json"]:
        fpath = SPIDER_DIR / name
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} samples from {name}")
            samples.extend(data)
    if not samples:
        raise FileNotFoundError("Cannot find train_spider.json or train_others.json")
    return samples


def load_spider_test():
    """加载Spider官方测试集（test.json，2147条）。"""
    fpath = SPIDER_DIR / "test.json"
    if not fpath.exists():
        raise FileNotFoundError(f"Cannot find {fpath}")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples from test.json")
    return data


def build_dataset(samples: list, schemas: dict, id_offset: int = 0) -> list:
    """为每条样本关联schema文本，构建统一格式。"""
    dataset = []
    missing_schema = set()

    for i, item in enumerate(samples):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item.get("query", "")

        if db_id not in schemas:
            missing_schema.add(db_id)
            continue

        schema_text = schemas[db_id]["schema_text"]

        dataset.append({
            "id": id_offset + i,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "db_schema": schema_text,
        })

    if missing_schema:
        print(f"  [WARN] Missing schemas for {len(missing_schema)} databases: {missing_schema}")

    return dataset


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schemas = json.load(f)
    print(f"Loaded schemas for {len(schemas)} databases")

    print("\n--- Loading Spider official TRAIN split ---")
    train_samples = load_spider_train()
    train_dataset = build_dataset(train_samples, schemas, id_offset=0)
    print(f"Train dataset: {len(train_dataset)} samples")

    print("\n--- Loading Spider official TEST split ---")
    test_samples = load_spider_test()
    test_dataset = build_dataset(test_samples, schemas, id_offset=len(train_dataset))
    print(f"Test dataset: {len(test_dataset)} samples")

    train_path = OUTPUT_DIR / "train_for_cot.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved train: {len(train_dataset)} samples -> {train_path}")

    test_path = OUTPUT_DIR / "test_for_cot.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved test:  {len(test_dataset)} samples -> {test_path}")

    print(f"\n=== Summary ===")
    print(f"  Train (official train_spider + train_others): {len(train_dataset)}")
    print(f"  Test  (official test.json):                   {len(test_dataset)}")
    print(f"  Total:                                        {len(train_dataset) + len(test_dataset)}")


if __name__ == "__main__":
    main()
