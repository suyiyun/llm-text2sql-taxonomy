"""
把 BIRD dev 集转成与 Spider 侧一致的统一格式，供同一套 L1/L2/L3 pipeline 复用
（见 docs/exp_design_l1l2l3_controlled_backbone.md 第 5.2 节）。

关键处理:
  1. Schema 提取: 直接从每个数据库的 .sqlite 文件读取表/列/类型/主键/外键
     （BIRD 的 sqlite 文件里外键约束是完整定义的，做法与 Spider 侧的
     01_extract_schema.py 一致），并合并 database_description/*.csv 里的列
     描述和取值说明 -- 这是 BIRD 相对 Spider 的关键差异（更丰富的语义信息）。
  2. Evidence 折叠: 把每条样本的 evidence 字段拼进 question 文本里
     （"[Hint] ..."），这样下游所有 prompt builder（L1/L2/L3 的 planner/coder/
     reviewer）不需要单独改代码就能统一用上 external knowledge，三个等级的
     处理方式完全一致。
  3. Few-shot 留出: 从 3 个不同数据库各挑 1 条样本作为固定 few-shot 示例，
     从评测池中剔除，避免用同一条数据既做演示又做评测。
  4. 难度标签: 直接使用 BIRD 官方的 simple/moderate/challenging 字段，不用
     Spider 那套规则分类器。

用法:
  python 00b_prepare_bird_dataset.py
"""

import argparse
import csv
import random
import sqlite3
from pathlib import Path

from common import (
    BIRD_DATA_DIR,
    BIRD_DB_DIR,
    BIRD_DEV_JSON,
    BIRD_PROCESSED_DIR,
    load_json,
    save_json,
)

SQLITE_TYPE_MAP = {
    "INTEGER": "number", "INT": "number", "SMALLINT": "number", "BIGINT": "number",
    "TINYINT": "number", "MEDIUMINT": "number", "FLOAT": "number", "DOUBLE": "number",
    "REAL": "number", "NUMERIC": "number", "DECIMAL": "number",
    "TEXT": "text", "VARCHAR": "text", "CHAR": "text", "NCHAR": "text",
    "NVARCHAR": "text", "CLOB": "text",
    "BOOLEAN": "boolean", "BOOL": "boolean",
    "DATE": "date", "DATETIME": "date", "TIMESTAMP": "date", "TIME": "time",
    "BLOB": "blob",
}

MAX_DESC_CHARS = 200  # cap per-field description length to avoid schema bloat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", type=str, default=str(BIRD_DB_DIR))
    parser.add_argument("--dev_json", type=str, default=str(BIRD_DEV_JSON))
    parser.add_argument("--output_dir", type=str, default=str(BIRD_PROCESSED_DIR))
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_type(raw_type: str) -> str:
    if not raw_type:
        return "text"
    upper = raw_type.upper().split("(")[0].strip()
    return SQLITE_TYPE_MAP.get(upper, "text")


def load_column_descriptions(db_dir: Path) -> dict:
    """Returns {table_name: {original_column_name: {description, value_description}}}."""
    desc_dir = db_dir / "database_description"
    result = {}
    if not desc_dir.exists():
        return result
    for csv_path in desc_dir.glob("*.csv"):
        table_name = csv_path.stem
        col_map = {}
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                with open(csv_path, "r", encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        orig = (row.get("original_column_name") or "").strip()
                        if not orig:
                            continue
                        desc = (row.get("column_description") or "").strip().replace("\n", " ")
                        val_desc = (row.get("value_description") or "").strip().replace("\n", " ")
                        col_map[orig] = {
                            "description": desc[:MAX_DESC_CHARS],
                            "value_description": val_desc[:MAX_DESC_CHARS],
                        }
                break
            except UnicodeDecodeError:
                continue
        result[table_name] = col_map
    return result


def extract_schema(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_info = {"tables": {}, "foreign_keys": [], "primary_keys": {}}
    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}");')
        columns = cursor.fetchall()
        col_list, pk_cols = [], []
        for cid, name, col_type, notnull, default, pk in columns:
            col_list.append({"name": name, "type": normalize_type(col_type)})
            if pk:
                pk_cols.append(name)
        schema_info["tables"][table] = col_list
        if pk_cols:
            schema_info["primary_keys"][table] = pk_cols

        cursor.execute(f'PRAGMA foreign_key_list("{table}");')
        for _id, _seq, ref_table, from_col, to_col, *_ in cursor.fetchall():
            schema_info["foreign_keys"].append({
                "from_table": table, "from_column": from_col,
                "to_table": ref_table, "to_column": to_col,
            })
    conn.close()
    return schema_info


def sample_values(db_path: Path, table: str, column: str, limit: int = 3) -> list:
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit};')
        values = [str(row[0]) for row in cursor.fetchall()]
        conn.close()
        return values
    except Exception:
        return []


def format_schema_text(schema_info: dict, descriptions: dict, db_path: Path) -> str:
    lines = []
    for table_name, columns in schema_info["tables"].items():
        col_strs = [f"{c['name']}:{c['type']}" for c in columns]
        lines.append(f"Table: {table_name}")
        lines.append(f"  Columns: {', '.join(col_strs)}")

        pk = schema_info["primary_keys"].get(table_name, [])
        if pk:
            lines.append(f"  Primary Key: {', '.join(pk)}")

        table_desc = descriptions.get(table_name, {})
        for c in columns:
            col_desc = table_desc.get(c["name"])
            if col_desc and (col_desc["description"] or col_desc["value_description"]):
                parts = []
                if col_desc["description"] and col_desc["description"].lower() != c["name"].lower():
                    parts.append(col_desc["description"])
                if col_desc["value_description"]:
                    parts.append(f"values: {col_desc['value_description']}")
                if parts:
                    lines.append(f"  Description({c['name']}): {' | '.join(parts)}")

            vals = sample_values(db_path, table_name, c["name"])
            if vals:
                lines.append(f"  Sample({c['name']}): [{', '.join(vals)}]")

    if schema_info["foreign_keys"]:
        lines.append("Foreign Keys:")
        for fk in schema_info["foreign_keys"]:
            lines.append(f"  {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}")

    return "\n".join(lines)


def main():
    args = parse_args()
    db_dir = Path(args.db_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_ids = sorted(p.name for p in db_dir.iterdir() if p.is_dir())
    print(f"Found {len(db_ids)} BIRD databases: {db_ids}")

    schemas = {}
    for db_id in db_ids:
        sqlite_path = db_dir / db_id / f"{db_id}.sqlite"
        if not sqlite_path.exists():
            print(f"[WARN] no sqlite file for {db_id}, skipping")
            continue
        schema_info = extract_schema(sqlite_path)
        descriptions = load_column_descriptions(db_dir / db_id)
        schemas[db_id] = format_schema_text(schema_info, descriptions, sqlite_path)
        print(f"  {db_id}: {len(schema_info['tables'])} tables, {sum(len(v) for v in schema_info['tables'].values())} columns")

    dev_data = load_json(args.dev_json)
    print(f"\nLoaded {len(dev_data)} BIRD dev items")

    random.seed(args.seed)
    by_db = {}
    for item in dev_data:
        by_db.setdefault(item["db_id"], []).append(item)

    fewshot_dbs = random.sample(sorted(by_db.keys()), min(args.num_shots, len(by_db)))
    fewshot_items = [random.choice(by_db[db_id]) for db_id in fewshot_dbs]
    fewshot_ids = {item["question_id"] for item in fewshot_items}
    print(f"Held out {len(fewshot_items)} few-shot examples from dbs: {fewshot_dbs}")

    def build_item(raw):
        db_id = raw["db_id"]
        question = raw["question"].strip()
        evidence = (raw.get("evidence") or "").strip()
        question_with_hint = f"{question}\n\n[Hint] {evidence}" if evidence else question
        return {
            "id": raw["question_id"],
            "db_id": db_id,
            "question": question_with_hint,
            "question_raw": question,
            "evidence": evidence,
            "gold_sql": raw["SQL"],
            "db_schema": schemas.get(db_id, ""),
            "bird_difficulty": raw["difficulty"],
        }

    fewshot_out = [build_item(item) for item in fewshot_items]
    test_out = [build_item(item) for item in dev_data if item["question_id"] not in fewshot_ids]

    save_json(fewshot_out, output_dir / "bird_fewshot.json")
    save_json(test_out, output_dir / "bird_test.json")
    print(f"\nSaved {len(fewshot_out)} few-shot examples -> {output_dir / 'bird_fewshot.json'}")
    print(f"Saved {len(test_out)} test items -> {output_dir / 'bird_test.json'}")


if __name__ == "__main__":
    main()
