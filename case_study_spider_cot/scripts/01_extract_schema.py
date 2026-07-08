"""
从Spider数据集的sqlite数据库中提取schema信息，
转换为DeepVis风格的结构化schema表达（table_name(column_name:type, ...)）。
同时提取外键关系和采样值。
"""

import sqlite3
import json
import os
from pathlib import Path

SPIDER_DIR = Path(__file__).resolve().parent.parent / "spider_data"
DATABASE_DIR = SPIDER_DIR / "database"
TEST_DATABASE_DIR = SPIDER_DIR / "test_database"
TABLES_JSON = SPIDER_DIR / "tables.json"
TEST_TABLES_JSON = SPIDER_DIR / "test_tables.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "processed_data" / "db_schemas.json"

SQLITE_TYPE_MAP = {
    "INTEGER": "number",
    "INT": "number",
    "SMALLINT": "number",
    "BIGINT": "number",
    "TINYINT": "number",
    "MEDIUMINT": "number",
    "FLOAT": "number",
    "DOUBLE": "number",
    "REAL": "number",
    "NUMERIC": "number",
    "DECIMAL": "number",
    "TEXT": "text",
    "VARCHAR": "text",
    "CHAR": "text",
    "NCHAR": "text",
    "NVARCHAR": "text",
    "CLOB": "text",
    "BOOLEAN": "boolean",
    "BOOL": "boolean",
    "DATE": "date",
    "DATETIME": "date",
    "TIMESTAMP": "date",
    "TIME": "time",
    "BLOB": "blob",
}


def normalize_type(raw_type: str) -> str:
    if not raw_type:
        return "text"
    upper = raw_type.upper().split("(")[0].strip()
    return SQLITE_TYPE_MAP.get(upper, "text")


def extract_schema_from_sqlite(db_path: str) -> dict:
    """直接从sqlite文件提取schema信息。"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_info = {"tables": {}, "foreign_keys": [], "primary_keys": {}}

    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}");')
        columns = cursor.fetchall()
        col_list = []
        pk_cols = []
        for col in columns:
            cid, name, col_type, notnull, default, pk = col
            col_list.append({"name": name, "type": normalize_type(col_type)})
            if pk:
                pk_cols.append(name)
        schema_info["tables"][table] = col_list
        if pk_cols:
            schema_info["primary_keys"][table] = pk_cols

        cursor.execute(f'PRAGMA foreign_key_list("{table}");')
        fks = cursor.fetchall()
        for fk in fks:
            _id, _seq, ref_table, from_col, to_col, *_ = fk
            schema_info["foreign_keys"].append({
                "from_table": table,
                "from_column": from_col,
                "to_table": ref_table,
                "to_column": to_col,
            })

    conn.close()
    return schema_info


def extract_schema_from_tables_json(tables_json_path: str) -> dict:
    """从tables.json提取schema信息作为备用/补充。"""
    with open(tables_json_path, "r", encoding="utf-8") as f:
        all_tables = json.load(f)

    schemas = {}
    for db in all_tables:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        col_names = db["column_names_original"]
        col_types = db["column_types"]
        primary_keys = db.get("primary_keys", [])
        foreign_keys = db.get("foreign_keys", [])

        tables = {}
        for i, tname in enumerate(table_names):
            tables[tname] = []

        for idx, (table_idx, col_name) in enumerate(col_names):
            if table_idx == -1:
                continue
            tname = table_names[table_idx]
            ctype = col_types[idx] if idx < len(col_types) else "text"
            tables[tname].append({"name": col_name, "type": normalize_type(ctype)})

        pk_map = {}
        for pk_idx in primary_keys:
            if isinstance(pk_idx, int) and pk_idx < len(col_names):
                table_idx, col_name = col_names[pk_idx]
                if table_idx >= 0:
                    tname = table_names[table_idx]
                    pk_map.setdefault(tname, []).append(col_name)

        fk_list = []
        for fk_pair in foreign_keys:
            if len(fk_pair) == 2:
                from_idx, to_idx = fk_pair
                if from_idx < len(col_names) and to_idx < len(col_names):
                    from_table_idx, from_col = col_names[from_idx]
                    to_table_idx, to_col = col_names[to_idx]
                    if from_table_idx >= 0 and to_table_idx >= 0:
                        fk_list.append({
                            "from_table": table_names[from_table_idx],
                            "from_column": from_col,
                            "to_table": table_names[to_table_idx],
                            "to_column": to_col,
                        })

        schemas[db_id] = {
            "tables": tables,
            "foreign_keys": fk_list,
            "primary_keys": pk_map,
        }

    return schemas


def sample_values(db_path: str, table: str, column: str, limit: int = 3) -> list:
    """从数据库中采样若干个不同的值。"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit};'
        )
        values = [str(row[0]) for row in cursor.fetchall()]
        conn.close()
        return values
    except Exception:
        return []


def format_schema_deepvis(schema_info: dict, db_path: str = None) -> str:
    """
    将schema转成DeepVis风格的文本表达：
    Table: table_name
    Columns: col1 (type), col2 (type), ...
    Primary Key: col
    Sample Values: col1 = [v1, v2, v3]
    """
    lines = []
    for table_name, columns in schema_info["tables"].items():
        col_strs = [f"{c['name']}:{c['type']}" for c in columns]
        lines.append(f"Table: {table_name}")
        lines.append(f"  Columns: {', '.join(col_strs)}")

        pk = schema_info["primary_keys"].get(table_name, [])
        if pk:
            lines.append(f"  Primary Key: {', '.join(pk)}")

        if db_path and os.path.exists(db_path):
            for c in columns:
                vals = sample_values(db_path, table_name, c["name"])
                if vals:
                    lines.append(f"  Sample({c['name']}): [{', '.join(vals)}]")

    if schema_info["foreign_keys"]:
        lines.append("Foreign Keys:")
        for fk in schema_info["foreign_keys"]:
            lines.append(
                f"  {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
            )

    return "\n".join(lines)


def process_db_dirs(db_base_dir: Path, json_schemas: dict, all_schemas: dict):
    """扫描一个数据库目录，提取 schema 并写入 all_schemas。"""
    if not db_base_dir.exists():
        return
    db_dirs = sorted(db_base_dir.iterdir())
    for db_dir in db_dirs:
        if not db_dir.is_dir():
            continue
        db_id = db_dir.name
        if db_id in all_schemas:
            continue
        sqlite_file = db_dir / f"{db_id}.sqlite"

        if sqlite_file.exists():
            schema_info = extract_schema_from_sqlite(str(sqlite_file))
        elif db_id in json_schemas:
            schema_info = json_schemas[db_id]
        else:
            print(f"[WARN] No schema found for {db_id}")
            continue

        db_path = str(sqlite_file) if sqlite_file.exists() else None
        schema_text = format_schema_deepvis(schema_info, db_path)

        all_schemas[db_id] = {
            "schema_info": schema_info,
            "schema_text": schema_text,
        }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    json_schemas = extract_schema_from_tables_json(str(TABLES_JSON))
    if TEST_TABLES_JSON.exists():
        test_json_schemas = extract_schema_from_tables_json(str(TEST_TABLES_JSON))
        json_schemas.update(test_json_schemas)
        print(f"Also loaded test_tables.json ({len(test_json_schemas)} databases)")

    all_schemas = {}

    process_db_dirs(DATABASE_DIR, json_schemas, all_schemas)
    print(f"After scanning database/: {len(all_schemas)} databases")

    process_db_dirs(TEST_DATABASE_DIR, json_schemas, all_schemas)
    print(f"After scanning test_database/: {len(all_schemas)} databases")

    for db_id, schema_info in json_schemas.items():
        if db_id not in all_schemas:
            schema_text = format_schema_deepvis(schema_info)
            all_schemas[db_id] = {
                "schema_info": schema_info,
                "schema_text": schema_text,
            }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_schemas, f, indent=2, ensure_ascii=False)

    print(f"Extracted schemas for {len(all_schemas)} databases -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
