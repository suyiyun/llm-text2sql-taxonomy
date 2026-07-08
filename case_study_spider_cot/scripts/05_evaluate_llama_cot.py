"""
评估 LLaMA3.1-8B-Instruct CoT 模型在 Spider 上的 Text-to-SQL 性能。
"""

import json
import re
import sqlite3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPIDER_DB_DIR = PROJECT_DIR / "spider_data" / "database"

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

DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/meta-llama/Meta-Llama-3.1-8B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str,
                        default=str(PROJECT_DIR / "sft_data" / "test_eval.json"))
    parser.add_argument("--db_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def extract_sql_from_cot(text: str) -> str:
    patterns = [
        r"Step\s*5.*?```sql\s*(.*?)```",
        r"```sql\s*(.*?)```",
        r"(?:Final SQL|final sql)[:\s]*(SELECT.*?)(?:\n\n|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.upper().startswith("SELECT"):
            return line
    return text.strip().split("\n")[-1]


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def execute_sql(db_path: str, sql: str):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return sorted([tuple(str(v) for v in row) for row in results])
    except Exception as e:
        return f"ERROR: {e}"


def find_db_path(db_id: str, db_dir: Path) -> Path:
    for base in [db_dir, db_dir.parent / "test_database"]:
        candidate = base / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            return candidate
        if (base / db_id).exists():
            sqlite_files = list((base / db_id).glob("*.sqlite"))
            if sqlite_files:
                return sqlite_files[0]
    return None


def evaluate_execution_accuracy(pred_sql, gold_sql, db_id, db_dir):
    db_path = find_db_path(db_id, db_dir)
    if db_path is None:
        return False
    pred_result = execute_sql(str(db_path), pred_sql)
    gold_result = execute_sql(str(db_path), gold_sql)
    if isinstance(pred_result, str) or isinstance(gold_result, str):
        return False
    return pred_result == gold_result


def evaluate_exact_match(pred_sql, gold_sql):
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


def load_model(model_path: str, base_model: str = None, use_4bit: bool = True):
    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    adapter_config_file = model_path / "adapter_config.json"
    if adapter_config_file.exists():
        with open(adapter_config_file, "r") as f:
            adapter_config = json.load(f)
        base_path = base_model or adapter_config.get("base_model_name_or_path") or DEFAULT_BASE_MODEL
        if not Path(base_path).exists():
            base_path = DEFAULT_BASE_MODEL
        print(f"Loading base model from {base_path} (4-bit={use_4bit}), then adapter from {model_path}...")
        kwargs = dict(device_map="auto", trust_remote_code=True)
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
        model = PeftModel.from_pretrained(model, str(model_path), is_trainable=False)
    else:
        print(f"Loading merged model from {model_path} (4-bit={use_4bit})...")
        kwargs = dict(device_map="auto", trust_remote_code=True)
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    args = parse_args()

    db_dir = Path(args.db_dir) if args.db_dir else SPIDER_DB_DIR
    if not db_dir.exists():
        for candidate in [Path("/root/spider_data/database"), Path("/root/spider/spider_data/database")]:
            if candidate.exists():
                db_dir = candidate
                break
    print(f"Database dir: {db_dir}")

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")

    model, tokenizer = load_model(args.model_path, args.base_model, use_4bit=not args.no_4bit)

    results = []
    ex_correct = 0
    em_correct = 0
    total = len(test_data)

    for i, item in enumerate(test_data):
        print(f"\n[{i+1}/{total}] db={item['db_id']}")

        response = generate_response(model, tokenizer, item["input_prompt"], args.max_new_tokens)
        pred_sql = extract_sql_from_cot(response)

        ex = evaluate_execution_accuracy(pred_sql, item["gold_sql"], item["db_id"], db_dir)
        em = evaluate_exact_match(pred_sql, item["gold_sql"])

        if ex:
            ex_correct += 1
        if em:
            em_correct += 1

        results.append({
            "id": item["id"],
            "db_id": item["db_id"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
            "pred_sql": pred_sql,
            "full_response": response,
            "execution_accuracy": ex,
            "exact_match": em,
        })

        print(f"  EX={ex}, EM={em} | Pred: {pred_sql[:80]}...")

        if (i + 1) % 50 == 0:
            interim_file = args.output_file or str(PROJECT_DIR / "outcome" / "eval_results_llama_cot.json")
            Path(interim_file).parent.mkdir(parents=True, exist_ok=True)
            with open(interim_file, "w", encoding="utf-8") as f:
                json.dump({"metrics": {"execution_accuracy": ex_correct / (i+1), "exact_match": em_correct / (i+1), "total": i+1}, "results": results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Execution Accuracy (EX): {ex_correct}/{total} = {ex_correct/total*100:.2f}%")
    print(f"Exact Match (EM):        {em_correct}/{total} = {em_correct/total*100:.2f}%")
    print("=" * 60)

    output_file = args.output_file or str(PROJECT_DIR / "outcome" / "eval_results_llama_cot.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {"execution_accuracy": ex_correct / total, "exact_match": em_correct / total, "total": total},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
