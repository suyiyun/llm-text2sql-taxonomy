"""
用主流大模型 API (DeepSeek V3, GPT-5, Claude, Gemini) 进行 3-shot Text-to-SQL 评估。
从训练集随机选 3 条作为 few-shot 示例，在测试集上直接让模型输出 SQL。

用法:
  python 07_baseline_api_eval.py \
    --provider deepseek \
    --api_key YOUR_KEY \
    --test_file ../sft_data/test_eval.json \
    --train_file ../processed_data/train_for_cot.json \
    --db_dir ../spider_data/database \
    --output_file ../outcome/eval_results_deepseek_3shot.json \
    --workers 5

支持的 provider: deepseek, openai, claude, gemini
"""

import json
import re
import sqlite3
import time
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPIDER_DB_DIR = PROJECT_DIR / "spider_data" / "database"

PROVIDER_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "api_style": "openai",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5",
        "api_style": "openai",
    },
    "claude": {
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-20250514",
        "api_style": "anthropic",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",
        "api_style": "gemini",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4-flash-250414",
        "api_style": "openai_raw",
        "request_delay": 5.0,
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1/",
        "model": "moonshot-v1-32k",
        "api_style": "openai_raw",
        "request_delay": 1.2,
    },
}

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural language question, "
    "generate the correct SQL query. Output ONLY the SQL query, nothing else."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, required=True,
                        choices=["deepseek", "openai", "claude", "gemini", "glm", "kimi"])
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="Override default model name")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Override default base URL")
    parser.add_argument("--test_file", type=str,
                        default=str(PROJECT_DIR / "sft_data" / "test_eval.json"))
    parser.add_argument("--train_file", type=str,
                        default=str(PROJECT_DIR / "processed_data" / "train_for_cot.json"))
    parser.add_argument("--db_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def select_few_shot_examples(train_data: list, num_shots: int, seed: int) -> list:
    """从训练集中选取多样化的 few-shot 示例（不同数据库、不同难度）。"""
    random.seed(seed)
    by_db = {}
    for item in train_data:
        by_db.setdefault(item["db_id"], []).append(item)

    db_ids = list(by_db.keys())
    random.shuffle(db_ids)

    examples = []
    for db_id in db_ids:
        if len(examples) >= num_shots:
            break
        candidates = by_db[db_id]
        examples.append(random.choice(candidates))

    return examples


def build_few_shot_prompt(examples: list, test_item: dict) -> str:
    """构建 3-shot prompt。"""
    parts = []

    for i, ex in enumerate(examples, 1):
        parts.append(
            f"### Example {i}\n"
            f"[Database Schema]\n{ex['db_schema']}\n\n"
            f"[Question]\n{ex['question']}\n\n"
            f"[SQL]\n{ex['gold_sql']}"
        )

    parts.append(
        f"### Your Task\n"
        f"[Database Schema]\n{test_item['db_schema']}\n\n"
        f"[Question]\n{test_item['question']}\n\n"
        f"[SQL]"
    )

    return "\n\n".join(parts)


def truncate_schema_in_prompt(prompt: str, max_chars: int = 15000) -> str:
    if len(prompt) <= max_chars:
        return prompt
    lines = prompt.split("\n")
    short = [l for l in lines if not l.strip().startswith("Sample(")]
    return "\n".join(short)[:max_chars]


def call_openai_style_api(base_url, api_key, model, prompt, max_tokens):
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    elif "/v1" in base_url:
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                return f"ERROR: {e}"


def call_anthropic_api(api_key, model, prompt, max_tokens):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                return f"ERROR: {e}"


def call_gemini_api(api_key, model, prompt, max_tokens):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                return f"ERROR: {e}"


def call_openai_raw_api(base_url, api_key, model, prompt, max_tokens, temperature=0.0):
    """base_url 已包含完整路径前缀（如 /v4/），直接拼 chat/completions。"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                if attempt < 4:
                    try:
                        err = resp.json()
                        msg = str(err).lower()
                        m = re.search(r"after\s+(\d+)\s*seconds?", msg)
                        wait = int(m.group(1)) + 2 if m else 5
                        wait = max(wait, 5)
                    except Exception:
                        wait = 10 + attempt * 15
                    wait = min(wait, 120)
                    time.sleep(wait)
                    continue
                try:
                    err_body = resp.json()
                    return f"ERROR: {resp.status_code} {resp.reason} {err_body}"
                except Exception:
                    return f"ERROR: {resp.status_code} {resp.reason}"
            if resp.status_code != 200:
                try:
                    err_body = resp.json()
                    return f"ERROR: {resp.status_code} {resp.reason} {err_body}"
                except Exception:
                    return f"ERROR: {resp.status_code} {resp.reason} {resp.text[:200]}"
            data = resp.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content")
            if content is None or (isinstance(content, str) and content.strip() == ""):
                return f"ERROR: empty content from API, raw_message={msg!r}"[:500]
            return str(content).strip()
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** (attempt + 1))
            else:
                return f"ERROR: {e}"


def call_api(provider, base_url, api_key, model, prompt, max_tokens):
    config = PROVIDER_CONFIGS[provider]
    api_style = config["api_style"]
    temperature = config.get("temperature", 0.0)

    if api_style == "openai":
        return call_openai_style_api(base_url, api_key, model, prompt, max_tokens)
    elif api_style == "openai_raw":
        return call_openai_raw_api(base_url, api_key, model, prompt, max_tokens, temperature)
    elif api_style == "anthropic":
        return call_anthropic_api(api_key, model, prompt, max_tokens)
    elif api_style == "gemini":
        return call_gemini_api(api_key, model, prompt, max_tokens)


def extract_sql(text: str) -> str:
    if text.startswith("ERROR:"):
        return text
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(SELECT.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.upper().startswith("SELECT"):
            return line
    return text.strip().split("\n")[0]


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


def evaluate_one(pred_sql, gold_sql, db_id, db_dir):
    db_path = find_db_path(db_id, db_dir)
    ex = False
    if db_path:
        pred_result = execute_sql(str(db_path), pred_sql)
        gold_result = execute_sql(str(db_path), gold_sql)
        if not isinstance(pred_result, str) and not isinstance(gold_result, str):
            ex = pred_result == gold_result
    em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
    return ex, em


def process_one_item(item, examples, provider, base_url, api_key, model, max_tokens, db_dir, request_delay=0):
    if request_delay > 0:
        time.sleep(request_delay)
    prompt = build_few_shot_prompt(examples, item)
    prompt = truncate_schema_in_prompt(prompt)

    response = call_api(provider, base_url, api_key, model, prompt, max_tokens)
    pred_sql = extract_sql(response)
    ex, em = evaluate_one(pred_sql, item["gold_sql"], item["db_id"], db_dir)

    return {
        "id": item["id"],
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["gold_sql"],
        "pred_sql": pred_sql,
        "full_response": response,
        "execution_accuracy": ex,
        "exact_match": em,
    }


def main():
    args = parse_args()
    config = PROVIDER_CONFIGS[args.provider]
    base_url = args.base_url or config["base_url"]
    model = args.model or config["model"]

    db_dir = Path(args.db_dir) if args.db_dir else SPIDER_DB_DIR
    if not db_dir.exists():
        for candidate in [Path("/root/spider_data/database"), Path("/root/spider/spider_data/database")]:
            if candidate.exists():
                db_dir = candidate
                break

    print(f"Provider: {args.provider} | Model: {model}")
    print(f"Database dir: {db_dir}")

    with open(args.train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    examples = select_few_shot_examples(train_data, args.num_shots, args.seed)
    print(f"Selected {len(examples)} few-shot examples from dbs: {[e['db_id'] for e in examples]}")

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")

    output_file = args.output_file or str(
        PROJECT_DIR / "outcome" / f"eval_results_{args.provider}_3shot.json"
    )

    done_ids = set()
    results = []
    if args.resume and Path(output_file).exists():
        with open(output_file, "r", encoding="utf-8") as f:
            old = json.load(f)
        results = old.get("results", [])
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} already done")

    remaining = [item for item in test_data if item["id"] not in done_ids]
    print(f"Remaining: {len(remaining)} samples")

    ex_correct = sum(1 for r in results if r["execution_accuracy"])
    em_correct = sum(1 for r in results if r["exact_match"])

    request_delay = config.get("request_delay", 0)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for item in remaining:
            future = executor.submit(
                process_one_item, item, examples,
                args.provider, base_url, args.api_key, model,
                args.max_tokens, db_dir, request_delay,
            )
            futures[future] = item

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.provider}"):
            result = future.result()
            results.append(result)

            if result["execution_accuracy"]:
                ex_correct += 1
            if result["exact_match"]:
                em_correct += 1

            done = len(results)
            if done % 50 == 0:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "provider": args.provider,
                        "model": model,
                        "num_shots": args.num_shots,
                        "metrics": {
                            "execution_accuracy": ex_correct / done,
                            "exact_match": em_correct / done,
                            "total": done,
                        },
                        "results": results,
                    }, f, indent=2, ensure_ascii=False)
                print(f"  [{done}/{len(test_data)}] EX={ex_correct/done*100:.1f}% EM={em_correct/done*100:.1f}%")

    total = len(results)
    print("\n" + "=" * 60)
    print(f"[{args.provider} / {model}] 3-shot Results:")
    print(f"Execution Accuracy (EX): {ex_correct}/{total} = {ex_correct/total*100:.2f}%")
    print(f"Exact Match (EM):        {em_correct}/{total} = {em_correct/total*100:.2f}%")
    print("=" * 60)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "provider": args.provider,
            "model": model,
            "num_shots": args.num_shots,
            "metrics": {
                "execution_accuracy": ex_correct / total,
                "exact_match": em_correct / total,
                "total": total,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
