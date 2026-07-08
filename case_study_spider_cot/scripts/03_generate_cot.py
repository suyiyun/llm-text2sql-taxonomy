"""
使用DeepSeek V3 API批量生成CoT（Chain of Thought）推理过程。
读取训练集样本，将(question, db_schema, gold_sql)填入prompt模板，
调用DeepSeek V3生成5步CoT推理，保存结果。

用法:
  python 03_generate_cot.py --api_key YOUR_DEEPSEEK_KEY --split train --workers 10
"""

import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROMPT_TEMPLATE_PATH = PROJECT_DIR.parent / "prompt.txt"
PROCESSED_DIR = PROJECT_DIR / "processed_data"

MAX_PROMPT_CHARS = 12000


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CoT using DeepSeek V3")
    parser.add_argument("--api_key", type=str, required=True, help="DeepSeek API key")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com",
                        help="API base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="Model name (deepseek-chat for V3)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Which split to generate CoT for")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    return parser.parse_args()


def load_prompt_template():
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"prompt.txt not found at {PROMPT_TEMPLATE_PATH}")
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def truncate_schema(schema_text: str, max_chars: int = 8000) -> str:
    if len(schema_text) <= max_chars:
        return schema_text
    lines = schema_text.split("\n")
    short_lines = [l for l in lines if not l.strip().startswith("Sample(")]
    result = "\n".join(short_lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (schema truncated)"
    return result


def fill_prompt(template: str, item: dict) -> str:
    schema = truncate_schema(item["db_schema"])
    return template.replace("{db_schema}", schema) \
                   .replace("{question}", item["question"]) \
                   .replace("{gold_sql}", item["gold_sql"])


def call_llm(prompt: str, args) -> str:
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n... (truncated)"

    url = f"{args.base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are an expert in database querying and SQL."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 400:
                err_msg = resp.text[:200]
                raise ValueError(f"400 Bad Request (prompt {len(prompt)} chars): {err_msg}")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except ValueError:
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def process_item(item: dict, template: str, args) -> dict:
    prompt = fill_prompt(template, item)
    try:
        cot = call_llm(prompt, args)
    except ValueError as e:
        print(f"  [SKIP] id={item['id']}: {e}")
        return None
    result = item.copy()
    result["cot"] = cot
    return result


def main():
    args = parse_args()

    template = load_prompt_template()
    print(f"Loaded prompt template ({len(template)} chars)")

    input_path = PROCESSED_DIR / f"{args.split}_for_cot.json"
    output_path = PROCESSED_DIR / f"{args.split}_with_cot.json"

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples from {input_path}")

    done_ids = set()
    results = []
    skipped_ids = set()
    skipped_path = PROCESSED_DIR / f"{args.split}_skipped_ids.json"
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        if skipped_path.exists():
            with open(skipped_path, "r") as f:
                skipped_ids = set(json.load(f))
        done_ids |= skipped_ids
        print(f"Resuming: {len(results)} done, {len(skipped_ids)} skipped")

    todo = [item for item in dataset if item["id"] not in done_ids]
    print(f"To process: {len(todo)} samples")

    if not todo:
        print("Nothing to do.")
        return

    failed = []
    skipped = []
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(process_item, item, template, args): item["id"]
        for item in todo
    }

    pbar = tqdm.tqdm(total=len(todo), desc="Generating CoT")
    for future in as_completed(futures):
        item_id = futures[future]
        try:
            result = future.result(timeout=300)
            if result is not None:
                results.append(result)
                if len(results) % 50 == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                skipped.append(item_id)
            pbar.update(1)
        except Exception as e:
            print(f"\n[FAIL] id={item_id}: {e}")
            failed.append(item_id)
            pbar.update(1)

    pbar.close()
    executor.shutdown()

    if skipped:
        all_skipped = list(skipped_ids | set(skipped))
        with open(skipped_path, "w") as f:
            json.dump(all_skipped, f)
        print(f"Skipped (400 etc): {len(skipped)} this run, total {len(all_skipped)}")

    results.sort(key=lambda x: x["id"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(results)} samples -> {output_path}")
    if failed:
        print(f"Failed: {len(failed)} items: {failed}")
        failed_path = PROCESSED_DIR / f"{args.split}_failed_ids.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f)


if __name__ == "__main__":
    main()
