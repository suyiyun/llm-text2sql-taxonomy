"""
L1 -- Single-Turn Generation. 一次 LLM 调用，输出即为最终 SQL，不做任何执行反馈
或多 agent 协作（见 docs/exp_design_l1l2l3_controlled_backbone.md 第 4 节）。

这是 L2/L3 共同的第 0 次调用，三个等级在同一个抽样子集上跑，backbone 也固定，
只有这一步的输出结构不同。

用法:
  python 02_run_l1_single_turn.py --provider deepseek --api_key YOUR_KEY --workers 5
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common import (
    ABLATION_DIR,
    BASE_SYSTEM_PROMPT,
    PROVIDER_CONFIGS,
    resolve_api_key,
    resolve_dataset_defaults,
    build_few_shot_prompt,
    call_llm,
    extract_sql,
    evaluate_one,
    load_json,
    save_json,
    select_few_shot_examples,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spider", choices=["spider", "bird"])
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "glm"])
    parser.add_argument("--api_key", type=str, default=None,
                        help="Falls back to <PROVIDER>_API_KEY env var (e.g. DEEPSEEK_API_KEY) if omitted")
    parser.add_argument("--model", type=str, default=None, help="Override the provider's default model id")
    parser.add_argument("--base_url", type=str, default=None, help="Override the provider's default base URL")
    parser.add_argument("--subset_file", type=str, default=None, help="Default depends on --dataset")
    parser.add_argument("--train_file", type=str, default=None, help="Few-shot pool; default depends on --dataset")
    parser.add_argument("--db_dir", type=str, default=None, help="Default depends on --dataset")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 条，用于 smoke test")
    return parser.parse_args()


def process_one(item, examples, provider, api_key, max_tokens, db_dir, model=None, base_url=None):
    prompt = build_few_shot_prompt(examples, item["question"], item["db_schema"])

    result = call_llm(provider, api_key, BASE_SYSTEM_PROMPT, prompt, max_tokens=max_tokens, model=model, base_url=base_url)
    pred_sql = extract_sql(result.text)
    ex, em = evaluate_one(pred_sql, item["gold_sql"], item["db_id"], Path(db_dir))

    return {
        "id": item["id"],
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["gold_sql"],
        "spider_difficulty": item.get("spider_difficulty"),
        "bird_difficulty": item.get("bird_difficulty"),
        "pred_sql": pred_sql,
        "full_response": result.text,
        "execution_accuracy": ex,
        "exact_match": em,
        "num_llm_calls": 1,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": round(result.latency_s, 3) if result.latency_s is not None else None,
    }


def main():
    args = parse_args()
    api_key = resolve_api_key(args.provider, args.api_key)
    defaults = resolve_dataset_defaults(args.dataset)
    subset_file = args.subset_file or str(defaults["subset_file"])
    db_dir = args.db_dir or str(defaults["db_dir"])
    train_file = args.train_file or str(defaults["train_file"])
    output_file = args.output_file or str(ABLATION_DIR / "outcome" / f"l1_{args.provider}_{args.dataset}.json")

    subset = load_json(subset_file)
    items = subset["items"]
    if args.limit:
        items = items[: args.limit]
    resolved_model = args.model or PROVIDER_CONFIGS[args.provider]["model"]
    print(f"Loaded {len(items)} subset items (level=L1, dataset={args.dataset}, provider={args.provider}, model={resolved_model})")

    train_data = load_json(train_file)
    examples = select_few_shot_examples(train_data, args.num_shots, args.seed)
    print(f"Selected {len(examples)} few-shot examples from dbs: {[e['db_id'] for e in examples]}")

    results = []
    ex_correct = 0
    em_correct = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, item, examples, args.provider, api_key, args.max_tokens, db_dir, args.model, args.base_url): item
            for item in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="L1"):
            r = future.result()
            results.append(r)
            ex_correct += int(r["execution_accuracy"])
            em_correct += int(r["exact_match"])

    total = len(results)
    print(f"\nL1 results: EX={ex_correct}/{total}={ex_correct/total*100:.2f}%  EM={em_correct}/{total}={em_correct/total*100:.2f}%")

    save_json({
        "level": "L1",
        "dataset": args.dataset,
        "provider": args.provider,
        "model": resolved_model,
        "num_shots": args.num_shots,
        "metrics": {"execution_accuracy": ex_correct / total, "exact_match": em_correct / total, "total": total},
        "results": results,
    }, output_file)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
