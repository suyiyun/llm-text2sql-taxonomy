"""
L2 -- Iterative Refinement (execution-feedback self-correction). 见 docs/
exp_design_l1l2l3_controlled_backbone.md 第 4 节:

  Y_0 = generate(I, Q, S)
  for t in range(MAX_ITER):
      result, err = execute(Y_t, db)
      if err is None: break
      Y_{t+1} = generate(I, Q, S, Y_t, err)
  final_Y = Y_t

只处理执行报错这一类反馈，不引入语义判断（语义判断留给 L3 的 Reviewer）。

用法:
  python 03_run_l2_iterative_refine.py --provider deepseek --api_key YOUR_KEY --workers 5
  python 03_run_l2_iterative_refine.py --dataset bird --provider deepseek --workers 5
"""

import argparse
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
    build_refinement_prompt,
    call_llm,
    extract_sql,
    execute_sql,
    find_db_path,
    evaluate_one,
    load_json,
    save_json,
    select_few_shot_examples,
)

MAX_ITER = 3


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
    parser.add_argument("--max_iter", type=int, default=MAX_ITER)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 条，用于 smoke test")
    return parser.parse_args()


def process_one(item, examples, provider, api_key, max_tokens, db_dir, max_iter, model=None, base_url=None):
    db_dir = Path(db_dir)
    db_path = find_db_path(item["db_id"], db_dir)

    prompt = build_few_shot_prompt(examples, item["question"], item["db_schema"])
    result = call_llm(provider, api_key, BASE_SYSTEM_PROMPT, prompt, max_tokens=max_tokens, model=model, base_url=base_url)

    calls = [result]
    pred_sql = extract_sql(result.text)
    iterations_used = 0
    last_error = None

    for t in range(max_iter):
        if pred_sql.startswith("ERROR:"):
            # LLM call itself failed (network/auth) -- nothing to execute, stop refining.
            break
        if db_path is None:
            break
        _, err = execute_sql(str(db_path), pred_sql)
        if err is None:
            break
        last_error = err
        iterations_used = t + 1
        refine_prompt = build_refinement_prompt(item["question"], item["db_schema"], pred_sql, err)
        result = call_llm(provider, api_key, BASE_SYSTEM_PROMPT, refine_prompt, max_tokens=max_tokens, model=model, base_url=base_url)
        calls.append(result)
        pred_sql = extract_sql(result.text)

    ex, em = evaluate_one(pred_sql, item["gold_sql"], item["db_id"], db_dir)

    total_in = sum(c.input_tokens for c in calls if c.input_tokens is not None) or None
    total_out = sum(c.output_tokens for c in calls if c.output_tokens is not None) or None
    total_latency = sum(c.latency_s for c in calls if c.latency_s is not None)

    return {
        "id": item["id"],
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["gold_sql"],
        "spider_difficulty": item.get("spider_difficulty"),
        "bird_difficulty": item.get("bird_difficulty"),
        "pred_sql": pred_sql,
        "final_response": calls[-1].text,
        "execution_accuracy": ex,
        "exact_match": em,
        "num_llm_calls": len(calls),
        "iterations_used": iterations_used,
        "last_execution_error": last_error,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "latency_s": round(total_latency, 3),
    }


def main():
    args = parse_args()
    api_key = resolve_api_key(args.provider, args.api_key)
    defaults = resolve_dataset_defaults(args.dataset)
    subset_file = args.subset_file or str(defaults["subset_file"])
    db_dir = args.db_dir or str(defaults["db_dir"])
    train_file = args.train_file or str(defaults["train_file"])
    output_file = args.output_file or str(ABLATION_DIR / "outcome" / f"l2_{args.provider}_{args.dataset}.json")

    subset = load_json(subset_file)
    items = subset["items"]
    if args.limit:
        items = items[: args.limit]
    resolved_model = args.model or PROVIDER_CONFIGS[args.provider]["model"]
    print(f"Loaded {len(items)} subset items (level=L2, dataset={args.dataset}, provider={args.provider}, model={resolved_model}, max_iter={args.max_iter})")

    train_data = load_json(train_file)
    examples = select_few_shot_examples(train_data, args.num_shots, args.seed)
    print(f"Selected {len(examples)} few-shot examples from dbs: {[e['db_id'] for e in examples]}")

    results = []
    ex_correct = 0
    em_correct = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, item, examples, args.provider, api_key, args.max_tokens, db_dir, args.max_iter, args.model, args.base_url): item
            for item in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="L2"):
            r = future.result()
            results.append(r)
            ex_correct += int(r["execution_accuracy"])
            em_correct += int(r["exact_match"])

    total = len(results)
    avg_calls = sum(r["num_llm_calls"] for r in results) / total
    print(f"\nL2 results: EX={ex_correct}/{total}={ex_correct/total*100:.2f}%  EM={em_correct}/{total}={em_correct/total*100:.2f}%  avg_calls={avg_calls:.2f}")

    save_json({
        "level": "L2",
        "dataset": args.dataset,
        "provider": args.provider,
        "model": resolved_model,
        "num_shots": args.num_shots,
        "max_iter": args.max_iter,
        "metrics": {
            "execution_accuracy": ex_correct / total,
            "exact_match": em_correct / total,
            "total": total,
            "avg_llm_calls": avg_calls,
        },
        "results": results,
    }, output_file)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
