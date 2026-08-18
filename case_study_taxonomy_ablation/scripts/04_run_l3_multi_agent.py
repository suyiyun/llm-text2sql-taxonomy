"""
L3 -- Multi-Agent Collaboration (minimal Planner-Coder-Reviewer pipeline in
the spirit of MAC-SQL; NOT a reproduction of the MAC-SQL paper). All three
roles call the SAME backbone model -- only the system prompt/role differs --
so that L1/L2/L3 differ purely in orchestration structure, not model
capability. See docs/exp_design_l1l2l3_controlled_backbone.md section 4.

Pipeline:
  1. Planner:  filtered_schema, subplan = planner(I, Q, S)
  2. Coder:    Y_0 = coder(I, Q, filtered_schema, subplan)
  3. Refine:   execution-feedback loop (same as L2, but scoped to Coder), MAX_ITER
  4. Reviewer: verdict, critique = reviewer(Q, filtered_schema, Y_t, sample_rows)
               if REJECT: one more Coder revision using the critique, capped at 1 extra call

用法:
  python 04_run_l3_multi_agent.py --provider deepseek --api_key YOUR_KEY --workers 5
  python 04_run_l3_multi_agent.py --dataset bird --provider deepseek --workers 5
"""

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common import (
    ABLATION_DIR,
    BASE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    PROVIDER_CONFIGS,
    resolve_api_key,
    resolve_dataset_defaults,
    build_planner_prompt,
    parse_planner_output,
    build_coder_prompt,
    build_coder_revision_prompt,
    build_reviewer_prompt,
    parse_reviewer_output,
    format_sample_rows,
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
DEFAULT_MAX_TOKENS = {"spider": 384, "bird": 1024}  # BIRD schemas/descriptions are much larger


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
    parser.add_argument("--max_tokens", type=int, default=None, help="Default depends on --dataset (BIRD needs more room)")
    parser.add_argument("--max_iter", type=int, default=MAX_ITER)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 条，用于 smoke test")
    return parser.parse_args()


def process_one(item, examples, provider, api_key, max_tokens, db_dir, max_iter, model=None, base_url=None):
    db_dir = Path(db_dir)
    db_path = find_db_path(item["db_id"], db_dir)
    question = item["question"]
    full_schema = item["db_schema"]
    calls = []

    # --- Agent 1: Planner ---
    planner_prompt = build_planner_prompt(question, full_schema)
    planner_result = call_llm(provider, api_key, PLANNER_SYSTEM_PROMPT, planner_prompt, max_tokens=max_tokens, model=model, base_url=base_url)
    calls.append(planner_result)
    filtered_schema, subplan, planner_parse_ok = parse_planner_output(planner_result.text, fallback_schema=full_schema)

    # --- Agent 2: Coder (initial) ---
    coder_prompt = build_coder_prompt(examples, question, filtered_schema, subplan)
    coder_result = call_llm(provider, api_key, BASE_SYSTEM_PROMPT, coder_prompt, max_tokens=max_tokens, model=model, base_url=base_url)
    calls.append(coder_result)
    pred_sql = extract_sql(coder_result.text)

    # --- Coder self-refine loop on execution errors (same mechanism as L2) ---
    iterations_used = 0
    last_error = None
    for t in range(max_iter):
        if pred_sql.startswith("ERROR:") or db_path is None:
            break
        _, err = execute_sql(str(db_path), pred_sql)
        if err is None:
            break
        last_error = err
        iterations_used = t + 1
        revision_prompt = build_coder_revision_prompt(question, filtered_schema, pred_sql, err, "Execution Error")
        coder_result = call_llm(provider, api_key, BASE_SYSTEM_PROMPT, revision_prompt, max_tokens=max_tokens, model=model, base_url=base_url)
        calls.append(coder_result)
        pred_sql = extract_sql(coder_result.text)

    # --- Agent 3: Reviewer (only if we have an executable SQL to review) ---
    reviewer_verdict = None
    reviewer_critique = None
    reviewer_parse_ok = None
    reviewer_triggered_revision = False
    if not pred_sql.startswith("ERROR:") and db_path is not None:
        rows, err = execute_sql(str(db_path), pred_sql)
        sample_text = format_sample_rows(rows) if err is None else f"(execution error: {err})"
        reviewer_prompt = build_reviewer_prompt(question, filtered_schema, pred_sql, sample_text)
        reviewer_result = call_llm(provider, api_key, REVIEWER_SYSTEM_PROMPT, reviewer_prompt, max_tokens=max_tokens, model=model, base_url=base_url)
        calls.append(reviewer_result)
        reviewer_verdict, reviewer_critique, reviewer_parse_ok = parse_reviewer_output(reviewer_result.text)

        if reviewer_verdict == "REJECT" and reviewer_critique and reviewer_critique.lower() != "none":
            reviewer_triggered_revision = True
            revision_prompt = build_coder_revision_prompt(question, filtered_schema, pred_sql, reviewer_critique, "Reviewer Critique")
            coder_result = call_llm(provider, api_key, BASE_SYSTEM_PROMPT, revision_prompt, max_tokens=max_tokens, model=model, base_url=base_url)
            calls.append(coder_result)
            pred_sql = extract_sql(coder_result.text)

    ex, em = evaluate_one(pred_sql, item["gold_sql"], item["db_id"], db_dir)

    total_in = sum(c.input_tokens for c in calls if c.input_tokens is not None) or None
    total_out = sum(c.output_tokens for c in calls if c.output_tokens is not None) or None
    total_latency = sum(c.latency_s for c in calls if c.latency_s is not None)

    return {
        "id": item["id"],
        "db_id": item["db_id"],
        "question": question,
        "gold_sql": item["gold_sql"],
        "spider_difficulty": item.get("spider_difficulty"),
        "bird_difficulty": item.get("bird_difficulty"),
        "pred_sql": pred_sql,
        "execution_accuracy": ex,
        "exact_match": em,
        "num_llm_calls": len(calls),
        "planner_parse_ok": planner_parse_ok,
        "subplan": subplan,
        "iterations_used": iterations_used,
        "last_execution_error": last_error,
        "reviewer_verdict": reviewer_verdict,
        "reviewer_critique": reviewer_critique,
        "reviewer_parse_ok": reviewer_parse_ok,
        "reviewer_triggered_revision": reviewer_triggered_revision,
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
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS[args.dataset]
    output_file = args.output_file or str(ABLATION_DIR / "outcome" / f"l3_{args.provider}_{args.dataset}.json")

    subset = load_json(subset_file)
    items = subset["items"]
    if args.limit:
        items = items[: args.limit]
    resolved_model = args.model or PROVIDER_CONFIGS[args.provider]["model"]
    print(f"Loaded {len(items)} subset items (level=L3, dataset={args.dataset}, provider={args.provider}, model={resolved_model}, max_tokens={max_tokens}, max_iter={args.max_iter})")

    train_data = load_json(train_file)
    examples = select_few_shot_examples(train_data, args.num_shots, args.seed)
    print(f"Selected {len(examples)} few-shot examples from dbs: {[e['db_id'] for e in examples]}")

    results = []
    ex_correct = 0
    em_correct = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, item, examples, args.provider, api_key, max_tokens, db_dir, args.max_iter, args.model, args.base_url): item
            for item in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="L3"):
            r = future.result()
            results.append(r)
            ex_correct += int(r["execution_accuracy"])
            em_correct += int(r["exact_match"])

    total = len(results)
    avg_calls = sum(r["num_llm_calls"] for r in results) / total
    planner_ok_rate = sum(r["planner_parse_ok"] for r in results) / total
    print(f"\nL3 results: EX={ex_correct}/{total}={ex_correct/total*100:.2f}%  EM={em_correct}/{total}={em_correct/total*100:.2f}%  "
          f"avg_calls={avg_calls:.2f}  planner_parse_ok_rate={planner_ok_rate*100:.1f}%")

    save_json({
        "level": "L3",
        "dataset": args.dataset,
        "provider": args.provider,
        "model": resolved_model,
        "num_shots": args.num_shots,
        "max_iter": args.max_iter,
        "max_tokens": max_tokens,
        "metrics": {
            "execution_accuracy": ex_correct / total,
            "exact_match": em_correct / total,
            "total": total,
            "avg_llm_calls": avg_calls,
            "planner_parse_ok_rate": planner_ok_rate,
        },
        "results": results,
    }, output_file)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
