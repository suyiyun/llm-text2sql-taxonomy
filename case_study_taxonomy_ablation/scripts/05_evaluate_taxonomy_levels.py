"""
汇总 L1/L2/L3 三个等级的输出，产出总体 EX/EM 表和按难度分组的 EX 表
（对应 docs/exp_design_l1l2l3_controlled_backbone.md 第 7 节）。

用法:
  python 05_evaluate_taxonomy_levels.py --dataset spider
  python 05_evaluate_taxonomy_levels.py --dataset bird
"""

import argparse
from collections import defaultdict

from common import (
    ABLATION_DIR,
    SPIDER_DIFFICULTY_LABELS,
    SPIDER_DIFFICULTY_ORDER,
    BIRD_DIFFICULTY_LABELS,
    BIRD_DIFFICULTY_ORDER,
    load_json,
    save_json,
)

DATASET_CONFIGS = {
    "spider": {
        "difficulty_order": SPIDER_DIFFICULTY_ORDER,
        "difficulty_labels": SPIDER_DIFFICULTY_LABELS,
        "difficulty_field": "spider_difficulty",
    },
    "bird": {
        "difficulty_order": BIRD_DIFFICULTY_ORDER,
        "difficulty_labels": BIRD_DIFFICULTY_LABELS,
        "difficulty_field": "bird_difficulty",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spider", choices=["spider", "bird"])
    parser.add_argument("--provider", type=str, default="deepseek")
    parser.add_argument("--l1_file", type=str, default=None)
    parser.add_argument("--l2_file", type=str, default=None)
    parser.add_argument("--l3_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def difficulty_breakdown(results, config):
    difficulty_field = config["difficulty_field"]
    difficulty_order = config["difficulty_order"]
    difficulty_labels = config["difficulty_labels"]

    stats = defaultdict(lambda: {"total": 0, "ex_correct": 0})
    for r in results:
        diff = r.get(difficulty_field, "unknown")
        stats[diff]["total"] += 1
        stats[diff]["ex_correct"] += int(r["execution_accuracy"])
    out = {}
    for diff in difficulty_order:
        s = stats[diff]
        out[difficulty_labels[diff]] = {
            "total": s["total"],
            "ex_correct": s["ex_correct"],
            "ex_accuracy": round(s["ex_correct"] / s["total"] * 100, 2) if s["total"] else None,
        }
    return out


def summarize(level_name, data, config):
    results = data["results"]
    total = len(results)
    ex = sum(int(r["execution_accuracy"]) for r in results)
    em = sum(int(r["exact_match"]) for r in results)
    avg_calls = sum(r["num_llm_calls"] for r in results) / total
    tokens = [r["input_tokens"] + r["output_tokens"] for r in results
              if r.get("input_tokens") is not None and r.get("output_tokens") is not None]
    avg_tokens = sum(tokens) / len(tokens) if tokens else None
    avg_latency = sum(r["latency_s"] for r in results if r.get("latency_s") is not None) / total

    summary = {
        "level": level_name,
        "total": total,
        "ex_accuracy": round(ex / total * 100, 2),
        "em_accuracy": round(em / total * 100, 2),
        "avg_llm_calls": round(avg_calls, 2),
        "avg_total_tokens": round(avg_tokens, 1) if avg_tokens is not None else None,
        "avg_latency_s": round(avg_latency, 3),
        "difficulty_breakdown": difficulty_breakdown(results, config),
    }
    return summary


def print_summary_table(summaries, config):
    print("\n" + "=" * 78)
    print(f"{'Level':<6} {'EX(%)':>8} {'EM(%)':>8} {'AvgCalls':>10} {'AvgTokens':>10} {'AvgLat(s)':>10}")
    print("-" * 78)
    for s in summaries:
        print(f"{s['level']:<6} {s['ex_accuracy']:>8.2f} {s['em_accuracy']:>8.2f} "
              f"{s['avg_llm_calls']:>10.2f} {str(s['avg_total_tokens']):>10} {s['avg_latency_s']:>10.3f}")
    print("=" * 78)

    print(f"\n{'Difficulty':<12}" + "".join(f"{s['level']+' EX(%)':>12}" for s in summaries))
    for diff in config["difficulty_order"]:
        label = config["difficulty_labels"][diff]
        row = f"{label:<12}"
        for s in summaries:
            acc = s["difficulty_breakdown"][label]["ex_accuracy"]
            row += f"{acc if acc is not None else float('nan'):>12.2f}"
        print(row)


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    outcome_dir = ABLATION_DIR / "outcome"
    files = {
        "L1": args.l1_file or str(outcome_dir / f"l1_{args.provider}_{args.dataset}.json"),
        "L2": args.l2_file or str(outcome_dir / f"l2_{args.provider}_{args.dataset}.json"),
        "L3": args.l3_file or str(outcome_dir / f"l3_{args.provider}_{args.dataset}.json"),
    }
    output_file = args.output_file or str(outcome_dir / f"taxonomy_comparison_{args.dataset}.json")

    summaries = []
    for level, path in files.items():
        try:
            data = load_json(path)
        except FileNotFoundError:
            print(f"[skip] {level}: {path} not found yet")
            continue
        summaries.append(summarize(level, data, config))

    if not summaries:
        print("No result files found. Run 02/03/04_run_*.py first.")
        return

    print_summary_table(summaries, config)
    save_json({"dataset": args.dataset, "summaries": summaries}, output_file)
    print(f"\nSaved comparison table to {output_file}")


if __name__ == "__main__":
    main()
