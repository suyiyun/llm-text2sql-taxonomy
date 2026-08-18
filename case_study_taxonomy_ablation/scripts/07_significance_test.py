"""
McNemar's exact test，两两比较 L1/L2/L3 在同一个抽样子集上的 EX 结果是否有
统计显著差异（docs/exp_design_l1l2l3_controlled_backbone.md 第 7 节）。

三个等级必须在完全相同的样本 id 集合上跑过（即都用 01_sample_stratified_subset.py
产出的同一份 subset），否则配对检验没有意义 -- 脚本会先检查 id 集合是否一致。

用法:
  python 07_significance_test.py --dataset spider
  python 07_significance_test.py --dataset bird
"""

import argparse
from itertools import combinations

from scipy.stats import binomtest

from common import ABLATION_DIR, load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spider", choices=["spider", "bird"])
    parser.add_argument("--provider", type=str, default="deepseek")
    parser.add_argument("--l1_file", type=str, default=None)
    parser.add_argument("--l2_file", type=str, default=None)
    parser.add_argument("--l3_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def load_ex_by_id(path):
    data = load_json(path)
    return {r["id"]: bool(r["execution_accuracy"]) for r in data["results"]}


def mcnemar_exact(ex_a: dict, ex_b: dict):
    """ex_a/ex_b: {id: bool}. Returns dict with b, c, n_discordant, p_value."""
    common_ids = set(ex_a) & set(ex_b)
    b = sum(1 for i in common_ids if ex_a[i] and not ex_b[i])   # A correct, B wrong
    c = sum(1 for i in common_ids if not ex_a[i] and ex_b[i])   # A wrong, B correct
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return {
        "n_paired_samples": len(common_ids),
        "n_discordant": n,
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "p_value": round(p_value, 4),
    }


def main():
    args = parse_args()
    outcome_dir = ABLATION_DIR / "outcome"
    files = {
        "L1": args.l1_file or str(outcome_dir / f"l1_{args.provider}_{args.dataset}.json"),
        "L2": args.l2_file or str(outcome_dir / f"l2_{args.provider}_{args.dataset}.json"),
        "L3": args.l3_file or str(outcome_dir / f"l3_{args.provider}_{args.dataset}.json"),
    }
    output_file = args.output_file or str(outcome_dir / f"significance_test_{args.dataset}.json")

    ex_by_level = {}
    for level, path in files.items():
        try:
            ex_by_level[level] = load_ex_by_id(path)
        except FileNotFoundError:
            print(f"[skip] {level}: {path} not found yet")

    if len(ex_by_level) < 2:
        print("Need at least 2 levels' result files to run a pairwise test.")
        return

    id_sets = {level: set(d.keys()) for level, d in ex_by_level.items()}
    all_same = len(set(frozenset(s) for s in id_sets.values())) == 1
    if not all_same:
        print("WARNING: sample id sets differ across levels -- McNemar's test will only use the "
              "intersection, which weakens the paired design. Re-run all levels on the same "
              "01_sample_stratified_subset.py output to fix this.")
        for level, s in id_sets.items():
            print(f"  {level}: {len(s)} ids")

    print("\n" + "=" * 78)
    print(f"{'Pair':<10} {'n_paired':>10} {'n_discordant':>14} {'b (A>B)':>10} {'c (A<B)':>10} {'p-value':>10} {'sig(a='+str(args.alpha)+')':>12}")
    print("-" * 78)

    pairwise = []
    for level_a, level_b in combinations(ex_by_level.keys(), 2):
        stat = mcnemar_exact(ex_by_level[level_a], ex_by_level[level_b])
        stat["pair"] = f"{level_a} vs {level_b}"
        sig = "yes" if stat["p_value"] < args.alpha else "no"
        print(f"{level_a+'/'+level_b:<10} {stat['n_paired_samples']:>10} {stat['n_discordant']:>14} "
              f"{stat['b_a_correct_b_wrong']:>10} {stat['c_a_wrong_b_correct']:>10} {stat['p_value']:>10} {sig:>12}")
        stat["significant_at_alpha"] = stat["p_value"] < args.alpha
        pairwise.append(stat)

    print("=" * 78)
    print("b = # samples where the first level is correct and the second is wrong; "
          "c = the reverse. McNemar's exact test (two-sided binomial on the discordant pairs).")

    save_json({"alpha": args.alpha, "pairwise_tests": pairwise}, output_file)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
