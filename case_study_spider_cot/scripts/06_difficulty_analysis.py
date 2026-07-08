"""
基于已有的eval_results.json，按Spider SQL难度等级（Easy/Medium/Hard/Extra Hard）
分别计算Execution Accuracy。

Spider难度分级标准（基于SQL结构复杂度）：
  - Easy: 单表，无JOIN/子查询/GROUP BY/ORDER BY/集合操作等
  - Medium: 含JOIN或GROUP BY或ORDER BY或HAVING等
  - Hard: 含子查询或2个以上SELECT组件
  - Extra Hard: 含嵌套子查询或集合操作(INTERSECT/UNION/EXCEPT)等

用法（本地运行，不需要GPU）:
  python 06_difficulty_analysis.py --eval_file ../eval_results.json
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default="eval_results.json")
    parser.add_argument("--output_file", type=str, default="difficulty_results.json")
    return parser.parse_args()


def count_sql_components(sql: str) -> dict:
    """统计SQL中各种组件的出现情况。"""
    sql_upper = sql.upper()
    sql_upper = re.sub(r"'[^']*'", "''", sql_upper)
    sql_upper = re.sub(r'"[^"]*"', '""', sql_upper)

    components = {
        "select_count": len(re.findall(r'\bSELECT\b', sql_upper)),
        "where": bool(re.search(r'\bWHERE\b', sql_upper)),
        "group_by": bool(re.search(r'\bGROUP\s+BY\b', sql_upper)),
        "order_by": bool(re.search(r'\bORDER\s+BY\b', sql_upper)),
        "having": bool(re.search(r'\bHAVING\b', sql_upper)),
        "limit": bool(re.search(r'\bLIMIT\b', sql_upper)),
        "join": bool(re.search(r'\bJOIN\b', sql_upper)),
        "or": bool(re.search(r'\bOR\b', sql_upper)),
        "like": bool(re.search(r'\bLIKE\b', sql_upper)),
        "intersect": bool(re.search(r'\bINTERSECT\b', sql_upper)),
        "union": bool(re.search(r'\bUNION\b', sql_upper)),
        "except": bool(re.search(r'\bEXCEPT\b', sql_upper)),
        "nested_subquery": sql_upper.count('SELECT') > 1 and bool(re.search(r'\(\s*SELECT', sql_upper)),
    }

    num_components = 0
    for key in ["where", "group_by", "order_by", "having", "limit", "join", "or", "like"]:
        if components[key]:
            num_components += 1

    components["num_components"] = num_components
    return components


def classify_difficulty(sql: str) -> str:
    """
    按Spider标准对SQL进行难度分级。
    参考Spider官方evaluation.py中的hardness分类逻辑。
    """
    comp = count_sql_components(sql)

    if comp["intersect"] or comp["union"] or comp["except"]:
        return "extra"

    if comp["nested_subquery"] and comp["select_count"] > 2:
        return "extra"

    if comp["nested_subquery"]:
        return "hard"

    if comp["select_count"] > 1:
        return "hard"

    if comp["num_components"] >= 3:
        return "hard"

    if comp["num_components"] >= 1:
        return "medium"

    return "easy"


DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
    "extra": "Extra Hard",
}
DIFFICULTY_ORDER = ["easy", "medium", "hard", "extra"]


def main():
    args = parse_args()

    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", data)
    if isinstance(data, dict) and "results" in data:
        results = data["results"]

    stats = defaultdict(lambda: {"total": 0, "ex_correct": 0, "em_correct": 0})

    for item in results:
        gold_sql = item["gold_sql"]
        difficulty = classify_difficulty(gold_sql)

        stats[difficulty]["total"] += 1
        if item.get("execution_accuracy", False):
            stats[difficulty]["ex_correct"] += 1
        if item.get("exact_match", False):
            stats[difficulty]["em_correct"] += 1

    print("=" * 70)
    print(f"{'Difficulty':<15} {'Count':>8} {'EX Correct':>12} {'EX Acc(%)':>10} {'EM Acc(%)':>10}")
    print("-" * 70)

    total_all = 0
    ex_all = 0
    em_all = 0

    output_stats = {}
    for diff in DIFFICULTY_ORDER:
        s = stats[diff]
        total_all += s["total"]
        ex_all += s["ex_correct"]
        em_all += s["em_correct"]

        ex_acc = s["ex_correct"] / s["total"] * 100 if s["total"] > 0 else 0
        em_acc = s["em_correct"] / s["total"] * 100 if s["total"] > 0 else 0

        print(f"{DIFFICULTY_LABELS[diff]:<15} {s['total']:>8} {s['ex_correct']:>12} {ex_acc:>9.2f}% {em_acc:>9.2f}%")

        output_stats[DIFFICULTY_LABELS[diff]] = {
            "total": s["total"],
            "ex_correct": s["ex_correct"],
            "ex_accuracy": round(ex_acc, 2),
            "em_correct": s["em_correct"],
            "em_accuracy": round(em_acc, 2),
        }

    print("-" * 70)
    overall_ex = ex_all / total_all * 100 if total_all > 0 else 0
    overall_em = em_all / total_all * 100 if total_all > 0 else 0
    print(f"{'All':<15} {total_all:>8} {ex_all:>12} {overall_ex:>9.2f}% {overall_em:>9.2f}%")
    print("=" * 70)

    output_stats["All"] = {
        "total": total_all,
        "ex_correct": ex_all,
        "ex_accuracy": round(overall_ex, 2),
        "em_correct": em_all,
        "em_accuracy": round(overall_em, 2),
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_stats, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
