"""
按难度分层抽样，产出 L1/L2/L3 三个等级共用的固定样本子集（见 docs/
exp_design_l1l2l3_controlled_backbone.md 第 5.3 节）。

Spider: 用规则式分类器（Easy/Medium/Hard/Extra Hard）。
BIRD:   直接用官方自带的 simple/moderate/challenging 标签，不额外分类
        （两边难度口径不同，不能跨数据集直接比较数值，见设计文档第 11 节）。

用法:
  python 01_sample_stratified_subset.py --dataset spider --per_class 90 --seed 42
  python 01_sample_stratified_subset.py --dataset bird --per_class 90 --seed 42
"""

import argparse
import random
from collections import defaultdict

from common import (
    ABLATION_DIR,
    SPIDER_CASE_STUDY_DIR,
    SPIDER_DIFFICULTY_LABELS,
    SPIDER_DIFFICULTY_ORDER,
    BIRD_DIFFICULTY_LABELS,
    BIRD_DIFFICULTY_ORDER,
    BIRD_PROCESSED_DIR,
    classify_spider_difficulty,
    load_json,
    save_json,
)

DATASET_CONFIGS = {
    "spider": {
        "default_test_file": SPIDER_CASE_STUDY_DIR / "processed_data" / "test_for_cot.json",
        "difficulty_order": SPIDER_DIFFICULTY_ORDER,
        "difficulty_labels": SPIDER_DIFFICULTY_LABELS,
        "difficulty_field": "spider_difficulty",
        "default_output": ABLATION_DIR / "samples" / "spider_subset.json",
    },
    "bird": {
        "default_test_file": BIRD_PROCESSED_DIR / "bird_test.json",
        "difficulty_order": BIRD_DIFFICULTY_ORDER,
        "difficulty_labels": BIRD_DIFFICULTY_LABELS,
        "difficulty_field": "bird_difficulty",
        "default_output": ABLATION_DIR / "samples" / "bird_subset.json",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spider", choices=["spider", "bird"])
    parser.add_argument("--test_file", type=str, default=None, help="Default depends on --dataset")
    parser.add_argument("--per_class", type=int, default=90,
                        help="每个难度档抽多少条（Spider 四档默认约 360 条，BIRD 三档默认约 270 条）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None, help="Default depends on --dataset")
    return parser.parse_args()


def get_difficulty(item: dict, dataset: str) -> str:
    if dataset == "spider":
        return classify_spider_difficulty(item["gold_sql"])
    return item["bird_difficulty"]


def main():
    args = parse_args()
    random.seed(args.seed)

    config = DATASET_CONFIGS[args.dataset]
    test_file = args.test_file or str(config["default_test_file"])
    output_file = args.output_file or str(config["default_output"])
    difficulty_order = config["difficulty_order"]
    difficulty_labels = config["difficulty_labels"]
    difficulty_field = config["difficulty_field"]

    test_data = load_json(test_file)
    print(f"Loaded {len(test_data)} {args.dataset} test samples")

    by_difficulty = defaultdict(list)
    for item in test_data:
        diff = get_difficulty(item, args.dataset)
        by_difficulty[diff].append(item)

    for diff in difficulty_order:
        print(f"  {difficulty_labels[diff]:<12} available={len(by_difficulty[diff])}")

    sampled = []
    shortfall = {}
    for diff in difficulty_order:
        pool = by_difficulty[diff]
        k = min(args.per_class, len(pool))
        if k < args.per_class:
            shortfall[diff] = (k, args.per_class)
        chosen = random.sample(pool, k)
        for item in chosen:
            item = dict(item)
            item[difficulty_field] = diff
            sampled.append(item)

    random.shuffle(sampled)

    print(f"\nSampled {len(sampled)} items total (target was {args.per_class} x {len(difficulty_order)} = {args.per_class * len(difficulty_order)})")
    if shortfall:
        print("WARNING: some difficulty classes had fewer available items than requested:")
        for diff, (got, want) in shortfall.items():
            print(f"  {difficulty_labels[diff]}: got {got}, wanted {want}")

    save_json({
        "dataset": args.dataset,
        "seed": args.seed,
        "per_class": args.per_class,
        "source_file": test_file,
        "total": len(sampled),
        "items": sampled,
    }, output_file)
    print(f"\nSaved subset to {output_file}")


if __name__ == "__main__":
    main()
