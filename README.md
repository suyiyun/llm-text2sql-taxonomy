# llm-text2sql-taxonomy

Official code repository for **"Agentic-SQL Revisited: Autonomy-Based Taxonomy and Empirical Benchmark Analysis for LLM Text-to-SQL"**.

## Overview

The paper reframes LLM-based Text-to-SQL evaluation as a leaderboard aggregation problem: it organizes reported results from existing systems along an inference-autonomy axis (constrained → in-context → iterative → agentic → reasoning-internalized), and anchors the aggregation with a focused empirical case study on Spider.

This repository is organized around those two contributions:

| Directory | Contents |
|---|---|
| `leaderboard/` | Python harness for the autonomy-axis leaderboard aggregation — collects and organizes metrics that systems' authors report on Spider, BIRD, and Spider 2.0. *(placeholder — to be filled in)* |
| `case_study_spider_cot/` | Full pipeline for the empirical case study: Spider preprocessing, DeepSeek V3-generated CoT rationales, LoRA fine-tuning (Qwen3-8B / LLaMA-3.1-8B) with CoT vs. No-CoT ablation, and comparison against DeepSeek V3 / GLM-4 3-shot baselines |
| `docs/` | Supplementary write-up material (dataset statistics, paper outline) |

## Case Study Quickstart

See [`case_study_spider_cot/README.md`](case_study_spider_cot/README.md) for the full pipeline (schema extraction → CoT generation → SFT data construction → fine-tuning → evaluation → difficulty analysis).

## Citation

```
@article{agenticsql2025revisited,
  title={Agentic-SQL Revisited: Autonomy-Based Taxonomy and Empirical Benchmark Analysis for LLM Text-to-SQL},
  year={2025}
}
```
