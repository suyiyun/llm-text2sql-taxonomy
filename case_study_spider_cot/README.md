# CoT-SFT Text-to-SQL Benchmark

A benchmark study of Chain-of-Thought supervised fine-tuning for cross-domain Text-to-SQL on Spider.

## Overview

This repository provides the full pipeline for:
1. Preprocessing the Spider Text-to-SQL dataset
2. Generating 5-step CoT rationales via DeepSeek V3 (teacher model)
3. Building ShareGPT-format SFT datasets (CoT and No-CoT variants)
4. Fine-tuning open-source LLMs (Qwen3-8B, LLaMA-3.1-8B) with LoRA
5. Evaluating with Execution Accuracy (EX) and Exact Match (EM)
6. Comparing against few-shot prompting baselines (DeepSeek V3, GLM-4)

## Research Questions

1. Does CoT supervised fine-tuning improve Text-to-SQL performance on a cross-domain benchmark?
2. Is the CoT benefit consistent across different base models (Qwen3-8B vs LLaMA-3.1-8B)?
3. Does CoT help more on harder SQL queries (Hard / Extra Hard)?
4. Can fine-tuned open-source 8B models match or exceed strong closed-source 3-shot baselines?

## Dataset

Built on [Spider](https://yale-lily.github.io/spider) — a cross-domain Text-to-SQL benchmark.

| Split | Samples | Databases |
|-------|--------:|----------:|
| Train (`train_spider` + `train_others`) | 8,659 | 146 |
| Test  | 2,147 | 40 |

Train and test databases are **completely non-overlapping**, ensuring cross-domain generalization evaluation.

### SQL Difficulty Distribution

| Difficulty | Train | Train % | Test | Test % |
|------------|------:|--------:|-----:|-------:|
| Easy       | 962   | 11.11%  | 296  | 13.79% |
| Medium     | 4,902 | 56.61%  | 1,101 | 51.28% |
| Hard       | 2,196 | 25.36%  | 567  | 26.41% |
| Extra Hard | 599   | 6.92%   | 183  |  8.52% |

## Pipeline

```
Spider raw data + SQLite databases
        |
        | 01_extract_schema.py
        v
Schema representations (db_schemas.json)
        |
        | 02_prepare_dataset.py
        v
question + schema + gold SQL
        |
        | 03_generate_cot.py  (calls DeepSeek V3 API)
        v
question + schema + gold SQL + CoT rationale
        |
        | 04_build_sft_dataset.py / 04_build_sft_dataset_nocot.py
        v
ShareGPT SFT data + evaluation data
        |
        | LoRA fine-tuning (LLaMA-Factory)
        v
Fine-tuned model
        |
        | 05_evaluate*.py
        v
Evaluation results (EX / EM)
        |
        | 06_difficulty_analysis.py
        v
Difficulty-wise breakdown
```

## CoT Format

Each training sample uses a 5-step reasoning chain:

1. **Schema Linking** — identify relevant tables and columns
2. **Logic Decomposition** — break the question into SQL-level operations
3. **Draft SQL** — generate an initial SQL query
4. **SQL Revision** — check and fix errors
5. **Final SQL** — output the final SQL query

## Scripts

| Script | Description |
|--------|-------------|
| `01_extract_schema.py` | Extract schema from SQLite databases |
| `02_prepare_dataset.py` | Build `question + schema + gold SQL` samples |
| `03_generate_cot.py` | Generate CoT rationales using DeepSeek V3 |
| `04_build_sft_dataset.py` | Build ShareGPT CoT SFT data |
| `04_build_sft_dataset_nocot.py` | Build No-CoT ablation data |
| `05_evaluate.py` | Evaluate Qwen3-8B CoT model |
| `05_evaluate_nocot.py` | Evaluate Qwen3-8B No-CoT model |
| `05_evaluate_llama_cot.py` | Evaluate LLaMA-3.1-8B CoT model |
| `05_evaluate_llama_nocot.py` | Evaluate LLaMA-3.1-8B No-CoT model |
| `06_difficulty_analysis.py` | Analyze results by SQL difficulty level |
| `07_baseline_api_eval.py` | 3-shot baseline with commercial APIs |

## Results

### Overall Performance

| Model | Method | EX (%) | EM (%) |
|-------|--------|-------:|-------:|
| Qwen3-8B | CoT SFT | 82.24 | 44.67 |
| Qwen3-8B | No-CoT SFT | 77.04 | 47.60 |
| LLaMA-3.1-8B | CoT SFT | 76.01 | 29.02 |
| LLaMA-3.1-8B | No-CoT SFT | 76.01 | 31.35 |
| DeepSeek V3 | 3-shot | 51.47 | 19.56 |
| GLM-4 | 3-shot | 66.28 | 20.49 |

### Difficulty-wise EX (%)

| Model | Method | Easy | Medium | Hard | Extra Hard |
|-------|--------|-----:|-------:|-----:|-----------:|
| Qwen3-8B | CoT SFT | 94.95 | 80.32 | 81.09 | 79.25 |
| Qwen3-8B | No-CoT SFT | 96.62 | 76.20 | 70.19 | 71.58 |
| LLaMA-3.1-8B | CoT SFT | 96.62 | 74.93 | 68.25 | 73.22 |
| LLaMA-3.1-8B | No-CoT SFT | 96.28 | 75.30 | 69.49 | 67.76 |
| DeepSeek V3 | 3-shot | 96.62 | 53.22 | 33.33 | 24.04 |
| GLM-4 | 3-shot | 92.57 | 67.03 | 59.08 | 41.53 |

**EX** = Execution Accuracy (primary metric). **EM** = Exact Match (auxiliary metric).

## Setup

```bash
pip install -r requirements.txt
```

Download the Spider dataset from https://yale-lily.github.io/spider and place it under `spider_data/`.

## Usage

```bash
# Step 1: Extract database schemas
python scripts/01_extract_schema.py

# Step 2: Build train/test samples
python scripts/02_prepare_dataset.py

# Step 3: Generate CoT rationales (requires DeepSeek V3 API key)
python scripts/03_generate_cot.py

# Step 4a: Build CoT SFT data
python scripts/04_build_sft_dataset.py

# Step 4b: Build No-CoT ablation data
python scripts/04_build_sft_dataset_nocot.py

# Step 5: Evaluate (after LoRA fine-tuning with LLaMA-Factory)
python scripts/05_evaluate.py

# Step 6: Difficulty analysis
python scripts/06_difficulty_analysis.py

# Step 7: API baseline evaluation
python scripts/07_baseline_api_eval.py
```

## Citation

If you use this code or benchmark, please cite:

```
@article{cot4sql2025,
  title={A Benchmark Study of Chain-of-Thought Supervised Fine-Tuning for Cross-Domain Text-to-SQL},
  year={2025}
}
```
