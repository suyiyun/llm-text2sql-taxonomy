# Taxonomy Ablation: Controlled-Backbone L1/L2/L3 Comparison

Implementation of the experiment designed in
[`../docs/exp_design_l1l2l3_controlled_backbone.md`](../docs/exp_design_l1l2l3_controlled_backbone.md):
fix the backbone LLM and vary only the inference-time orchestration
(L1 single-turn / L2 execution-feedback refinement / L3 Planner-Coder-Reviewer),
to isolate the effect of taxonomy level from backbone capability. Supports
`papers/survey.pdf`'s Table 2.

**Status: done for both Spider and BIRD.** Full runs (N=360 Spider, N=270 BIRD) completed
with DeepSeek V4-Flash (`deepseek-v4-flash`, thinking disabled) as the fixed backbone.
Results below. GLM-4 robustness backbone not yet run (optional, RQ4).

## Results

### Spider (N=360, stratified Easy/Medium/Hard/Extra Hard, 90 each)

| Level | EX | EM | Avg calls | Avg tokens | Avg latency |
|---|---|---|---|---|---|
| L1 | 79.44% | 22.22% | 1.00 | 2769 | 1.12s |
| L2 | 80.00% | 22.22% | 1.01 | 2771 | 1.11s |
| L3 | 78.61% | 22.22% | 3.03 | 3395 | 3.73s |

By difficulty (EX%): Easy 95.56/95.56/92.22, Medium 71.11/71.11/67.78, Hard 78.89/80.00/82.22,
Extra Hard 72.22/73.33/72.22 (L1/L2/L3).

McNemar's exact test: **no pairwise difference is significant** (L1-L2 p=0.625, L1-L3 p=0.736,
L2-L3 p=0.500). With a strong V4 backbone, Spider is essentially saturated for this model --
L3's extra machinery doesn't buy anything measurable, and the difficulty-stratified trend
(clean L1<L2<L3 concentrated on Hard/Extra Hard) that showed up in an earlier, buggy run
did not hold up once corrected.

### BIRD (N=270, official simple/moderate/challenging labels, 90 each)

| Level | EX | EM | Avg calls | Avg tokens | Avg latency |
|---|---|---|---|---|---|
| L1 | 48.89% | 2.59% | 1.00 | 6927 | 1.71s |
| L2 | 50.74% | 2.59% | 1.12 | 7193 | 1.90s |
| L3 | 54.81% | 2.59% | 3.19 | 8970 | 4.69s |

By difficulty (EX%): Simple 66.67/67.78/67.78, Moderate 47.78/48.89/56.67,
Challenging 32.22/35.56/40.00 (L1/L2/L3).

McNemar's exact test: **L1 vs L3 is significant, p=0.0365** (L1-L2 p=0.227, L2-L3 p=0.152 are
not). Gains are flat on Simple (near ceiling) and concentrated in Moderate/Challenging --
exactly the pattern the taxonomy predicts, with actual statistical support this time.
BIRD costs ~2.5x Spider's tokens per call (larger schemas + evidence field), as anticipated
in the design doc's cost estimate.

### Headline takeaway

The taxonomy-level effect (L1→L3 autonomy gain) is **real and statistically significant on
BIRD** -- the same benchmark `survey.pdf`'s Table 2 uses -- but **washes out on Spider** once
the backbone is strong enough to nearly saturate it. That contrast is a more defensible,
interesting story for the paper than a flat "more autonomy = better" claim: autonomy helps
on harder, more realistic queries and buys little-to-nothing on easy ones, which also lines
up with BIRD's own difficulty-stratified numbers (Simple ~flat, Challenging clear gain).

## Bugs found and fixed during this run (documented for anyone re-running or extending)

1. **`extract_sql` truncated multi-line SQL to its first line** (inherited from
   `case_study_spider_cot/scripts/07_baseline_api_eval.py`'s original fallback parser). Any
   response without a ` ```sql ` fence that put `FROM`/`WHERE`/`JOIN` on separate lines lost
   everything after the first `\n` -- silently turning correct SQL into garbage. This was
   invisible on Spider until BIRD's longer queries made the model format multi-line more
   often and results looked implausibly bad (20% EX on a BIRD L1 smoke test). Fixed in
   `common.py` to capture from the first `SELECT`/`WITH` to a trailing semicolon or blank
   line, not just the first line. **This changed the Spider numbers substantially** (L1 EX
   64.17% -> 79.44%) -- the first full Spider run reported before this fix is superseded by
   the corrected numbers above.
2. **`deepseek-v4-flash` defaults to thinking mode** and can spend the entire `max_tokens`
   budget on hidden `reasoning_content`, leaving `content` empty (`finish_reason: "length"`)
   -- verified on an Extra-Hard Spider sample. Fixed by passing
   `extra_payload: {"thinking": {"type": "disabled"}}`.
3. **Prompt-level schema truncation could delete the `[Question]`/`[SQL]` task section
   entirely** on BIRD's largest schemas (`european_football_2` is ~104K chars; a naive
   15,000-char cutoff on the *assembled* prompt landed inside a few-shot example's schema,
   before ever reaching the real task). Fixed by truncating each schema block individually
   (`truncate_schema_text`, applied inside each prompt builder) before assembly, so the task
   template text can never be at risk.
4. **`scipy`'s `binomtest().pvalue` is a numpy float64**, and `numpy.bool_` isn't JSON
   serializable -- crashed `07_significance_test.py` on save. Fixed by casting to native
   Python `float`.

## Setup

```bash
cd case_study_taxonomy_ablation
python3 -m venv .venv && source .venv/bin/activate
pip install requests tqdm scipy
```

Spider side reuses `../case_study_spider_cot/processed_data/test_for_cot.json` and
`../case_study_spider_cot/spider_data/database/` directly -- no download needed.

BIRD side downloads its own data (~330MB) into `bird_data/` (gitignored) via
`00a_download_bird.py`.

## Pipeline

```bash
cd scripts

# --- Spider ---
python 01_sample_stratified_subset.py --dataset spider --per_class 90 --seed 42
python 02_run_l1_single_turn.py      --dataset spider --provider deepseek --workers 8
python 03_run_l2_iterative_refine.py --dataset spider --provider deepseek --workers 8
python 04_run_l3_multi_agent.py      --dataset spider --provider deepseek --workers 8
python 05_evaluate_taxonomy_levels.py --dataset spider
python 07_significance_test.py       --dataset spider

# --- BIRD ---
python 00a_download_bird.py
python 00b_prepare_bird_dataset.py
python 01_sample_stratified_subset.py --dataset bird --per_class 90 --seed 42
python 02_run_l1_single_turn.py      --dataset bird --provider deepseek --workers 8
python 03_run_l2_iterative_refine.py --dataset bird --provider deepseek --workers 8
python 04_run_l3_multi_agent.py      --dataset bird --provider deepseek --workers 8
python 05_evaluate_taxonomy_levels.py --dataset bird
python 07_significance_test.py       --dataset bird
```

API key: pass `--api_key`, or set `DEEPSEEK_API_KEY` / `GLM_API_KEY` in the environment
(a local gitignored `.env` with `export DEEPSEEK_API_KEY=...` works well, `source .env`
before running). Every run script supports `--limit N` to smoke-test on a handful of
samples first -- do this before spending the full API budget, especially on BIRD where a
single L3 run over N=270 takes several minutes. `--provider glm` switches the backbone for
the optional robustness check (RQ4 in the design doc).

## Notes on the L3 implementation

`04_run_l3_multi_agent.py` is a **minimal, self-designed** Planner-Coder-Reviewer
pipeline in the spirit of MAC-SQL, not a reproduction of any specific paper. All three
roles call the same backbone model via `common.call_llm` with only the system prompt
differing, which is what keeps the L1/L2/L3 comparison controlled. Planner/Reviewer
outputs are parsed with a regex format contract; on parse failure the pipeline falls
back safely (full schema instead of filtered schema; ACCEPT instead of forcing a
revision) rather than crashing -- `planner_parse_ok` / `reviewer_parse_ok` in the output
record this (both were 100% on real runs, so parsing wasn't actually a problem in practice).

## Outstanding work

- [ ] Write these results into `acl_paper.tex` (a new table near Table 2, and update the
      qualitative claims in survey sections 6.2/6.4 that this data now either supports or
      contradicts -- notably the Spider "concentrated on Hard/Extra Hard" claim did NOT
      hold up after the bug fix, only BIRD's did)
- [ ] Optional: repeat with `--provider glm` for the robustness check (RQ4)
- [ ] Optional: qualitative error case export (L1-fails-L3-fixes examples) for the paper's
      Analysis section
