# TODO: paper revisions for "Agentic-SQL Revisited"

Findings from cross-checking `papers/benchmark.pdf` against the actual code and result
files in `case_study_spider_cot/`.

## P0 — must fix (verifiable factual error)

- [ ] **Unify the sample count for Qwen3-8B CoT-SFT.** `outcome/eval_results.json` has
  `total: 1762`, but the paper states all six configurations are evaluated on the full
  2,147-example Spider test split. Either:
  - re-run Qwen3-8B CoT-SFT on the full 2,147 examples and update Table 3 / Table 5
    (EX=82.24, EM=44.67, and the difficulty breakdown), or
  - keep the 1,762-sample result but explicitly footnote that this row's sample count
    differs from the other five, and explain why (invalid-response filtering)

## P1 — methodology claims don't match the code (overclaim)

- [ ] **EX scorer.** `case_study_spider_cot/scripts/05_evaluate.py`'s `execute_sql()`
  sorts result rows before comparing, which erases ORDER BY semantics. Either switch to
  Spider's official `evaluation.py` execution comparison, or soften "official Spider EX
  scorer" to describe the actual (custom) implementation.
- [ ] **EM scorer.** `normalize_sql()` only lowercases and collapses whitespace — it is
  not the "strict alias-canonicalization scorer of [2]" the paper claims. Either wire in
  Spider's official component-level EM comparison, or correct the paper's wording to
  describe it as a string-level approximation.
- [ ] **Difficulty bands.** `06_difficulty_analysis.py` is a custom SQL-component-counting
  heuristic, not literally Spider's official hardness classifier. Either re-run the
  difficulty split through Spider's official `evaluation.py` hardness function, or correct
  "follow the official Spider difficulty rules" in the Table 5 footnote.

## P2 — unfulfilled promises / dangling citation

- [ ] **Ship the leaderboard harness.** The abstract says "we release a Python harness,"
  but `leaderboard/` is currently just a placeholder README with no code. Either
  implement it before submission, or weaken the abstract wording (e.g. "we describe the
  design of a harness we plan to release").
- [ ] **Resolve the [20] self-citation.** Multiple sections point to "the companion paper
  [20]" for training details (prompts, hyperparameters, CoT generation protocol), but
  that manuscript (drafted only as notes in `benchmark_related_code/paper_outline.md`)
  isn't finished or public. Either finish and publish it first, or move the key training
  details into this paper's appendix / the repo README instead of citing a nonexistent
  document.
- [ ] **Check anonymity requirements.** If the target venue requires double-blind review,
  the title page (real author names/emails) also needs to be anonymized — not just the
  self-citation [20].

## P3 — minor clarity

- [ ] Add explicit (dev)/(test) tags to each row in Table 3 instead of relying on a
  blanket footnote ("literature rows default to dev unless otherwise marked").
