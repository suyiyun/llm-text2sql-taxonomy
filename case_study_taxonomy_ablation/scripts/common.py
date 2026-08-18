"""
Shared utilities for the L1/L2/L3 controlled-backbone taxonomy ablation
(see ../../docs/exp_design_l1l2l3_controlled_backbone.md).

Reused/adapted from case_study_spider_cot/scripts/07_baseline_api_eval.py
(API calling, SQL execution/comparison) and 06_difficulty_analysis.py
(Spider difficulty classification), so that L1/L2/L3 results stay
comparable to the existing 3-shot baselines.
"""

import json
import os
import re
import sqlite3
import time
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # llm-text2sql-taxonomy/
SPIDER_CASE_STUDY_DIR = PROJECT_DIR / "case_study_spider_cot"
SPIDER_DB_DIR = SPIDER_CASE_STUDY_DIR / "spider_data" / "database"
ABLATION_DIR = PROJECT_DIR / "case_study_taxonomy_ablation"

BIRD_DATA_DIR = ABLATION_DIR / "bird_data" / "dev_20240627"
BIRD_DB_DIR = BIRD_DATA_DIR / "dev_databases"
BIRD_DEV_JSON = BIRD_DATA_DIR / "dev.json"
BIRD_TABLES_JSON = BIRD_DATA_DIR / "dev_tables.json"
BIRD_PROCESSED_DIR = ABLATION_DIR / "processed_data"

# BIRD ships its own difficulty labels (self-reported by the dataset authors),
# unlike Spider's rule-based classifier -- NOT the same measuring stick, see
# docs/exp_design_l1l2l3_controlled_backbone.md section 11.
BIRD_DIFFICULTY_LABELS = {"simple": "Simple", "moderate": "Moderate", "challenging": "Challenging"}
BIRD_DIFFICULTY_ORDER = ["simple", "moderate", "challenging"]

# --------------------------------------------------------------------------
# Provider configs (subset of 07_baseline_api_eval.py; DeepSeek is the
# primary controlled backbone, GLM-4 is the optional robustness backbone).
#
# NOTE (2026-08): the legacy model id "deepseek-chat" (DeepSeek V3) was fully
# retired on 2026-07-24 in favor of DeepSeek V4 ("deepseek-v4-flash" /
# "deepseek-v4-pro"). API keys are account-level and unaffected by this --
# only the `model` string had to change. Default here is "deepseek-v4-flash"
# (the direct successor to "deepseek-chat"'s role as the general-purpose
# non-reasoning model); pass --model deepseek-v4-pro on any run script to use
# the larger/long-context variant instead. Because this ablation always uses
# the SAME backbone across L1/L2/L3, switching from V3 to V4 does not break
# the controlled-comparison design -- it only means the existing
# case_study_spider_cot/outcome/eval_results_deepseek_3shot.json baseline
# (EX 51.47%, run under the old "deepseek-chat"=V3) is no longer a valid
# calibration reference for the new L1 arm, since it's a different model
# version now. Re-verify the exact current model ids/pricing at
# https://api-docs.deepseek.com/ before a real run -- provider naming has
# already changed once during this project.
#
# NOTE (2026-08): unlike "deepseek-chat", "deepseek-v4-flash" defaults to
# thinking mode (emits a `reasoning_content` field that counts against
# max_tokens). Empirically this can consume the *entire* max_tokens budget on
# harder Spider queries and leave `content` empty (finish_reason="length")
# -- verified on an Extra-Hard sample. `extra_payload: {"thinking":
# {"type": "disabled"}}` turns this off. We disable it deliberately: L1 is
# meant to be a fast single-turn pass, and letting the backbone do its own
# internal chain-of-thought would blur exactly the L1/L2/L3 distinction this
# ablation is designed to isolate (reasoning should come from the taxonomy's
# orchestration structure, not from the backbone's own hidden reasoning).
# --------------------------------------------------------------------------

PROVIDER_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_style": "openai",
        "extra_payload": {"thinking": {"type": "disabled"}},
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4-flash-250414",
        "api_style": "openai_raw",
        "request_delay": 5.0,
    },
}

BASE_SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural language "
    "question, generate the correct SQL query. Output ONLY the SQL query, nothing else."
)


def resolve_dataset_defaults(dataset: str) -> dict:
    """Default subset_file / db_dir / few-shot pool per dataset, so 02/03/04
    only need a --dataset flag instead of three separate path flags each."""
    if dataset == "spider":
        return {
            "subset_file": ABLATION_DIR / "samples" / "spider_subset.json",
            "db_dir": SPIDER_DB_DIR,
            "train_file": SPIDER_CASE_STUDY_DIR / "processed_data" / "train_for_cot.json",
        }
    elif dataset == "bird":
        return {
            "subset_file": ABLATION_DIR / "samples" / "bird_subset.json",
            "db_dir": BIRD_DB_DIR,
            "train_file": BIRD_PROCESSED_DIR / "bird_fewshot.json",
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def resolve_api_key(provider: str, cli_value: str = None) -> str:
    """CLI --api_key takes precedence; otherwise falls back to
    <PROVIDER>_API_KEY (e.g. DEEPSEEK_API_KEY, GLM_API_KEY)."""
    if cli_value:
        return cli_value
    env_var = f"{provider.upper()}_API_KEY"
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(
            f"No API key given: pass --api_key or set the {env_var} env var "
            f"(e.g. `export {env_var}=...` or source a local .env file)."
        )
    return key


class LLMCallResult:
    """Return type for a single LLM call, carrying usage/latency for efficiency metrics."""

    def __init__(self, text, input_tokens=None, output_tokens=None, latency_s=None, reasoning_tokens=None):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_s = latency_s
        self.reasoning_tokens = reasoning_tokens

    @property
    def is_error(self):
        return self.text.startswith("ERROR:")


def _extract_message_content(message: dict) -> str:
    """Handles reasoning-model responses (e.g. deepseek-v4-flash with thinking
    left enabled) where `content` can come back empty because the whole
    max_tokens budget was spent on `reasoning_content` -- surface that as an
    explicit ERROR instead of silently returning an empty string, so it's
    visible in the pipeline output rather than masquerading as "wrong SQL"."""
    content = (message.get("content") or "").strip()
    if content:
        return content
    reasoning = (message.get("reasoning_content") or "").strip()
    if reasoning:
        return f"ERROR: empty content, truncated mid-reasoning (max_tokens too small): ...{reasoning[-200:]}"
    return "ERROR: empty content from API"


def _call_openai_style(base_url, api_key, model, system_prompt, user_prompt, max_tokens, temperature=0.0, extra_payload=None):
    base_url = base_url.rstrip("/")
    url = f"{base_url}/chat/completions" if base_url.endswith("/v1") or "/v1" in base_url else f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    payload.update(extra_payload or {})
    start = time.time()
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            content = _extract_message_content(message)
            usage = data.get("usage", {})
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
            return LLMCallResult(
                content,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                latency_s=time.time() - start,
                reasoning_tokens=reasoning_tokens,
            )
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                return LLMCallResult(f"ERROR: {e}", latency_s=time.time() - start)


def _call_openai_raw(base_url, api_key, model, system_prompt, user_prompt, max_tokens, temperature=0.0, extra_payload=None):
    """base_url already includes the full path prefix (e.g. /v4/)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    payload.update(extra_payload or {})
    start = time.time()
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                if attempt < 4:
                    wait = min(10 + attempt * 15, 120)
                    time.sleep(wait)
                    continue
                return LLMCallResult(f"ERROR: 429 {resp.text[:200]}", latency_s=time.time() - start)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            content = _extract_message_content(msg)
            usage = data.get("usage", {})
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
            return LLMCallResult(
                content,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                latency_s=time.time() - start,
                reasoning_tokens=reasoning_tokens,
            )
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** (attempt + 1))
            else:
                return LLMCallResult(f"ERROR: {e}", latency_s=time.time() - start)


def call_llm(provider, api_key, system_prompt, user_prompt, max_tokens=512, model=None, base_url=None, temperature=0.0):
    """Single entry point used by all three taxonomy levels (L1/L2/L3) so that
    every role/agent call goes through the same backbone with the same
    calling convention -- this is the "controlled backbone" invariant."""
    config = PROVIDER_CONFIGS[provider]
    model = model or config["model"]
    base_url = base_url or config["base_url"]
    api_style = config["api_style"]
    extra_payload = config.get("extra_payload")

    if api_style == "openai":
        return _call_openai_style(base_url, api_key, model, system_prompt, user_prompt, max_tokens, temperature, extra_payload)
    elif api_style == "openai_raw":
        return _call_openai_raw(base_url, api_key, model, system_prompt, user_prompt, max_tokens, temperature, extra_payload)
    raise ValueError(f"Unsupported api_style: {api_style}")


# --------------------------------------------------------------------------
# SQL extraction / normalization / execution (mirrors 07_baseline_api_eval.py
# so EX/EM numbers stay directly comparable to the existing 3-shot baselines)
# --------------------------------------------------------------------------

def extract_sql(text: str) -> str:
    """NOTE (2026-08): the original fallback path (no ```sql fence) used to
    grab only the FIRST line starting with SELECT and silently drop the rest
    -- this truncated any multi-line SQL (FROM/WHERE/JOIN on their own
    lines), which turned out to be common on BIRD's longer queries and
    corrupted results into false negatives. Fixed to capture from the first
    SELECT/WITH keyword through to a trailing semicolon or the end of the
    response, not just to the first newline."""
    if text.startswith("ERROR:"):
        return text
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*((?:SELECT|WITH).*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(?:SELECT|WITH)\b.*", text, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(0).strip()
        semi = candidate.find(";")
        if semi != -1:
            candidate = candidate[: semi + 1]
        # if the model added trailing prose after the query, a blank line is
        # the most reliable boundary marker between SQL and explanation
        blank = candidate.find("\n\n")
        if blank != -1:
            candidate = candidate[:blank]
        return candidate.strip()
    return text.strip().split("\n")[0]


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def execute_sql(db_path: str, sql: str, timeout: float = 30.0):
    """Returns (rows, None) on success, or (None, error_message) on failure."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return sorted([tuple(str(v) for v in row) for row in results]), None
    except Exception as e:
        return None, str(e)


def find_db_path(db_id: str, db_dir: Path) -> Path:
    for base in [db_dir, db_dir.parent / "test_database"]:
        candidate = base / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            return candidate
        if (base / db_id).exists():
            sqlite_files = list((base / db_id).glob("*.sqlite"))
            if sqlite_files:
                return sqlite_files[0]
    return None


def evaluate_one(pred_sql: str, gold_sql: str, db_id: str, db_dir: Path):
    """Returns (ex: bool, em: bool)."""
    db_path = find_db_path(db_id, db_dir)
    ex = False
    if db_path:
        pred_result, pred_err = execute_sql(str(db_path), pred_sql)
        gold_result, gold_err = execute_sql(str(db_path), gold_sql)
        if pred_err is None and gold_err is None:
            ex = pred_result == gold_result
    em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
    return ex, em


# --------------------------------------------------------------------------
# Spider difficulty classification (verbatim logic from
# case_study_spider_cot/scripts/06_difficulty_analysis.py, duplicated here
# rather than imported so this module has no path-hacking import dependency
# on a sibling case_study_* directory).
# --------------------------------------------------------------------------

def _count_sql_components(sql: str) -> dict:
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
    num_components = sum(
        1 for key in ["where", "group_by", "order_by", "having", "limit", "join", "or", "like"]
        if components[key]
    )
    components["num_components"] = num_components
    return components


def classify_spider_difficulty(sql: str) -> str:
    """Returns one of: easy, medium, hard, extra."""
    comp = _count_sql_components(sql)
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


SPIDER_DIFFICULTY_LABELS = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "extra": "Extra Hard"}
SPIDER_DIFFICULTY_ORDER = ["easy", "medium", "hard", "extra"]


# --------------------------------------------------------------------------
# Few-shot prompt construction (same format as 07_baseline_api_eval.py so
# the L1 arm here is directly comparable to the existing DeepSeek 3-shot run)
# --------------------------------------------------------------------------

def truncate_schema_text(schema_text: str, max_chars: int = 8000) -> str:
    """Truncates a SINGLE schema block, not a whole assembled prompt. This
    must be applied to each schema piece (every few-shot example's schema,
    and the target's schema) individually, BEFORE it's embedded into a
    prompt template -- truncating the final flattened prompt instead (the
    original approach) can silently eat the [Question]/[SQL] task section
    when a schema is large, which happens on several BIRD databases (e.g.
    european_football_2's schema is ~104K chars). Strips `Sample(...)` value
    lines first (cheapest to drop, least informative per char), then hard-
    truncates what's left if still over budget."""
    if len(schema_text) <= max_chars:
        return schema_text
    lines = schema_text.split("\n")
    short = [l for l in lines if not l.strip().startswith("Sample(")]
    shortened = "\n".join(short)
    if len(shortened) <= max_chars:
        return shortened
    return shortened[:max_chars] + "\n... [schema truncated]"


def build_few_shot_prompt(examples: list, question: str, db_schema: str) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"### Example {i}\n"
            f"[Database Schema]\n{truncate_schema_text(ex['db_schema'])}\n\n"
            f"[Question]\n{ex['question']}\n\n"
            f"[SQL]\n{ex['gold_sql']}"
        )
    parts.append(
        f"### Your Task\n"
        f"[Database Schema]\n{truncate_schema_text(db_schema)}\n\n"
        f"[Question]\n{question}\n\n"
        f"[SQL]"
    )
    return "\n\n".join(parts)


def build_refinement_prompt(question: str, db_schema: str, prev_sql: str, error_msg: str) -> str:
    """L2/L3 self-correction prompt: same schema+question, plus the failed SQL
    and its execution error, per the Y_{t+1} = pi(I, Q, S, Y_t, eps | theta)
    formulation in the survey (5.1)."""
    return (
        f"[Database Schema]\n{truncate_schema_text(db_schema)}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Previous SQL]\n{prev_sql}\n\n"
        f"[Execution Error]\n{error_msg}\n\n"
        f"The previous SQL failed to execute. Look at the error and the query, "
        f"then output a corrected SQL query that fixes the issue. "
        f"Output ONLY the corrected SQL query, nothing else.\n\n"
        f"[Corrected SQL]"
    )


def select_few_shot_examples(train_data: list, num_shots: int, seed: int) -> list:
    import random
    random.seed(seed)
    by_db = {}
    for item in train_data:
        by_db.setdefault(item["db_id"], []).append(item)
    db_ids = list(by_db.keys())
    random.shuffle(db_ids)
    examples = []
    for db_id in db_ids:
        if len(examples) >= num_shots:
            break
        examples.append(random.choice(by_db[db_id]))
    return examples


# --------------------------------------------------------------------------
# L3 Planner / Coder / Reviewer role prompts (minimal MAC-SQL-style pipeline,
# see docs/exp_design_l1l2l3_controlled_backbone.md section 4 -- all three
# roles call the SAME backbone model, only system prompt/role differs).
# --------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = (
    "You are the Planner in a multi-agent Text-to-SQL system. Given a full database "
    "schema and a natural language question, your job is to (1) identify only the "
    "tables and columns relevant to answering the question, and (2) break the question "
    "down into a short numbered list of logical sub-steps (e.g. which tables to join, "
    "what to filter/aggregate/sort). You do NOT write SQL."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are the Reviewer in a multi-agent Text-to-SQL system. Given a question, the "
    "schema, a candidate SQL query, and a sample of its execution result, judge whether "
    "the result plausibly answers the question (e.g. catch MIN/MAX confusion, wrong "
    "column selected, wrong filter direction) even though the query executed without error."
)


def build_planner_prompt(question: str, db_schema: str) -> str:
    """Planner gets a larger schema budget than Coder/Reviewer since its whole
    job is filtering the FULL schema down -- capping it too aggressively would
    defeat the point on BIRD's largest databases (e.g. european_football_2 at
    ~104K chars). Still capped, not unlimited: a giant schema is exactly the
    kind of case a real system would need RAG-based retrieval for (see survey
    section 6.4), which is out of scope for this minimal pipeline."""
    return (
        f"[Database Schema]\n{truncate_schema_text(db_schema, max_chars=20000)}\n\n"
        f"[Question]\n{question}\n\n"
        f"Respond in exactly this format:\n"
        f"[Relevant Schema]\n<only the relevant tables/columns, in the same CREATE-TABLE-like style as the input>\n\n"
        f"[Plan]\n<numbered list of sub-steps>"
    )


def parse_planner_output(text: str, fallback_schema: str):
    """Returns (filtered_schema, subplan, parse_ok)."""
    if text.startswith("ERROR:"):
        return fallback_schema, "", False
    schema_match = re.search(r"\[Relevant Schema\]\s*(.*?)(?=\[Plan\]|\Z)", text, re.DOTALL | re.IGNORECASE)
    plan_match = re.search(r"\[Plan\]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if not schema_match or not plan_match:
        return fallback_schema, "", False
    filtered_schema = schema_match.group(1).strip()
    subplan = plan_match.group(1).strip()
    if not filtered_schema:
        return fallback_schema, subplan, False
    return filtered_schema, subplan, True


def build_coder_prompt(examples: list, question: str, db_schema: str, subplan: str) -> str:
    """Coder's initial call. Keeps the same k-shot demonstrations as L1/L2 so
    the three levels share one base prompt structure per section 4 of the design doc."""
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"### Example {i}\n"
            f"[Database Schema]\n{truncate_schema_text(ex['db_schema'])}\n\n"
            f"[Question]\n{ex['question']}\n\n"
            f"[SQL]\n{ex['gold_sql']}"
        )
    parts.append(
        f"### Your Task\n"
        f"[Database Schema]\n{truncate_schema_text(db_schema)}\n\n"
        f"[Plan]\n{subplan if subplan else '(no plan provided, use the schema and question directly)'}\n\n"
        f"[Question]\n{question}\n\n"
        f"[SQL]"
    )
    return "\n\n".join(parts)


def build_coder_revision_prompt(question: str, db_schema: str, prev_sql: str, feedback: str, feedback_kind: str) -> str:
    """feedback_kind is 'Execution Error' (from the refine loop) or 'Reviewer Critique'
    (from the Reviewer agent) -- same shape, different label, so Coder handles both
    uniformly."""
    return (
        f"[Database Schema]\n{truncate_schema_text(db_schema)}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Previous SQL]\n{prev_sql}\n\n"
        f"[{feedback_kind}]\n{feedback}\n\n"
        f"Fix the SQL query based on the feedback above. "
        f"Output ONLY the corrected SQL query, nothing else.\n\n"
        f"[Corrected SQL]"
    )


def format_sample_rows(rows, max_rows: int = 5) -> str:
    if not rows:
        return "(query returned no rows)"
    shown = rows[:max_rows]
    lines = [str(r) for r in shown]
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def build_reviewer_prompt(question: str, db_schema: str, sql: str, sample_rows_text: str) -> str:
    return (
        f"[Database Schema]\n{truncate_schema_text(db_schema)}\n\n"
        f"[Question]\n{question}\n\n"
        f"[SQL]\n{sql}\n\n"
        f"[Sample Execution Result]\n{sample_rows_text}\n\n"
        f"Does this result plausibly answer the question? Respond in exactly this format:\n"
        f"[Verdict]\n<ACCEPT or REJECT>\n\n"
        f"[Critique]\n<if REJECT, briefly explain what is wrong; if ACCEPT, write 'none'>"
    )


def parse_reviewer_output(text: str):
    """Returns (verdict, critique, parse_ok). Defaults to ACCEPT on parse
    failure (fail-safe: never force an extra correction round on unparseable
    output)."""
    if text.startswith("ERROR:"):
        return "ACCEPT", "", False
    verdict_match = re.search(r"\[Verdict\]\s*(ACCEPT|REJECT)", text, re.IGNORECASE)
    critique_match = re.search(r"\[Critique\]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if not verdict_match:
        return "ACCEPT", "", False
    verdict = verdict_match.group(1).upper()
    critique = critique_match.group(1).strip() if critique_match else ""
    return verdict, critique, True


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
