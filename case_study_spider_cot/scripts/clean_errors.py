import json
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "outcome/eval_results_openai_3shot.json"

with open(filepath, "r", encoding="utf-8") as f:
    d = json.load(f)

before = len(d["results"])
d["results"] = [r for r in d["results"] if not r["full_response"].startswith("ERROR")]
after = len(d["results"])

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

print(f"Removed {before - after} errors, kept {after} valid results")
