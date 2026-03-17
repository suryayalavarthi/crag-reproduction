"""
error_analysis.py
─────────────────
Analyse CRAG results by question type and action.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

CRAG_FILE = Path("crag_results.json")
GT_FILE   = Path("popqa_full_gt.json")
OUT_FILE  = Path("error_analysis.json")

# ── 1. Load ───────────────────────────────────────────────────────────────────
with open(CRAG_FILE) as f:
    crag = json.load(f)
with open(GT_FILE) as f:
    gt = json.load(f)

print(f"Loaded {len(crag)} CRAG results, {len(gt)} GT entries\n")

# ── 2. Question type detector ─────────────────────────────────────────────────
def question_type(q: str) -> str:
    q = q.lower()
    if "occupation"                            in q: return "occupation"
    if "genre"                                 in q: return "genre"
    if "country"                               in q: return "country"
    if "city"  in q or "born"                  in q: return "city"
    if "sport"                                 in q: return "sport"
    if "director" in q or "directed"           in q: return "director"
    if "composer" in q or "composed"           in q: return "composer"
    if "author" in q or "screenwriter" in q or "wrote" in q: return "author"
    if "religion"                              in q: return "religion"
    return "other"

# ── 3. Match checker ──────────────────────────────────────────────────────────
def is_match(prediction: str, answers: list[str]) -> int:
    pred = prediction.lower().strip()
    for ans in answers:
        if ans.lower().strip() in pred:
            return 1
    return 0

# ── 4. Analyse each record ────────────────────────────────────────────────────
records = []

for r in crag:
    q       = r["question"]
    pred    = r.get("answer", "")
    action  = r.get("action", "UNKNOWN")
    qtype   = question_type(q)
    answers = gt.get(q, [])
    match   = is_match(pred, answers) if answers else None

    records.append({
        "question"     : q,
        "question_type": qtype,
        "action"       : action,
        "prediction"   : pred,
        "gt_answers"   : answers,
        "match"        : match,
    })

# ── 5. Aggregate ──────────────────────────────────────────────────────────────
# By question type
type_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "matched": 0, "actions": Counter()})
for r in records:
    qt = r["question_type"]
    type_stats[qt]["total"] += 1
    type_stats[qt]["actions"][r["action"]] += 1
    if r["match"] is not None:
        type_stats[qt]["matched"] += r["match"]

# By action
action_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "matched": 0, "by_type": Counter()})
for r in records:
    a = r["action"]
    action_stats[a]["total"] += 1
    action_stats[a]["by_type"][r["question_type"]] += 1
    if r["match"] is not None:
        action_stats[a]["matched"] += r["match"]

# ── 6. Print: by question type ────────────────────────────────────────────────
TYPE_ORDER = ["occupation","genre","country","city","sport","director",
              "composer","author","religion","other"]
ACTION_ORDER = ["CORRECT","AMBIGUOUS","INCORRECT"]

print("=" * 72)
print(f"{'QUESTION TYPE':14s}  {'Count':>6}  {'Accuracy':>9}  {'Most common action'}")
print("=" * 72)
for qt in TYPE_ORDER:
    if qt not in type_stats:
        continue
    s    = type_stats[qt]
    tot  = s["total"]
    acc  = s["matched"] / tot * 100 if tot > 0 else 0.0
    top_action = s["actions"].most_common(1)[0][0] if s["actions"] else "—"
    print(f"{qt:14s}  {tot:>6}  {acc:>8.1f}%  {top_action}")

# ── 7. Print: accuracy by action × question type ──────────────────────────────
print()
print("=" * 72)
print("ACCURACY BY ACTION × QUESTION TYPE")
print("=" * 72)

# Header
header = f"{'':14s}"
for a in ACTION_ORDER:
    header += f"  {a:>10}"
print(header)
print("─" * 72)

for qt in TYPE_ORDER:
    if qt not in type_stats:
        continue
    row = f"{qt:14s}"
    for a in ACTION_ORDER:
        # count matches for this (qt, action) pair
        subset = [r for r in records if r["question_type"] == qt and r["action"] == a]
        n = len(subset)
        if n == 0:
            row += f"  {'—':>10}"
        else:
            m = sum(r["match"] for r in subset if r["match"] is not None)
            row += f"  {m}/{n} {m/n*100:4.0f}%"
    print(row)

# ── 8. Print: overall accuracy by action ──────────────────────────────────────
print()
print("=" * 72)
print("OVERALL ACCURACY BY ACTION")
print("=" * 72)
for a in ACTION_ORDER:
    s   = action_stats[a]
    tot = s["total"]
    acc = s["matched"] / tot * 100 if tot > 0 else 0.0
    print(f"  {a:12s}: {s['matched']}/{tot}  ({acc:.1f}%)")

overall_match = sum(r["match"] for r in records if r["match"] is not None)
overall_tot   = sum(1 for r in records if r["match"] is not None)
print(f"\n  {'OVERALL':12s}: {overall_match}/{overall_tot}  ({overall_match/overall_tot*100:.1f}%)")

# ── 9. Save ───────────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as f:
    json.dump(records, f, indent=2)
print(f"\nSaved {len(records)} records to {OUT_FILE}")
