"""
build_ground_truth.py
─────────────────────
Match crag_results.json questions against the full PopQA dataset
to build ground truth answers for all 1,385 questions.

Output: popqa_full_gt.json  →  {question: [answer1, answer2, ...]}
"""

import subprocess, sys

try:
    import datasets
except ImportError:
    print("Installing datasets ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])

import json
from pathlib import Path
from datasets import load_dataset

CRAG_FILE = Path("crag_results.json")
OUT_FILE  = Path("popqa_full_gt.json")

# ── 1. Load crag_results.json ─────────────────────────────────────────────────
print(f"Loading {CRAG_FILE} ...")
with open(CRAG_FILE) as f:
    crag_results = json.load(f)
print(f"  Total questions in crag_results.json: {len(crag_results)}")

# ── 2. Load full PopQA dataset ────────────────────────────────────────────────
print("\nLoading PopQA dataset from HuggingFace ...")
dataset = load_dataset("akariasai/PopQA", split="test")
print(f"  PopQA test split size: {len(dataset)}")

# ── 3. Build lookup: exact question -> possible_answers ───────────────────────
print("\nBuilding lookup index ...")
popqa_index: dict[str, list[str]] = {}
for row in dataset:
    q = row["question"].strip()
    answers = row.get("possible_answers", [])
    # possible_answers may be a JSON string or already a list
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except Exception:
            answers = [answers]
    popqa_index[q] = answers

print(f"  Unique questions indexed: {len(popqa_index)}")

# ── 4. Match crag questions against PopQA ─────────────────────────────────────
print("\nMatching crag_results questions ...")
gt: dict[str, list[str]] = {}
matched   = 0
unmatched = []

for record in crag_results:
    q = record["question"].strip()
    if q in popqa_index:
        gt[q] = popqa_index[q]
        matched += 1
    else:
        # Try case-insensitive fallback
        q_lower = q.lower()
        found = next((v for k, v in popqa_index.items() if k.lower() == q_lower), None)
        if found is not None:
            gt[q] = found
            matched += 1
        else:
            unmatched.append(q)

# ── 5. Save ───────────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as f:
    json.dump(gt, f, indent=2)

# ── 6. Report ─────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Total questions in crag_results.json : {len(crag_results)}")
print(f"Questions matched with GT            : {matched}")
print(f"Questions with no match              : {len(unmatched)}")
print(f"Output saved to                      : {OUT_FILE}")

if unmatched:
    print(f"\nUnmatched questions ({min(5, len(unmatched))} shown):")
    for q in unmatched[:5]:
        print(f"  • {q}")

# ── 7. Sample 5 matched examples ──────────────────────────────────────────────
print("\n" + "─" * 60)
print("5 MATCHED EXAMPLES")
print("─" * 60)
for q, answers in list(gt.items())[:5]:
    print(f"  Q : {q}")
    print(f"  A : {answers}")
    print()
