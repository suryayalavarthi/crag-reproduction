"""
prepare_arc.py
──────────────
Download and prepare ARC-Challenge test set for CRAG evaluation.
Output: arc_challenge_test.json
"""

import subprocess, sys

try:
    import datasets
except ImportError:
    print("Installing datasets ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])

import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset

OUT_FILE = Path("arc_challenge_test.json")

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading ARC-Challenge test split ...")
dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
print(f"  Loaded {len(dataset)} examples\n")

# ── 2. Show first 3 raw examples ─────────────────────────────────────────────
print("─" * 60)
print("FIRST 3 RAW EXAMPLES (all fields)")
print("─" * 60)
for i in range(3):
    ex = dataset[i]
    print(f"\nExample {i+1}:")
    for k, v in ex.items():
        print(f"  {k}: {v}")

# ── 3. Format and save ────────────────────────────────────────────────────────
LABELS = ["A", "B", "C", "D", "E"]

def format_example(ex: dict) -> dict:
    choices   = ex["choices"]
    texts     = choices["text"]
    labels    = choices["label"]   # may be ["A","B","C","D"] or ["1","2","3","4"]

    # Normalise numeric labels → letters
    label_map = {}
    for lbl, txt in zip(labels, texts):
        if lbl.isdigit():
            label_map[lbl] = LABELS[int(lbl) - 1]
        else:
            label_map[lbl] = lbl

    # Build formatted question string
    parts = [f"Question: {ex['question']}"]
    for lbl, txt in zip(labels, texts):
        parts.append(f"{label_map[lbl]}) {txt}")
    formatted_q = "  ".join(parts)

    # Normalise answer key
    raw_key = ex["answerKey"]
    answer_key = label_map.get(raw_key, raw_key)

    # Get correct answer text
    correct_text = ""
    for lbl, txt in zip(labels, texts):
        if label_map.get(lbl, lbl) == answer_key:
            correct_text = txt
            break

    return {
        "id"          : ex["id"],
        "question"    : formatted_q,
        "answer_key"  : answer_key,
        "correct_text": correct_text,
    }

print("\n\nFormatting all examples ...")
records = [format_example(ex) for ex in dataset]

with open(OUT_FILE, "w") as f:
    json.dump(records, f, indent=2)

# ── 4. Stats ──────────────────────────────────────────────────────────────────
key_dist = Counter(r["answer_key"] for r in records)

print("=" * 60)
print(f"Total questions : {len(records)}")
print(f"\nAnswer key distribution:")
for k in sorted(key_dist):
    pct = key_dist[k] / len(records) * 100
    print(f"  {k}: {key_dist[k]:3d}  ({pct:.1f}%)")

print(f"\n{'─'*60}")
print("3 SAMPLE EXAMPLES (formatted)")
print("─" * 60)
for r in records[:3]:
    print(f"\nID          : {r['id']}")
    print(f"Question    : {r['question']}")
    print(f"Answer key  : {r['answer_key']}")
    print(f"Correct text: {r['correct_text']}")

print(f"\n✓ Saved {len(records)} records to {OUT_FILE}  ({OUT_FILE.stat().st_size:,} bytes)")
