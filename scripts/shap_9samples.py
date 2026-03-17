"""
shap_9samples.py
────────────────
SHAP analysis on 9 samples (3 per action: CORRECT, AMBIGUOUS, INCORRECT).
Uses exact Shapley: max_evals = 2 * max_words + 1
Estimated runtime: ~1.5 hours on M2 CPU.
"""

import matplotlib
matplotlib.use("Agg")

import os, json, pickle, time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap

CRAG_FILE = Path("crag_results.json")
CKPT      = Path("models/finetuned_t5_evaluator")
PKL_OUT   = Path("shap_values_9.pkl")
PLOT_OUT  = Path("shap_9_summary.png")

# ── 1. Load crag_results.json ─────────────────────────────────────────────────
print("Loading crag_results.json ...")
with open(CRAG_FILE) as f:
    crag = json.load(f)

ACTION_ORDER = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
N_PER_ACTION = 3

def first_n_words(text: str, n: int = 15) -> str:
    return " ".join(text.split()[:n])

selected = {a: [] for a in ACTION_ORDER}
for r in crag:
    action  = r.get("action", "")
    context = r.get("context", "").strip()
    if not context or action not in selected:
        continue
    if len(selected[action]) < N_PER_ACTION:
        doc  = first_n_words(context, 15)
        pair = f"{r['question']} [SEP] {doc}"
        selected[action].append({
            "question": r["question"],
            "action"  : action,
            "pair"    : pair,
        })

all_samples = []
for a in ACTION_ORDER:
    all_samples.extend(selected[a])
all_pairs = [s["pair"] for s in all_samples]

print(f"Selected {len(all_pairs)} samples:")
for s in all_samples:
    print(f"  [{s['action']:10s}] {s['pair'][:80]}")

# ── 2. Load model ─────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(str(CKPT), use_fast=False)

print("Loading model ...")
model = AutoModelForSequenceClassification.from_pretrained(
    str(CKPT), num_labels=1, ignore_mismatched_sizes=True
)
model.eval()
model.to("cpu")
print("✓ Model loaded.\n")

# ── 3. score_fn ───────────────────────────────────────────────────────────────
def score_fn(texts):
    if len(texts) == 0:
        return np.array([], dtype=np.float32).reshape(0, 1)
    enc = tokenizer(
        list(texts), return_tensors="pt",
        padding=True, truncation=True, max_length=512,
    )
    with torch.no_grad():
        out = model(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
    return out.logits.cpu().numpy().astype(np.float32)   # (N, 1)

# ── 4. Verify scores ──────────────────────────────────────────────────────────
print("── Score verification ──")
scores = score_fn(all_pairs)
all_ok = True
for s, score_row in zip(all_samples, scores):
    sc = float(score_row[0])
    expected = "> 0.5" if s["action"] == "CORRECT" else ("< -0.5" if s["action"] == "INCORRECT" else "any")
    flag = ""
    if s["action"] == "CORRECT"   and sc <= 0.5:  flag = "  ⚠ expected > 0.5"
    if s["action"] == "INCORRECT" and sc >= -0.5: flag = "  ⚠ expected < -0.5"
    if flag: all_ok = False
    print(f"  [{s['action']:10s}]  score={sc:+.4f}{flag}")
    print(f"             {s['pair'][:70]}")

if not all_ok:
    print("\n⚠ Some scores outside expected range — continuing anyway (AMBIGUOUS cases may score anywhere).")
print()

# ── 5. Compute max_evals ──────────────────────────────────────────────────────
max_words = max(len(p.split()) for p in all_pairs)
MAX_EVALS = 2 * max_words + 1
print(f"max words across samples : {max_words}")
print(f"max_evals (exact Shapley): {MAX_EVALS}")
print(f"Estimated time           : {MAX_EVALS * len(all_pairs) * 20 / 60:.0f}–{MAX_EVALS * len(all_pairs) * 30 / 60:.0f} min on M2 CPU\n")

# ── 6. Run SHAP sample-by-sample for progress printing ────────────────────────
print("Building SHAP explainer ...")
masker    = shap.maskers.Text(tokenizer, mask_token="[MASK]", collapse_mask_token=True)
explainer = shap.Explainer(score_fn, masker, max_evals=MAX_EVALS)

print("Running SHAP (one sample at a time for progress):\n")
all_sv = []
t_start = time.time()

for i, pair in enumerate(all_pairs):
    s = all_samples[i]
    t0 = time.time()
    print(f"  [{i+1}/9] [{s['action']:10s}] {pair[:60]} ...")
    sv_i = explainer([pair])
    elapsed = time.time() - t0
    all_sv.append(sv_i)
    print(f"         ✓ done in {elapsed:.0f}s\n")

print(f"Total SHAP time: {(time.time()-t_start)/60:.1f} min\n")

# ── 7. Save immediately ───────────────────────────────────────────────────────
# Merge individual SHAP results into one object for consistency
# Store as list of single-sample results
with open(PKL_OUT, "wb") as f:
    pickle.dump(all_sv, f)
print(f"✓ Saved {PKL_OUT}  ({PKL_OUT.stat().st_size:,} bytes)")

# Verify
with open(PKL_OUT, "rb") as f:
    sv_check = pickle.load(f)
print(f"✓ Verified: {len(sv_check)} items loaded\n")

# ── 8. Print top 10 tokens per sample ────────────────────────────────────────
def clean_token(t: str) -> str:
    return t.replace("▁", "").strip() or "·"

print("=" * 62)
print("TOP 10 TOKENS BY |SHAP VALUE| PER SAMPLE")
print("=" * 62)

for i, (sv_i, s) in enumerate(zip(all_sv, all_samples)):
    sc = float(scores[i][0])
    print(f"\nSample {i+1} [{s['action']}]  score={sc:+.4f}")
    print(f"  Q: {s['question'][:70]}")
    tokens = [clean_token(t) for t in sv_i.data[0]]
    vals   = np.array(sv_i.values[0])
    if vals.ndim == 2:
        vals = vals[:, 0]
    paired = sorted(zip(vals.tolist(), tokens), key=lambda x: abs(x[0]), reverse=True)
    for v, tok in paired[:10]:
        print(f"  {v:+.4f}  '{tok}'")

# ── 9. Plot 3×3 grid ──────────────────────────────────────────────────────────
POS_COLOR = "#d62728"
NEG_COLOR = "#1f77b4"

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(
    "SHAP Token Attributions — T5 Retrieval Evaluator (PopQA)\n9 samples · 3 per action type",
    fontsize=14, fontweight="bold", y=1.01
)

for idx, (sv_i, s) in enumerate(zip(all_sv, all_samples)):
    row, col = divmod(idx, 3)
    ax = axes[row][col]
    sc = float(scores[idx][0])

    tokens = [clean_token(t) for t in sv_i.data[0]]
    vals   = np.array(sv_i.values[0])
    if vals.ndim == 2:
        vals = vals[:, 0]

    # Top 8 by absolute value
    order     = np.argsort(np.abs(vals))[::-1][:8]
    top_vals  = vals[order]
    top_toks  = [tokens[j] for j in order]

    # Sort display: most negative bottom, most positive top
    disp_order = np.argsort(top_vals)
    top_vals   = top_vals[disp_order]
    top_toks   = [top_toks[j] for j in disp_order]

    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in top_vals]
    y_pos  = range(len(top_vals))
    bars   = ax.barh(y_pos, top_vals, color=colors,
                     height=0.65, edgecolor="white", linewidth=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_toks, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)

    for bar, v in zip(bars, top_vals):
        x_off = 0.003 if v >= 0 else -0.003
        ha    = "left" if v >= 0 else "right"
        ax.text(v + x_off, bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center", ha=ha, fontsize=6.5, color="black")

    ax.set_title(f"[{s['action']}]  score={sc:+.4f}\n{s['question'][:45]}",
                 fontsize=8, pad=4)
    ax.set_xlabel("SHAP value", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 1.2, xmax * 1.2)

legend_elements = [
    mpatches.Patch(facecolor=POS_COLOR, label="Positive attribution (↑ relevance score)"),
    mpatches.Patch(facecolor=NEG_COLOR, label="Negative attribution (↓ relevance score)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout()
fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

sz = PLOT_OUT.stat().st_size
assert sz > 20_000, f"Suspiciously small: {PLOT_OUT} ({sz} bytes)"
print(f"\n✓ Saved {PLOT_OUT}  ({sz:,} bytes) — verified non-blank")
print("\nAll done.")
