"""
shap_incorrect.py
─────────────────
Run SHAP on 3 hardcoded INCORRECT samples, merge with existing 6 samples
from shap_values_9.pkl, and regenerate shap_9_summary.png with all 9 panels.
"""

import matplotlib
matplotlib.use("Agg")

import json, pickle, time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap

CKPT        = Path("models/finetuned_t5_evaluator")
PKL_6       = Path("shap_values_9.pkl")
PKL_INC     = Path("shap_incorrect_3.pkl")
PLOT_OUT    = Path("shap_9_summary.png")

# ── Hardcoded INCORRECT samples ───────────────────────────────────────────────
INCORRECT_SAMPLES = [
    {
        "question": "What is John Finlay's occupation?",
        "action"  : "INCORRECT",
        "pair"    : "What is John Finlay's occupation? [SEP] The mitochondria is the powerhouse of the cell.",
    },
    {
        "question": "Who directed Titanic?",
        "action"  : "INCORRECT",
        "pair"    : "Who directed Titanic? [SEP] Paris is the capital city of France located in Western Europe.",
    },
    {
        "question": "What is the religion of Alfred Reid?",
        "action"  : "INCORRECT",
        "pair"    : "What is the religion of Alfred Reid? [SEP] Photosynthesis is the process by which plants convert sunlight into energy.",
    },
]
incorrect_pairs = [s["pair"] for s in INCORRECT_SAMPLES]

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(str(CKPT), use_fast=False)

print("Loading model ...")
model = AutoModelForSequenceClassification.from_pretrained(
    str(CKPT), num_labels=1, ignore_mismatched_sizes=True
)
model.eval()
model.to("cpu")
print("✓ Model loaded.\n")

# ── score_fn ──────────────────────────────────────────────────────────────────
def score_fn(texts):
    if len(texts) == 0:
        return np.array([], dtype=np.float32).reshape(0, 1)
    enc = tokenizer(
        list(texts), return_tensors="pt",
        padding=True, truncation=True, max_length=512,
    )
    with torch.no_grad():
        out = model(enc["input_ids"], attention_mask=enc["attention_mask"])
    return out.logits.cpu().numpy().astype(np.float32)

# ── Verify scores ─────────────────────────────────────────────────────────────
print("── Score verification (should be << 0) ──")
scores_inc = score_fn(incorrect_pairs)
for s, sc_row in zip(INCORRECT_SAMPLES, scores_inc):
    sc = float(sc_row[0])
    flag = "  ✓" if sc < -0.3 else "  ⚠ higher than expected"
    print(f"  score={sc:+.4f}{flag}")
    print(f"         {s['pair'][:80]}")
print()

# ── SHAP on 3 INCORRECT samples ───────────────────────────────────────────────
max_words = max(len(p.split()) for p in incorrect_pairs)
MAX_EVALS = 2 * max_words + 1
print(f"max words : {max_words}  →  max_evals={MAX_EVALS}")
print(f"Estimated : {MAX_EVALS * 3 * 20 / 60:.0f}–{MAX_EVALS * 3 * 30 / 60:.0f} min on M2 CPU\n")

masker    = shap.maskers.Text(tokenizer, mask_token="[MASK]", collapse_mask_token=True)
explainer = shap.Explainer(score_fn, masker, max_evals=MAX_EVALS)

print("Running SHAP on 3 INCORRECT samples:\n")
sv_incorrect = []
t_start = time.time()
for i, pair in enumerate(incorrect_pairs):
    t0 = time.time()
    print(f"  [{i+1}/3] {pair[:70]} ...")
    sv_i = explainer([pair])
    sv_incorrect.append(sv_i)
    print(f"         ✓ done in {time.time()-t0:.0f}s\n")
print(f"Total SHAP time: {(time.time()-t_start)/60:.1f} min\n")

# ── Save INCORRECT pkl ────────────────────────────────────────────────────────
with open(PKL_INC, "wb") as f:
    pickle.dump(sv_incorrect, f)
print(f"✓ Saved {PKL_INC}  ({PKL_INC.stat().st_size:,} bytes)")

with open(PKL_INC, "rb") as f:
    chk = pickle.load(f)
print(f"✓ Verified: {len(chk)} items\n")

# ── Load existing 6 samples ───────────────────────────────────────────────────
print(f"Loading existing 6 samples from {PKL_6} ...")
with open(PKL_6, "rb") as f:
    sv_6 = pickle.load(f)
print(f"✓ Loaded {len(sv_6)} items\n")

# ── Rebuild metadata for all 9 samples ───────────────────────────────────────
# Original 6: 3 CORRECT then 3 AMBIGUOUS (from shap_9samples.py)
CRAG_FILE = Path("crag_results.json")
with open(CRAG_FILE) as f:
    crag = json.load(f)

ACTION_ORDER = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
N_PER = 3

def first_n_words(text, n=15):
    return " ".join(text.split()[:n])

selected = {a: [] for a in ["CORRECT", "AMBIGUOUS"]}
for r in crag:
    action  = r.get("action", "")
    context = r.get("context", "").strip()
    if not context or action not in selected:
        continue
    if len(selected[action]) < N_PER:
        doc  = first_n_words(context, 15)
        pair = f"{r['question']} [SEP] {doc}"
        selected[action].append({
            "question": r["question"],
            "action"  : action,
            "pair"    : pair,
        })

# Scores for original 6
scores_6 = score_fn([s["pair"] for a in ["CORRECT","AMBIGUOUS"] for s in selected[a]])

all_samples = (
    selected["CORRECT"] +
    selected["AMBIGUOUS"] +
    INCORRECT_SAMPLES
)
all_sv = sv_6 + sv_incorrect
all_scores = list(scores_6) + list(scores_inc)

# ── Print top 10 for the 3 new INCORRECT samples ─────────────────────────────
def clean_token(t):
    return t.replace("▁", "").strip() or "·"

print("=" * 62)
print("TOP 10 TOKENS — INCORRECT SAMPLES")
print("=" * 62)
for i, (sv_i, s, sc_row) in enumerate(zip(sv_incorrect, INCORRECT_SAMPLES, scores_inc)):
    sc = float(sc_row[0])
    print(f"\nSample {i+1} [INCORRECT]  score={sc:+.4f}")
    print(f"  Q: {s['question']}")
    tokens = [clean_token(t) for t in sv_i.data[0]]
    vals   = np.array(sv_i.values[0])
    if vals.ndim == 2:
        vals = vals[:, 0]
    paired = sorted(zip(vals.tolist(), tokens), key=lambda x: abs(x[0]), reverse=True)
    for v, tok in paired[:10]:
        print(f"  {v:+.4f}  '{tok}'")

# ── Regenerate shap_9_summary.png ─────────────────────────────────────────────
POS_COLOR = "#d62728"
NEG_COLOR = "#1f77b4"

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(
    "SHAP Token Attributions — T5 Retrieval Evaluator (PopQA)\n"
    "9 samples · 3 per action type (CORRECT / AMBIGUOUS / INCORRECT)",
    fontsize=14, fontweight="bold", y=1.01
)

for idx, (sv_i, s, sc_row) in enumerate(zip(all_sv, all_samples, all_scores)):
    row, col = divmod(idx, 3)
    ax = axes[row][col]
    sc = float(sc_row[0]) if hasattr(sc_row, '__len__') else float(sc_row)

    tokens = [clean_token(t) for t in sv_i.data[0]]
    vals   = np.array(sv_i.values[0])
    if vals.ndim == 2:
        vals = vals[:, 0]

    order     = np.argsort(np.abs(vals))[::-1][:8]
    top_vals  = vals[order]
    top_toks  = [tokens[j] for j in order]

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
print("All done.")
