"""
SHAP token attribution for the fine-tuned T5 retrieval evaluator.
No plotting — computation and saving only.

Performance notes:
- T5-large on M2 CPU: ~20s per forward pass regardless of seq length (encoder bottleneck).
- SHAP Text masker masks at word level. Our inputs have 13-15 words each.
- Exact Shapley needs 2*num_words+1 evals. We use max_evals=2*max_words+1 = 31.
- Total: 31 evals * 3 samples * ~20s = ~20 minutes. Acceptable for a smoke test.
- score_fn batches ALL masked texts for a given sample in one call to amortise
  Python overhead (SHAP calls score_fn once per permutation by default, but the
  Exact explainer batches internally).
"""

import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap

CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "finetuned_t5_evaluator")

# ── 1. Load model ─────────────────────────────────────────────────────────────
print(f"Loading tokenizer from {CHECKPOINT} ...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)

print(f"Loading model from {CHECKPOINT} ...")
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=1,
    ignore_mismatched_sizes=True,
)
model.eval()
device = "cpu"
model.to(device)
print("✓ Model loaded on cpu.\n")

# ── 2. Define and test score_fn ───────────────────────────────────────────────
# score_fn receives a list of strings and returns (N, 1) float32 numpy array.
# We batch-tokenize and run inference in one shot to minimise overhead.
def score_fn(texts):
    if len(texts) == 0:
        return np.array([], dtype=np.float32).reshape(0, 1)
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,           # pad to longest in this batch (not global max)
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        out = model(
            enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
    scores = out.logits.cpu().numpy().astype(np.float32)   # (N, 1)
    return scores

test_pairs = [
    "What is Henry Feilden's occupation? [SEP] Henry Master Feilden was an English Conservative Party politician.",
    "What is Henry Feilden's occupation? [SEP] The mitochondria is the powerhouse of the cell.",
    "Who directed Titanic? [SEP] Titanic is a 1997 film directed by James Cameron.",
]

print("── score_fn smoke test ──")
test_scores = score_fn(test_pairs)
for text, score in zip(test_pairs, test_scores):
    print(f"  {score[0]:+.4f}  {text[:80]}")
print()

# ── 3. Run SHAP ───────────────────────────────────────────────────────────────
# Compute max words across all samples → set max_evals = 2*max_words + 1
# so we get exact Shapley values, not approximations.
max_words = max(len(p.split()) for p in test_pairs)
MAX_EVALS = 2 * max_words + 1
print(f"max words across samples: {max_words}  →  max_evals={MAX_EVALS}")
print(f"Estimated time: {MAX_EVALS * 3 * 20 / 60:.0f}-{MAX_EVALS * 3 * 30 / 60:.0f} min on M2 CPU")
print()

print("Building SHAP explainer ...")
masker = shap.maskers.Text(tokenizer, mask_token="[MASK]", collapse_mask_token=True)
explainer = shap.Explainer(score_fn, masker, max_evals=MAX_EVALS)

print("Running shap_values = explainer(test_pairs) ...")
print("Progress bar will appear below:\n")
shap_values = explainer(test_pairs)
print("\n✓ SHAP computation done.\n")

# ── 4. Save ───────────────────────────────────────────────────────────────────
pkl_path = os.path.join(os.path.dirname(__file__), "shap_values.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(shap_values, f)
print(f"✓ Saved shap_values.pkl")

# ── 5. Verify load ────────────────────────────────────────────────────────────
with open(pkl_path, "rb") as f:
    sv = pickle.load(f)
print(f"✓ Verified: {len(sv)} samples\n")

# ── 6. Top-15 tokens by absolute SHAP value per sample ───────────────────────
sample_labels = [
    "Sample 1 — Feilden (RELEVANT doc)",
    "Sample 2 — Feilden (IRRELEVANT doc)",
    "Sample 3 — Titanic (RELEVANT doc)",
]

print("=" * 60)
print("TOP 15 TOKENS BY |SHAP VALUE| PER SAMPLE")
print("=" * 60)

for i, label in enumerate(sample_labels):
    print(f"\n{label}")
    print(f"  Model score: {test_scores[i][0]:+.4f}")
    base = sv.base_values[i]
    print(f"  Base value : {float(np.array(base).flat[0]):+.4f}")
    print()

    tokens = sv.data[i]      # list of word strings for this sample
    values = sv.values[i]    # (num_words, 1) or (num_words,)

    vals = np.array(values)
    if vals.ndim == 2:
        vals = vals[:, 0]

    paired = sorted(zip(vals.tolist(), tokens), key=lambda x: abs(x[0]), reverse=True)

    shown = min(15, len(paired))
    for shap_val, token in paired[:shown]:
        display_token = token.replace("\u2581", " ").strip()
        print(f"  {shap_val:+.4f}  '{display_token}'")
