"""
error_analysis_plot.py
──────────────────────
Publication-ready figures from error_analysis.json.
  1. error_analysis_plot.png       — bar chart: accuracy by question type
  2. error_analysis_action_heatmap.png — heatmap: accuracy by (type × action)
"""

import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
from pathlib import Path

EA_FILE  = Path("error_analysis.json")
PLOT1    = Path("error_analysis_plot.png")
PLOT2    = Path("error_analysis_action_heatmap.png")

# ── Load ──────────────────────────────────────────────────────────────────────
with open(EA_FILE) as f:
    records = json.load(f)

TYPE_ORDER   = ["occupation","genre","country","city","sport",
                "director","composer","author","religion","other"]
ACTION_ORDER = ["CORRECT","AMBIGUOUS","INCORRECT"]
OVERALL_ACC  = 54.4   # pre-computed

# ── Aggregate per type ────────────────────────────────────────────────────────
type_data: dict[str, dict] = {qt: {"total": 0, "matched": 0, "actions": Counter()}
                               for qt in TYPE_ORDER}
# (type, action) → (matched, total)
ta_data: dict[tuple, list] = defaultdict(lambda: [0, 0])

for r in records:
    qt     = r["question_type"]
    action = r["action"]
    match  = r["match"]
    type_data[qt]["total"]   += 1
    type_data[qt]["actions"][action] += 1
    if match is not None:
        type_data[qt]["matched"] += match
        ta_data[(qt, action)][0] += match
        ta_data[(qt, action)][1] += 1

# ── Compute per-type accuracy & dominant action ───────────────────────────────
type_acc     = {}
type_dom_act = {}
for qt in TYPE_ORDER:
    d   = type_data[qt]
    tot = d["total"]
    type_acc[qt]     = d["matched"] / tot * 100 if tot > 0 else 0.0
    type_dom_act[qt] = d["actions"].most_common(1)[0][0] if d["actions"] else "other"

# Sort by accuracy descending
sorted_types = sorted(TYPE_ORDER, key=lambda qt: type_acc[qt], reverse=True)

# ── Colour map ────────────────────────────────────────────────────────────────
ACTION_COLOR = {
    "CORRECT"  : "#2ca02c",   # green
    "AMBIGUOUS": "#ff7f0e",   # orange
    "INCORRECT": "#d62728",   # red
}

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Bar chart
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(12, 6))

xs    = np.arange(len(sorted_types))
accs  = [type_acc[qt]     for qt in sorted_types]
cols  = [ACTION_COLOR[type_dom_act[qt]] for qt in sorted_types]
tots  = [type_data[qt]["total"] for qt in sorted_types]

bars = ax1.bar(xs, accs, color=cols, edgecolor="white", linewidth=0.6, width=0.65)

# Count labels on top of bars
for bar, n, acc in zip(bars, tots, accs):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"n={n}",
        ha="center", va="bottom", fontsize=9, color="#333333"
    )

# Overall accuracy dashed line
ax1.axhline(OVERALL_ACC, color="black", linewidth=1.2, linestyle="--", zorder=3)
ax1.text(len(sorted_types) - 0.5, OVERALL_ACC + 1.2,
         f"Overall {OVERALL_ACC}%", ha="right", fontsize=9, color="black")

# Axes formatting
ax1.set_xticks(xs)
ax1.set_xticklabels(sorted_types, fontsize=11)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_ylim(0, 100)
ax1.set_title("CRAG Accuracy by Question Type on PopQA", fontsize=14, fontweight="bold", pad=12)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
ax1.set_axisbelow(True)

# Legend
legend_patches = [
    mpatches.Patch(color=ACTION_COLOR["CORRECT"],   label="Dominant action: CORRECT"),
    mpatches.Patch(color=ACTION_COLOR["AMBIGUOUS"], label="Dominant action: AMBIGUOUS"),
    mpatches.Patch(color=ACTION_COLOR["INCORRECT"], label="Dominant action: INCORRECT"),
]
ax1.legend(handles=legend_patches, fontsize=9, frameon=False, loc="upper right")

fig1.tight_layout()
fig1.savefig(PLOT1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"✓ Saved {PLOT1}  ({PLOT1.stat().st_size:,} bytes)")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Heatmap (type × action)
# ══════════════════════════════════════════════════════════════════════════════

# Build matrix — rows = sorted_types, cols = ACTION_ORDER
matrix = np.full((len(sorted_types), len(ACTION_ORDER)), np.nan)
annot  = [[""] * len(ACTION_ORDER) for _ in sorted_types]

for i, qt in enumerate(sorted_types):
    for j, a in enumerate(ACTION_ORDER):
        m, n = ta_data[(qt, a)]
        if n > 0:
            acc = m / n * 100
            matrix[i, j] = acc
            annot[i][j]  = f"{acc:.0f}%\n(n={n})"
        else:
            annot[i][j] = "—"

fig2, ax2 = plt.subplots(figsize=(10, 8))

# Colormap: red (0) → yellow (50) → green (100), NaN → light grey
cmap = plt.get_cmap("RdYlGn").copy()
cmap.set_bad(color="#e8e8e8")

masked = np.ma.array(matrix, mask=np.isnan(matrix))
im = ax2.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

# Annotate cells
for i in range(len(sorted_types)):
    for j in range(len(ACTION_ORDER)):
        text  = annot[i][j]
        val   = matrix[i, j]
        tcolor = "white" if (not np.isnan(val) and (val < 25 or val > 75)) else "black"
        ax2.text(j, i, text, ha="center", va="center",
                 fontsize=9, color=tcolor, fontweight="normal")

# Axes
ax2.set_xticks(range(len(ACTION_ORDER)))
ax2.set_xticklabels(ACTION_ORDER, fontsize=12, fontweight="bold")
ax2.set_yticks(range(len(sorted_types)))
ax2.set_yticklabels(sorted_types, fontsize=11)
ax2.set_title("CRAG Accuracy by Question Type × Action (PopQA)",
              fontsize=13, fontweight="bold", pad=12)

# Cell borders
for i in range(len(sorted_types)):
    for j in range(len(ACTION_ORDER)):
        ax2.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="#cccccc", linewidth=0.6
        ))

# Colorbar
cbar = fig2.colorbar(im, ax=ax2, orientation="vertical",
                     pad=0.02, shrink=0.85, label="Accuracy (%)")
cbar.ax.tick_params(labelsize=9)

fig2.tight_layout()
fig2.savefig(PLOT2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"✓ Saved {PLOT2}  ({PLOT2.stat().st_size:,} bytes)")

# ── Verify non-blank ──────────────────────────────────────────────────────────
for p in [PLOT1, PLOT2]:
    sz = p.stat().st_size
    assert sz > 20_000, f"Suspiciously small: {p} ({sz} bytes)"
    print(f"  Verified non-blank: {p.name}  {sz:,} bytes")
