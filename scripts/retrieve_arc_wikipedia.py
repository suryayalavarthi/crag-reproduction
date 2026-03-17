"""
retrieve_arc_wikipedia.py
─────────────────────────
Retrieve Wikipedia documents for ARC-Challenge questions.
Output: arc_challenge_with_docs.json
"""

import subprocess, sys

for pkg, imp in [("Wikipedia-API", "wikipediaapi"), ("wikipedia", "wikipedia")]:
    try:
        __import__(imp)
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import json, re, threading, time
from collections import Counter
from pathlib import Path

import wikipedia as wiki_pkg
import wikipediaapi

IN_FILE   = Path("arc_challenge_test.json")
OUT_FILE  = Path("arc_challenge_with_docs.json")
CKPT_FILE = Path("arc_wiki_checkpoint.json")

CHECKPOINT_EVERY = 100
PROGRESS_EVERY   = 50
TIMEOUT_SEC      = 5

wiki_api = wikipediaapi.Wikipedia(user_agent="CRAGResearch/1.0", language="en")
wiki_pkg.set_lang("en")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print(f"Loading {IN_FILE} ...")
with open(IN_FILE) as f:
    records = json.load(f)
print(f"  {len(records)} questions loaded\n")

# ── 2. Extract bare question + key terms ──────────────────────────────────────
_STOPWORDS = {
    "what","which","who","where","when","why","how","the","a","an","of","in",
    "and","or","is","are","was","were","be","been","being","to","for","on",
    "at","by","as","with","from","that","this","these","those","their","its",
    "it","do","does","did","not","most","likely","best","following","result",
    "would","will","can","could","should","have","has","had","each","all",
    "both","between","than","more","less","some","about","into","through",
    "during","before","after","above","below","up","down","out","off","over",
    "under","again","further","then","once","if","while","although","because",
    "they","them","he","she","we","you","his","her","our","your","my","new",
    "used","using","use","effect","cause","type","kind","example","part",
    "made","make","help","helps","found","called","known","given","due",
    "able","needed","process","system","increased","decreased","change",
    "changes","different","same","similar","many","also","only","just",
    "first","second","third","number","amount","large","small","high","low",
}

def bare_question(formatted_q: str) -> str:
    """Strip 'Question: ' prefix and choice text."""
    q = re.sub(r"^Question:\s*", "", formatted_q)
    q = re.sub(r"\s{2,}[A-E]\).*", "", q)   # remove "  A) ..." onwards
    return q.strip()

def extract_terms(question: str) -> list[str]:
    """Return key content words for Wikipedia search."""
    words = re.findall(r"[A-Za-z][a-z]*(?:'s)?", question)
    terms = []
    for w in words:
        clean = w.rstrip("'s").strip()
        if len(clean) > 3 and clean.lower() not in _STOPWORDS:
            terms.append(clean)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return unique[:4]   # top 4 content words

# ── helpers (same pattern as wikipedia_ambiguous_v2.py) ──────────────────────
def _first_n_sentences(text: str, n: int = 3) -> str:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return " ".join(parts[:n]).strip()

def _is_disambig(summary: str) -> bool:
    low = summary.lower()
    return ("may refer to" in low or "can refer to" in low or
            "refers to" in low) and len(summary) < 400

def _fetch_page(title: str) -> tuple[str, str] | None:
    holder: list = []
    def _do():
        try:
            p = wiki_api.page(title)
            if p.exists() and p.summary:
                holder.append((p.summary, p.fullurl))
            else:
                holder.append(None)
        except Exception:
            holder.append(None)
    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=TIMEOUT_SEC)
    if holder and holder[0]:
        return holder[0]
    return None

def _search_fetch(query: str) -> tuple[str, str] | None:
    holder: list = []
    def _do():
        try:
            titles = wiki_pkg.search(query, results=3)
            for title in titles:
                p = wiki_api.page(title)
                if p.exists() and p.summary and not _is_disambig(p.summary):
                    holder.append((p.summary, p.fullurl))
                    return
            holder.append(None)
        except Exception:
            holder.append(None)
    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=TIMEOUT_SEC * 2)
    if holder and holder[0]:
        return holder[0]
    return None

# ── 3. Main search function ───────────────────────────────────────────────────
def search_for_question(formatted_q: str) -> tuple[str, str, str]:
    """Returns (doc_text, url, strategy). strategy='' if miss."""
    q     = bare_question(formatted_q)
    terms = extract_terms(q)
    query = " ".join(terms[:3])

    # Stage 1: direct lookup with top terms joined
    if query:
        result = _fetch_page(query)
        if result:
            s, u = result
            if not _is_disambig(s) and len(s) > 60:
                return _first_n_sentences(s), u, "direct"

    # Stage 2: direct lookup with first 2 terms
    if len(terms) >= 2:
        result = _fetch_page(" ".join(terms[:2]))
        if result:
            s, u = result
            if not _is_disambig(s) and len(s) > 60:
                return _first_n_sentences(s), u, "direct2"

    # Stage 3: wikipedia.search() with full query
    if query:
        result = _search_fetch(query)
        if result:
            s, u = result
            return _first_n_sentences(s), u, "search"

    # Stage 4: wikipedia.search() with bare question
    result = _search_fetch(q[:80])
    if result:
        s, u = result
        return _first_n_sentences(s), u, "search_q"

    return "", "", ""

# ── 4. Main loop ──────────────────────────────────────────────────────────────
results_out: list[dict] = []
strategy_counts: Counter = Counter()
got, miss = 0, 0

print(f"Retrieving Wikipedia docs for {len(records)} questions ...\n")
start_time = time.time()

for idx, rec in enumerate(records):
    doc, url, strategy = search_for_question(rec["question"])

    strategy_counts[strategy if strategy else "miss"] += 1
    if doc:
        got += 1
    else:
        miss += 1

    results_out.append({
        "id"          : rec["id"],
        "question"    : rec["question"],
        "answer_key"  : rec["answer_key"],
        "correct_text": rec["correct_text"],
        "retrieved_doc": doc,
        "wiki_url"    : url,
        "wiki_source" : strategy,
    })

    # Progress
    if (idx + 1) % PROGRESS_EVERY == 0 or idx == 0:
        elapsed = time.time() - start_time
        rate    = (idx + 1) / max(elapsed, 0.001)
        eta     = (len(records) - idx - 1) / rate
        pct     = got / (idx + 1) * 100
        print(f"  [{idx+1:4d}/{len(records)}]  hit={got}  miss={miss}"
              f"  rate={pct:.0f}%  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        q_bare = bare_question(rec["question"])
        print(f"          Q : {q_bare[:70]}")
        if doc:
            print(f"          W : {doc[:80]}  [{strategy}]")
        else:
            print(f"          W : (no result)  terms={extract_terms(q_bare)}")
        print()

    # Checkpoint
    if (idx + 1) % CHECKPOINT_EVERY == 0:
        with open(CKPT_FILE, "w") as f:
            json.dump(results_out, f, indent=2)
        print(f"  ✓ Checkpoint saved ({idx+1} done)\n")

# ── 5. Save final ─────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as f:
    json.dump(results_out, f, indent=2)

elapsed_total = time.time() - start_time
print("=" * 62)
print(f"Done.  {elapsed_total:.0f}s  ({elapsed_total/60:.1f} min)")
print(f"  Processed : {len(results_out)}")
print(f"  Hit       : {got}  ({got/len(results_out)*100:.1f}%)")
print(f"  Miss      : {miss}  ({miss/len(results_out)*100:.1f}%)")
print(f"\nBreakdown by strategy:")
for strat, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
    print(f"  {strat:12s}: {cnt:4d}  ({cnt/len(results_out)*100:.1f}%)")
print(f"\n  Output → {OUT_FILE}  ({OUT_FILE.stat().st_size:,} bytes)")
