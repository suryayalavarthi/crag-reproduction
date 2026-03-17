"""
wikipedia_ambiguous_v2.py
─────────────────────────
Improved Wikipedia search for AMBIGUOUS-action CRAG questions.
Target hit-rate: ~80%+  (v1 baseline: 63.3%)

Five-stage fallback chain per question
────────────────────────────────────────
  direct   — plain entity name direct page lookup
  typed    — entity + type suffix ("album", "film", "footballer", …)
  search   — wikipedia.search(entity, results=3) → fetch top result
  disambig — if direct hit is a disambiguation page, fetch its first link
  hint     — entity + parenthetical descriptor mined from CRAG context
"""

import subprocess, sys

for pkg, imp in [("Wikipedia-API", "wikipediaapi"), ("wikipedia", "wikipedia")]:
    try:
        __import__(imp)
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import json
import re
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

import wikipedia as wiki_pkg          # for .search()
import wikipediaapi                   # for .page() — cleaner API / better Unicode

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
IN_FILE          = BASE_DIR / "crag_results.json"
OUT_FILE         = BASE_DIR / "ambiguous_with_wiki_v2.json"
CKPT_FILE        = BASE_DIR / "ambiguous_wiki_v2_checkpoint.json"
CHECKPOINT_EVERY = 50
PROGRESS_EVERY   = 20
TIMEOUT_SEC      = 6

wiki_api = wikipediaapi.Wikipedia(user_agent="CRAGResearch/1.0", language="en")
wiki_pkg.set_lang("en")

# ── helpers ───────────────────────────────────────────────────────────────────
def _is_cap(w: str) -> bool:
    return bool(w) and unicodedata.category(w[0]) == "Lu"

_STOPWORDS = {
    "what", "who", "where", "when", "why", "how", "the", "a", "an",
    "is", "was", "are", "were", "did", "do", "does", "in", "of",
    "for", "by", "at", "on", "to",
}

def _first_n_sentences(text: str, n: int = 3) -> str:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return " ".join(parts[:n]).strip()

def _is_disambig(summary: str) -> bool:
    low = summary.lower()
    return ("may refer to" in low or "can refer to" in low
            or "refers to" in low) and len(summary) < 300

# ── entity extraction ─────────────────────────────────────────────────────────
def extract_entity(question: str) -> str:
    """Return the primary entity string from the question."""
    # Strip possessives and trailing punctuation
    q = question.rstrip("?")
    q = re.sub(r"'s$", "", q)

    # "What is X's occupation" → X
    m = re.match(r"(?:what is|what was)\s+(.+?)'s\s+\w+", q, re.I)
    if m:
        return m.group(1).strip()

    # "Who was the DIR/COMP/SCREEN/PROD of TITLE"
    m = re.match(
        r"who was (?:the\s+)?(?:director|composer|screenwriter|producer|author) of (.+)",
        q, re.I)
    if m:
        return m.group(1).strip()

    # "Who directed/composed/wrote TITLE"
    m = re.match(r"who (?:directed|composed|wrote|created|produced) (.+)", q, re.I)
    if m:
        return m.group(1).strip()

    # "What genre is TITLE"
    m = re.match(r"what genre is (.+)", q, re.I)
    if m:
        return m.group(1).strip()

    # "In what city/country was/is X …"
    m = re.match(r"in what (?:city|country|state|region)\s+(?:was|is)\s+(.+?)(?:\s+born)?$", q, re.I)
    if m:
        return m.group(1).strip()

    # "What sport does X play"
    m = re.match(r"what sport does (.+?) play", q, re.I)
    if m:
        return m.group(1).strip()

    # "What is the religion/nationality of X"
    m = re.match(r"what is the \w+ of (.+)", q, re.I)
    if m:
        return m.group(1).strip()

    # "Who is the father/mother of X"
    m = re.match(r"who is the \w+ of (.+)", q, re.I)
    if m:
        return m.group(1).strip()

    # Generic: collect runs of capitalised words, skipping leading wh-word
    words = q.split()
    start = 1 if (words and words[0].lower() in _STOPWORDS) else 0
    run = []
    for w in words[start:]:
        clean = w.strip(".,;:()")
        if _is_cap(clean) and clean.lower() not in _STOPWORDS:
            run.append(clean)
        elif run:
            break
    if run:
        return " ".join(run)

    # Last resort: strip wh-word
    return re.sub(r"^(what|who|where|when|why|how)\b\s*", "", q,
                  flags=re.I).strip()


def typed_suffixes(question: str) -> list[str]:
    """Return type-specific suffixes to try for the entity."""
    q = question.lower()
    if "occupation" in q or "religion" in q or "nationality" in q or "born" in q:
        return ["politician", "footballer", "musician", "actor", "singer",
                "bishop", "author", "athlete"]
    if "genre" in q:
        return ["album", "film", "song", "EP", "soundtrack", "book"]
    if "director" in q or "directed" in q:
        return ["film", "movie"]
    if "composer" in q or "composed" in q:
        return ["film", "album", "opera"]
    if "screenwriter" in q or "author" in q or "wrote" in q:
        return ["film", "novel", "book"]
    if "producer" in q or "produced" in q:
        return ["film", "album"]
    if "country" in q:
        return ["municipality", "city", "town", "village"]
    if "city" in q:
        return ["city", "town", "municipality"]
    if "sport" in q:
        return ["football club", "basketball team", "soccer team",
                "hockey league", "sports club"]
    return []


def hint_disambig(hint: str) -> str:
    """Extract a parenthetical descriptor from CRAG context to disambiguate."""
    m = re.search(r'\(([^)]{3,40})\)', hint)
    if m:
        # e.g. "(Conservative politician)" → "Conservative politician"
        # Take just the last meaningful word as it's more specific
        words = [w for w in m.group(1).split() if len(w) > 3]
        if words:
            return words[-1].lower()
    return ""


# ── single-page fetch (threaded) ──────────────────────────────────────────────
def _page_fetch(title: str, holder: list):
    try:
        p = wiki_api.page(title)
        if p.exists() and p.summary:
            holder.append((p.summary, p.fullurl))
        else:
            holder.append(None)
    except Exception:
        holder.append(None)


def _fetch_page(title: str) -> tuple[str, str] | None:
    """Fetch a Wikipedia page with timeout. Returns (summary, url) or None."""
    holder: list = []
    t = threading.Thread(target=_page_fetch, args=(title, holder), daemon=True)
    t.start()
    t.join(timeout=TIMEOUT_SEC)
    if holder and holder[0] is not None:
        summary, url = holder[0]
        return summary, url
    return None


def _search_fetch(query: str) -> tuple[str, str, str] | None:
    """
    Use wikipedia.search() to find candidate titles, then fetch the first
    non-disambig result. Returns (summary, url, page_title) or None.
    """
    holder: list = []

    def _do():
        try:
            titles = wiki_pkg.search(query, results=4)
            for title in titles:
                p = wiki_api.page(title)
                if p.exists() and p.summary and not _is_disambig(p.summary):
                    holder.append((p.summary, p.fullurl, title))
                    return
            holder.append(None)
        except Exception:
            holder.append(None)

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=TIMEOUT_SEC * 2)   # search + fetch takes longer
    if holder and holder[0] is not None:
        return holder[0]
    return None


def _disambig_first_link(summary: str, url: str) -> tuple[str, str] | None:
    """
    Given a disambiguation page summary, extract the first article title
    mentioned and try to fetch it.
    """
    # Pull all titles within parentheses that look like Wikipedia links
    # e.g. "John Finlay (footballer, born 1995)" → try "John Finlay (footballer)"
    links = re.findall(r'([A-Z][^,\n]+?)\s*\(([^)]+)\)', summary)
    for base, qual in links[:4]:
        title = f"{base.strip()} ({qual.strip()})"
        result = _fetch_page(title)
        if result:
            s, u = result
            if not _is_disambig(s) and len(s) > 60:
                return s, u
    return None


# ── main search function ──────────────────────────────────────────────────────
def search_wikipedia_v2(question: str, hint: str = "") -> tuple[str, str, str]:
    """
    Returns (wiki_3_sentences, url, strategy_name).
    strategy_name ∈ {"direct", "typed", "search", "disambig", "hint", ""}
    """
    entity  = extract_entity(question)
    suffixes = typed_suffixes(question)
    hint_d  = hint_disambig(hint)

    # ── Stage 1: direct ───────────────────────────────────────────────────────
    result = _fetch_page(entity)
    if result:
        summary, url = result
        if not _is_disambig(summary) and len(summary) > 60:
            return _first_n_sentences(summary), url, "direct"

        # Stage 4: disambig — use first meaningful link from the disambig page
        disambig_result = _disambig_first_link(summary, url)
        if disambig_result:
            s, u = disambig_result
            return _first_n_sentences(s), u, "disambig"

    # ── Stage 2: typed suffixes ───────────────────────────────────────────────
    for suffix in suffixes[:4]:
        typed_title = f"{entity} ({suffix})"
        result = _fetch_page(typed_title)
        if result:
            s, u = result
            if not _is_disambig(s) and len(s) > 60:
                return _first_n_sentences(s), u, "typed"
        # Also try plain "entity suffix" without parentheses (e.g. "Drive On album")
        plain_typed = f"{entity} {suffix}"
        result = _fetch_page(plain_typed)
        if result:
            s, u = result
            if not _is_disambig(s) and len(s) > 60:
                return _first_n_sentences(s), u, "typed"

    # ── Stage 3: wikipedia.search() ──────────────────────────────────────────
    result = _search_fetch(entity)
    if result:
        s, u, _ = result
        return _first_n_sentences(s), u, "search"

    # Also try search with top typed suffix
    if suffixes:
        result = _search_fetch(f"{entity} {suffixes[0]}")
        if result:
            s, u, _ = result
            return _first_n_sentences(s), u, "search"

    # ── Stage 5: hint disambiguator ───────────────────────────────────────────
    if hint_d:
        titled = f"{entity} ({hint_d})"
        result = _fetch_page(titled)
        if result:
            s, u = result
            if not _is_disambig(s) and len(s) > 60:
                return _first_n_sentences(s), u, "hint"
        # Also try search with hint word appended
        result = _search_fetch(f"{entity} {hint_d}")
        if result:
            s, u, _ = result
            return _first_n_sentences(s), u, "hint"

    return "", "", ""


# ── load data ─────────────────────────────────────────────────────────────────
print(f"Loading {IN_FILE} ...")
with open(IN_FILE) as f:
    all_results = json.load(f)

ambiguous = [r for r in all_results if r["action"] == "AMBIGUOUS"]
print(f"Total records : {len(all_results)}")
print(f"AMBIGUOUS     : {len(ambiguous)}\n")

# ── main loop ─────────────────────────────────────────────────────────────────
results_out: list[dict] = []
strategy_counts: Counter = Counter()
got_context   = 0
empty_context = 0

print(f"Processing {len(ambiguous)} AMBIGUOUS questions ...\n")
start_time = time.time()

for idx, record in enumerate(ambiguous):
    question = record["question"]
    hint     = record.get("context", "")[:150]

    wiki_text, wiki_url, strategy = search_wikipedia_v2(question, hint)

    strategy_counts[strategy if strategy else "miss"] += 1
    if wiki_text:
        got_context += 1
    else:
        empty_context += 1

    results_out.append({
        "question"        : question,
        "original_context": record.get("context", ""),
        "wiki_context"    : wiki_text,
        "wiki_url"        : wiki_url,
        "wiki_source"     : strategy,
        "action"          : "AMBIGUOUS",
    })

    # Progress
    if (idx + 1) % PROGRESS_EVERY == 0 or idx == 0:
        elapsed = time.time() - start_time
        rate    = (idx + 1) / max(elapsed, 0.001)
        eta     = (len(ambiguous) - idx - 1) / rate
        pct     = got_context / (idx + 1) * 100
        print(f"  [{idx+1:3d}/{len(ambiguous)}]  hit={got_context}  miss={empty_context}"
              f"  rate={pct:.0f}%  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        print(f"         Q : {question[:70]}")
        if wiki_text:
            print(f"         W : {wiki_text[:90]}  [{strategy}]")
        else:
            entity = extract_entity(question)
            print(f"         W : (no result)  entity={entity!r}")
        print()

    # Checkpoint
    if (idx + 1) % CHECKPOINT_EVERY == 0:
        with open(CKPT_FILE, "w") as f:
            json.dump(results_out, f, indent=2)
        print(f"  ✓ Checkpoint saved ({idx+1} done)\n")

# ── save ──────────────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as f:
    json.dump(results_out, f, indent=2)

elapsed_total = time.time() - start_time
print("=" * 60)
print(f"Done.  {elapsed_total:.0f}s  ({elapsed_total/60:.1f} min)")
print(f"  Processed  : {len(results_out)}")
print(f"  Hit        : {got_context}  ({got_context/len(results_out)*100:.1f}%)")
print(f"  Miss       : {empty_context}  ({empty_context/len(results_out)*100:.1f}%)")
print(f"\nBreakdown by strategy:")
for strat, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
    pct = cnt / len(results_out) * 100
    print(f"  {strat:12s}: {cnt:3d}  ({pct:.1f}%)")
print(f"\n  Output → {OUT_FILE}")
