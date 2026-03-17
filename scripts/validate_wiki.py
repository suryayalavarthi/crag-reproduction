"""
validate_wiki.py
────────────────
Post-filter ambiguous_with_wiki_v2.json:
  - Reject contexts where the entity name is absent
  - Reject remaining disambiguation pages
  - Reject generic concept pages
  - Keep everything else (including short but entity-relevant entries)

Output: ambiguous_with_wiki_v2_clean.json
"""

import json
import re
import unicodedata
from pathlib import Path

IN_FILE  = Path("ambiguous_with_wiki_v2.json")
OUT_FILE = Path("ambiguous_with_wiki_v2_clean.json")

# ── entity extraction (same logic as v2 script) ───────────────────────────────
def extract_entity(question: str) -> str:
    q = question.rstrip("?")
    q = re.sub(r"'s$", "", q)
    for pat, grp in [
        (r"(?:what is|what was)\s+(.+?)'s\s+\w+", 1),
        (r"who was (?:the\s+)?(?:director|composer|screenwriter|producer|author) of (.+)", 1),
        (r"what genre is (.+)", 1),
        (r"in what (?:city|country|state|region)\s+(?:was|is)\s+(.+?)(?:\s+born)?$", 1),
        (r"what sport does (.+?) play", 1),
        (r"what is the \w+ of (.+)", 1),
        (r"who is the \w+ of (.+)", 1),
        (r"who (?:directed|composed|wrote|created|produced) (.+)", 1),
    ]:
        m = re.match(pat, q, re.I)
        if m:
            return m.group(grp).strip()
    return q


def significant_words(entity: str) -> list[str]:
    """
    Return the meaningful (non-stop, len>2) words from the entity,
    preserving Unicode capital letters.
    Handles transliteration: 'Franghískos' ↔ 'Frangkiskos' won't match by substring,
    so we also return individual letters for short entities.
    """
    _SKIP = {"the", "a", "an", "of", "in", "and", "or", "de", "van", "von"}
    words = []
    for w in entity.split():
        # strip punctuation brackets etc.
        clean = re.sub(r"[^a-zA-ZÀ-ÿ0-9]", "", w)
        if len(clean) > 2 and clean.lower() not in _SKIP:
            words.append(clean)
    return words


# ── disambiguation / generic page detectors ──────────────────────────────────
_DISAMBIG_RE = re.compile(
    r"(may refer to|can refer to|refers to|also refer to"
    r"|is an? (album|film|song|ep|record|single|studio album|compilation|"
    r"collection of audio|live album)"  # catch generic "An album is..."
    r"|is a (device|practice|concept|term|"
    r"state of being|common|perennial|rural village in the|"
    r"right tributary|populated place"
    r"))",
    re.I
)

# Patterns that indicate the context is about a *general concept*, not the entity
_GENERIC_CONCEPT_RE = re.compile(
    r"^(a gift (or present )?is |"
    r"death is the (end|irreversible)|"
    r"an album is a collection|"
    r"heaven[,.]? (or the heavens[,.]?)? is a common|"
    r"an empire is a realm|"
    r"hunting is the human practice|"
    r"(in physics|in mathematics)|"
    r"joy is the state of being|"
    r"reminiscence is the act|"
    r"revelation[,.]? in religion|"
    r"the provisional irish republican army"
    r")",
    re.I
)


def validate(record: dict) -> tuple[bool, str]:
    """
    Returns (is_valid, reason_if_rejected).
    """
    wc = record.get("wiki_context", "")

    # ── 1. Empty (already blank) ───────────────────────────────────────────────
    if not wc or not wc.strip():
        return False, "empty"

    # ── 2. Generic concept page ────────────────────────────────────────────────
    if _GENERIC_CONCEPT_RE.search(wc):
        return False, "generic_concept"

    # ── 3. Disambiguation page that slipped through ────────────────────────────
    # Covers both short ("Sisters or The Sisters may also refer to:") and
    # long ("Progression may refer to:\nIn mathematics: Arithmetic progression...")
    first_200 = wc[:200]
    if re.search(r"(may|can|also)\s+refer to", first_200, re.I):
        return False, "disambiguation_page"
    if re.search(r"\brefers? to\b.{0,30}(following|below|places|:\s*\n)", first_200, re.I):
        return False, "disambiguation_page"

    # ── 4. Entity name absent from context ────────────────────────────────────
    entity = extract_entity(record["question"])
    sig_words = significant_words(entity)

    # For single-word entities (e.g. "Solo", "Vera") we require the exact word
    # For multi-word entities we require at least ONE significant word to appear
    if sig_words:
        wc_lower = wc.lower()
        found = any(w.lower() in wc_lower for w in sig_words)

        # Special case: transliterated names (Unicode chars in entity but ASCII in wiki)
        # Fall back to checking first 3 letters of each word
        if not found:
            for w in sig_words:
                prefix = w[:4].lower()
                if len(prefix) >= 3 and prefix in wc_lower:
                    found = True
                    break

        if not found:
            return False, "entity_absent"

    # ── 5. Passes all checks ───────────────────────────────────────────────────
    return True, ""


# ── load & validate ───────────────────────────────────────────────────────────
with open(IN_FILE) as f:
    data = json.load(f)

passed = 0
rejected = 0
rejection_reasons: dict[str, list[dict]] = {}
results_out = []

for record in data:
    is_valid, reason = validate(record)

    if is_valid:
        results_out.append(record)
        passed += 1
    else:
        out_rec = dict(record)
        out_rec["wiki_context"] = ""
        out_rec["wiki_url"]     = ""
        out_rec["wiki_source"]  = "rejected"
        results_out.append(out_rec)
        rejected += 1
        rejection_reasons.setdefault(reason, []).append(record)

# ── save ──────────────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as f:
    json.dump(results_out, f, indent=2)

# ── report ────────────────────────────────────────────────────────────────────
print("=" * 62)
print(f"Validation complete — {len(data)} records processed")
print(f"  Passed   : {passed}  ({passed/len(data)*100:.1f}%)")
print(f"  Rejected : {rejected}  ({rejected/len(data)*100:.1f}%)")
print(f"\nRejection breakdown:")
for reason, items in sorted(rejection_reasons.items(), key=lambda x: -len(x[1])):
    print(f"  {reason:25s}: {len(items)}")

# ── show 10 rejected examples ─────────────────────────────────────────────────
print("\n" + "─" * 62)
print("10 REJECTED examples")
print("─" * 62)
shown = 0
for reason, items in sorted(rejection_reasons.items(), key=lambda x: -len(x[1])):
    for r in items:
        entity = extract_entity(r["question"])
        print(f"[{reason}]")
        print(f"  Q      : {r['question']}")
        print(f"  entity : {entity}")
        print(f"  W      : {r['wiki_context'][:110]}")
        print()
        shown += 1
        if shown >= 10:
            break
    if shown >= 10:
        break

# ── show 5 accepted examples ──────────────────────────────────────────────────
print("─" * 62)
print("5 ACCEPTED examples")
print("─" * 62)
accepted_records = [r for r in results_out if r.get("wiki_source") != "rejected" and r.get("wiki_context")]
for r in accepted_records[:5]:
    entity = extract_entity(r["question"])
    print(f"[{r['wiki_source']}]")
    print(f"  Q      : {r['question']}")
    print(f"  entity : {entity}")
    print(f"  W      : {r['wiki_context'][:110]}")
    print()

print(f"Output saved to {OUT_FILE}")
