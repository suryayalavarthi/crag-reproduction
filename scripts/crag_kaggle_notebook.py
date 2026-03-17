# %% [markdown]
# # CRAG — Corrective Retrieval Augmented Generation
# Reproducing Yan et al. (arXiv 2401.15884) on Kaggle with 2x Tesla T4
#
# **Pipeline:**
# 1. T5-large retrieval evaluator scores each (question, passage) pair
# 2. Action trigger classifies as CORRECT / AMBIGUOUS / INCORRECT
# 3. Knowledge refinement filters irrelevant sentence strips (CORRECT/AMBIGUOUS)
# 4. Mistral-7B-Instruct generates the final answer

# %% [markdown]
# ## Cell 1 — Install packages

# %%
import subprocess, sys

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install(
    "transformers==4.36.1",
    "accelerate==0.26.1",
    "bitsandbytes==0.41.3.post2",
    "huggingface_hub==0.20.3",
    "sentencepiece==0.1.99",
    "safetensors==0.4.1",
    "shap==0.44.1",
)

print("✓ All packages installed.")

# %% [markdown]
# ## Cell 2 — Unzip data

# %%
import os
import zipfile

INPUT_DIR  = "/kaggle/input/crag-resources"
WORK_DIR   = "/kaggle/working"
T5_DIR     = os.path.join(WORK_DIR, "t5_evaluator")
POPQA_DIR  = os.path.join(WORK_DIR, "popqa_data")

os.makedirs(T5_DIR,    exist_ok=True)
os.makedirs(POPQA_DIR, exist_ok=True)

def unzip(src, dst):
    print(f"Unzipping {os.path.basename(src)} → {dst} ...")
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dst)
    print(f"  Done. Files:")
    for root, dirs, files in os.walk(dst):
        level = root.replace(dst, "").count(os.sep)
        indent = "  " * (level + 1)
        for f in files:
            print(f"{indent}{os.path.join(root, f).replace(dst, '').lstrip('/')}")

unzip(os.path.join(INPUT_DIR, "t5_evaluator.zip"),  T5_DIR)
unzip(os.path.join(INPUT_DIR, "popqa_data.zip"),    POPQA_DIR)

# %% [markdown]
# ## Cell 2b — Download PopQA ground truth from HuggingFace

# %%
from huggingface_hub import hf_hub_download

gt_download_path = hf_hub_download(
    repo_id="akariasai/PopQA",
    filename="popqa_longtail_w_gs.jsonl",
    repo_type="dataset",
    local_dir="/kaggle/working/popqa_data/"
)
print(f"Ground truth file downloaded to: {gt_download_path}")

# %% [markdown]
# ## Cell 3 — Load T5 evaluator

# %%
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── locate the checkpoint directory ────────────────────────────────────────────
def find_evaluator_ckpt(base):
    """Walk base to find the directory that contains config.json + a weights file."""
    for root, dirs, files in os.walk(base):
        has_cfg = "config.json" in files
        has_wts = any(
            f in files for f in [
                "pytorch_model.bin",
                "model.safetensors",
                "model.safetensors.index.json",
            ]
        ) or any(f.startswith("model-") and f.endswith(".safetensors") for f in files)
        if has_cfg and has_wts:
            return root
    return None

EVAL_CKPT = find_evaluator_ckpt(T5_DIR)
if EVAL_CKPT is None:
    # Fallback: try the well-known relative path used in the paper repo
    EVAL_CKPT = os.path.join(T5_DIR, "models", "finetuned_t5_evaluator")
print(f"Evaluator checkpoint: {EVAL_CKPT}")

# ── if the checkpoint uses shard-named weights (model-001.safetensors),
#    build the index.json that HuggingFace expects ──────────────────────────────
_index_path = os.path.join(EVAL_CKPT, "model.safetensors.index.json")
if not os.path.exists(_index_path):
    _shard = next(
        (f for f in os.listdir(EVAL_CKPT)
         if f.startswith("model-") and f.endswith(".safetensors")),
        None
    )
    if _shard:
        import json
        from safetensors.torch import load_file as _st_load
        _keys = list(_st_load(os.path.join(EVAL_CKPT, _shard)).keys())
        _idx  = {"metadata": {"total_size": 0},
                 "weight_map": {k: _shard for k in _keys}}
        with open(_index_path, "w") as _f:
            json.dump(_idx, _f)
        print(f"  Created {_index_path} ({len(_keys)} tensors → {_shard})")

EVAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading evaluator on {EVAL_DEVICE} ...")

eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_CKPT)
eval_model     = AutoModelForSequenceClassification.from_pretrained(
    EVAL_CKPT,
    num_labels=1,
    ignore_mismatched_sizes=True,   # head size is already correct; suppresses warning
)
eval_model.eval()
eval_model.to(EVAL_DEVICE)
print("✓ T5 evaluator loaded.")

# ── scoring function ────────────────────────────────────────────────────────────
def score_documents(question: str, docs: list[str]) -> list[float]:
    """
    Score each document against the question.
    Input format mirrors the paper: 'question [SEP] document'
    Returns a list of floats (raw logits; higher = more relevant).
    """
    scores = []
    for doc in docs:
        if not doc or not doc.strip() or doc.strip().endswith("[SEP]"):
            scores.append(-1.0)
            continue
        input_text = question + " [SEP] " + doc
        enc = eval_tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        try:
            with torch.no_grad():
                out = eval_model(
                    enc["input_ids"].to(EVAL_DEVICE),
                    attention_mask=enc["attention_mask"].to(EVAL_DEVICE),
                )
            scores.append(float(out.logits.cpu()))
        except Exception as e:
            print(f"  [score_documents] error: {e}")
            scores.append(-1.0)
    return scores

# smoke test
_test_q    = "What is the capital of France?"
_test_docs = [
    "Paris is the capital and most populous city of France.",
    "The mitochondria is the powerhouse of the cell.",
]
_test_scores = score_documents(_test_q, _test_docs)
print(f"Smoke-test scores: {[round(s, 4) for s in _test_scores]}")
print("  (Expect: first score > second score)")

# %% [markdown]
# ## Cell 4 — Load Mistral-7B generator (4-bit)

# %%
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

GEN_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
print(f"Loading {GEN_MODEL_ID} in 4-bit ...")

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
gen_model     = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.float16,
)
gen_model.eval()
print("✓ Mistral-7B loaded.")

def generate_answer(question: str, context: str, max_new_tokens: int = 100) -> str:
    """
    Generate an answer using Mistral-7B-Instruct.
    Prompt format follows the paper's PopQA template.
    """
    if context and context.strip():
        prompt = (
            "Refer to the following documents, follow the instruction and answer the question.\n\n"
            f"Documents: {context}\n\n"
            f"Instruction: Answer the question: {question}"
        )
    else:
        prompt = f"Answer the following question concisely: {question}"

    # Apply Mistral chat template
    messages = [{"role": "user", "content": prompt}]
    formatted = gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = gen_tokenizer(formatted, return_tensors="pt").to(gen_model.device)
    try:
        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,     # ignored when do_sample=False
                pad_token_id=gen_tokenizer.eos_token_id,
            )
        # decode only the newly generated tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        answer = gen_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"  [generate_answer] error: {e}")
        answer = ""
    return answer

# %% [markdown]
# ## Cell 5 — CRAG action trigger

# %%
def get_action(scores: list[float], upper: float = 0.59, lower: float = -0.99) -> str:
    """
    Determine the CRAG retrieval action from a set of document scores.

    Logic (from paper, Table 2 / §3.2):
      - If ANY doc scores ≥ upper  → CORRECT   (at least one reliable doc)
      - Elif ANY doc scores ≥ lower → AMBIGUOUS  (mixed quality)
      - Else all docs < lower       → INCORRECT  (no reliable doc)

    Thresholds are the PopQA values reported in the paper (upper=0.59, lower=-0.99).
    """
    if not scores:
        return "INCORRECT"
    if any(s >= upper for s in scores):
        return "CORRECT"
    elif any(s >= lower for s in scores):
        return "AMBIGUOUS"
    else:
        return "INCORRECT"

# unit tests
assert get_action([0.7, 0.1])          == "CORRECT"
assert get_action([-0.5, -0.3])        == "AMBIGUOUS"
assert get_action([-1.5, -2.0])        == "INCORRECT"
assert get_action([])                  == "INCORRECT"
print("✓ get_action unit tests passed.")

# %% [markdown]
# ## Cell 6 — Knowledge refinement

# %%
def _split_into_strips(doc: str) -> list[str]:
    """
    'Excerption' decomposition from §3.3 of the paper:
    Split on '?' and '. ', then concatenate into 3-sentence windows.
    Short fragments (<= 5 words) are merged with the previous strip.
    """
    num_concat = 3
    origin_strips = []
    for part in doc.split("?"):
        origin_strips += part.split(". ")

    strips = []
    for s in origin_strips:
        if not strips:
            strips.append(s)
        elif len(s.split()) > 5:
            strips.append(s)
        else:
            strips[-1] += s

    # Concatenate into windows of num_concat sentences
    final_strips = []
    buf = []
    for s in strips:
        buf.append(s)
        if len(buf) == num_concat:
            final_strips.append(" ".join(buf))
            buf = []
    if buf:
        final_strips.append(" ".join(buf))
    return final_strips


def refine_knowledge(
    question: str,
    doc: str,
    evaluator_fn,
    score_threshold: float = -0.5,
    top_n: int = 6,
) -> str:
    """
    Decompose doc into sentence strips, score each with the T5 evaluator,
    keep strips with score > score_threshold (up to top_n), return concatenation.

    Falls back to the full doc if no strips pass the threshold.
    """
    if not doc or not doc.strip():
        return ""

    strips = _split_into_strips(doc)
    strips = [s for s in strips if len(s.split()) >= 4]   # drop trivially short fragments

    if not strips:
        return doc

    scores = evaluator_fn(question, strips)
    scored = sorted(zip(scores, strips), key=lambda x: x[0], reverse=True)

    kept = [s for sc, s in scored[:top_n] if sc > score_threshold]
    return "; ".join(kept) if kept else doc   # fallback: return the original

# %% [markdown]
# ## Cell 7 — Full CRAG pipeline

# %%
def crag_pipeline(
    question: str,
    retrieved_docs: list[str],
    upper_threshold: float = 0.59,
    lower_threshold: float = -0.99,
) -> dict:
    """
    Full CRAG pipeline (§3 of the paper).

    Args:
        question:       The input question string.
        retrieved_docs: List of retrieved passage strings.

    Returns a dict with keys: question, action, context, answer.
    """
    result = {"question": question, "action": None, "context": None, "answer": None}

    # ── Step 1: score all documents ────────────────────────────────────────────
    try:
        scores = score_documents(question, retrieved_docs)
    except Exception as e:
        print(f"  [crag_pipeline] scoring failed: {e}")
        scores = [-1.0] * len(retrieved_docs)

    # ── Step 2: trigger action ─────────────────────────────────────────────────
    action = get_action(scores, upper=upper_threshold, lower=lower_threshold)
    result["action"] = action

    # ── Step 3: build context based on action ──────────────────────────────────
    if action == "CORRECT":
        # Refine: keep only the most relevant strips from the best documents
        best_docs = [
            doc for score, doc in
            sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
            if score >= upper_threshold
        ]
        refined_parts = []
        for doc in best_docs[:3]:   # top-3 correct docs
            refined = refine_knowledge(question, doc, score_documents)
            if refined:
                refined_parts.append(refined)
        context = " ".join(refined_parts)

    elif action == "INCORRECT":
        # Web search is skipped; use empty context per the task instructions
        context = ""

    else:  # AMBIGUOUS
        # Combine refined versions of all retrieved docs
        refined_parts = []
        for doc in retrieved_docs:
            refined = refine_knowledge(question, doc, score_documents)
            if refined:
                refined_parts.append(refined)
        context = " ".join(refined_parts)

    result["context"] = context

    # ── Step 4: generate answer ────────────────────────────────────────────────
    try:
        answer = generate_answer(question, context)
    except Exception as e:
        print(f"  [crag_pipeline] generation failed: {e}")
        answer = ""
    result["answer"] = answer

    return result

# %% [markdown]
# ## Cell 8 — Run on PopQA (first 50 questions)

# %%
import json
from collections import Counter

# ── locate input files ─────────────────────────────────────────────────────────
def find_file(base, filename):
    for root, dirs, files in os.walk(base):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_dir(base, dirname):
    for root, dirs, files in os.walk(base):
        if dirname in dirs:
            return os.path.join(root, dirname)
    return None

test_file = find_file(POPQA_DIR, "test_popqa.txt")
print(f"test_popqa.txt : {test_file}")

# ── parse test_popqa.txt ───────────────────────────────────────────────────────
# Format (from data_process.py): "question [SEP] passage\t0_or_1"
# Multiple lines per question (one per retrieved passage)
questions_all  = []
passages_map   = {}   # question → [passage, ...]
answers_map    = {}   # question → [answer, ...] (from possible answer fields)

if test_file:
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # strip optional label
            if "\t" in line:
                content, _ = line.rsplit("\t", 1)
            else:
                content = line
            if " [SEP] " not in content:
                continue
            q, p = content.split(" [SEP] ", 1)
            if q not in passages_map:
                questions_all.append(q)
                passages_map[q] = []
            passages_map[q].append(p)
else:
    print("WARNING: test_popqa.txt not found — trying sources + retrieved_psgs files.")
    sources_file = find_file(POPQA_DIR, "sources")
    psgs_file    = find_file(POPQA_DIR, "retrieved_psgs")
    if sources_file and psgs_file:
        with open(sources_file) as f:
            questions_all = [l.strip() for l in f if l.strip()]
        with open(psgs_file) as f:
            for q, line in zip(questions_all, f):
                passages_map[q] = [p.strip() for p in line.strip().split("[sep]") if p.strip()]
    else:
        raise FileNotFoundError("Cannot locate PopQA data files in " + POPQA_DIR)

# ── also try to find a JSONL with ground-truth answers ────────────────────────
gt_file = find_file(POPQA_DIR, "popqa_longtail_w_gs.jsonl")
gt_answers = {}   # question → list[str]
if gt_file:
    print(f"Ground-truth file: {gt_file}")
    with open(gt_file, "r") as f:
        for line in f:
            item = json.loads(line)
            q = item.get("question", "")
            # answers may be in "answers" or "output" field
            ans = item.get("answers", item.get("output", []))
            if isinstance(ans, str):
                ans = [ans]
            gt_answers[q] = [a.lower().strip() for a in ans]
else:
    print("NOTE: popqa_longtail_w_gs.jsonl not found — accuracy cell will be skipped.")

print(f"\nTotal questions loaded: {len(questions_all)}")
questions = questions_all[:50]
print(f"Running CRAG on first {len(questions)} questions ...")

# ── run the pipeline ───────────────────────────────────────────────────────────
results = []
for i, q in enumerate(questions):
    docs = passages_map.get(q, [])
    print(f"[{i+1:02d}/{len(questions)}] {q[:70]!r}  ({len(docs)} docs)")
    try:
        res = crag_pipeline(q, docs)
    except Exception as e:
        print(f"  ERROR: {e}")
        res = {"question": q, "action": "ERROR", "context": "", "answer": ""}
    results.append(res)
    print(f"  → action={res['action']}  answer={res['answer'][:80]!r}")

# ── save results ───────────────────────────────────────────────────────────────
out_path = os.path.join(WORK_DIR, "crag_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved {len(results)} results to {out_path}")

# ── action distribution ────────────────────────────────────────────────────────
action_counts = Counter(r["action"] for r in results)
print("\nAction distribution:")
for action, count in sorted(action_counts.items()):
    pct = 100 * count / len(results)
    print(f"  {action:12s}: {count:3d}  ({pct:.1f}%)")

# ── first 5 results ────────────────────────────────────────────────────────────
print("\n--- First 5 results ---")
for r in results[:5]:
    print(f"Q : {r['question']}")
    print(f"A : {r['answer']}")
    print(f"   [action={r['action']}  context_len={len(r['context'])}]")
    print()

# %% [markdown]
# ## Cell 9 — Basic accuracy evaluation

# %%
if not gt_answers:
    print("No ground-truth answers available — skipping accuracy evaluation.")
else:
    correct = 0
    evaluated = 0
    per_result = []

    for r in results:
        q   = r["question"]
        ans = r["answer"].lower().strip()
        gts = gt_answers.get(q, [])

        if not gts:
            per_result.append({"question": q, "answer": ans, "match": None})
            continue

        # String-match: check if any ground-truth string appears in the answer
        match = any(gt in ans for gt in gts)
        correct   += int(match)
        evaluated += 1
        per_result.append({"question": q, "answer": ans, "ground_truth": gts, "match": match})

    accuracy = correct / evaluated if evaluated > 0 else 0.0
    print(f"Accuracy (string match): {correct}/{evaluated} = {accuracy*100:.1f}%")
    print(f"  (Note: {len(results) - evaluated} questions had no ground-truth entry)")

    # break down by action
    print("\nAccuracy by action:")
    for action in ["CORRECT", "AMBIGUOUS", "INCORRECT"]:
        subset = [
            pr for pr, res in zip(per_result, results)
            if res["action"] == action and pr.get("match") is not None
        ]
        if subset:
            acc = sum(int(r["match"]) for r in subset) / len(subset)
            print(f"  {action:12s}: {sum(int(r['match']) for r in subset)}/{len(subset)} = {acc*100:.1f}%")
        else:
            print(f"  {action:12s}: no evaluated samples")

    # save accuracy
    acc_path = os.path.join(WORK_DIR, "accuracy.txt")
    with open(acc_path, "w") as f:
        f.write(f"Accuracy (string match, n={evaluated}): {accuracy*100:.2f}%\n")
        f.write(f"CORRECT:   {action_counts.get('CORRECT', 0)}\n")
        f.write(f"AMBIGUOUS: {action_counts.get('AMBIGUOUS', 0)}\n")
        f.write(f"INCORRECT: {action_counts.get('INCORRECT', 0)}\n")
    print(f"\n✓ Accuracy saved to {acc_path}")
