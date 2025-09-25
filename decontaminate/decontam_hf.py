#!/usr/bin/env python3
"""
Self-contained, fault-tolerant decontamination tool that:
  1) Downloads HF benchmarks and builds a pickled 13-gram index.
  2) Converts the pickle index to a fast, native binary format.
  3) Downloads a target HF dataset (like MixtureVitae) to a local cache.
  4) Scans the cached files in parallel using a high-performance Rust worker.

This file-based approach is fast, scalable, and fault-tolerant.


1. # Install dependencies
   # poetry install (to get datasets, pandas, pyarrow, zstandard, maturin)
   # (And install Rust)

2. # Build the Rust worker
   cd decontam_fast && python -m maturin develop --release && cd ..

3. # Build the benchmark index (creates .pkl)
   python decontam_hf.py build-index \
     --out-index ./decontam_index_13gram.pkl

4. # Convert the index to a fast binary format (creates .native)
   python decontam_hf.py build-index-native \
     --in-pickle ./decontam_index_13gram.pkl \
     --out-native ./decontam_index_13gram.native

5. # Download the target dataset to a local cache
   python decontam_hf.py download-dataset \
     --hf-dataset "ontocord/MixtureVitae-300BT" \
     --cache-dir "./hf_cache"

6. # Run the scan using the native index and cached files
   python decontam_hf.py scan \
     --input-glob "./hf_cache/ontocord___mixture_vitae-300_bt/*/*/*.parquet" \
     --text-key "text" \
     --id-key "id" \
     --index ./decontam_index_13gram.native \
     --out-dir ./decontam_results \
     --workers 64
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import re
import sys
import unicodedata
import time
import pickle
from collections import defaultdict
import struct

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
from multiprocessing import get_context
from functools import partial
from pathlib import Path

# External deps
from datasets import load_dataset, disable_caching, get_dataset_config_names

try:
    import pandas as pd
except Exception:
    pd = None

# --- Import the Rust native module ---
# This will fail if `maturin develop` has not been run
try:
    from fast_decont import build_index_native_rust, scan_file_rust
except ImportError:
    print("=" * 50, file=sys.stderr)
    print("ERROR: Could not import the 'fast_decont' Rust module.", file=sys.stderr)
    print("Please compile it by running:", file=sys.stderr)
    print("  cd decontam_fast && python -m maturin develop --release && cd ..", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    sys.exit(1)


STOP_WORDS = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "this",
        "that",
        "these",
        "those",
        "or",
        "if",
        "then",
        "else",
        "not",
    ]
)

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


# Compiled (case-insensitive). If ANY of these match the 13-gram, we DROP it.
NGRAM_REJECT_PATTERNS = [
    # --- Multiple-choice scaffolding / roman numerals / options (largely unaffected by stopwords) ---
    r"\b(i{1,4})\b[^a-z0-9]+only\b",  # "i/ii/iii/iv ... only"
    r"\b(i{1,4})\b[^a-z0-9]+(and|or)[^a-z0-9]+\b(i{1,4})\b",  # "ii and iv", "i or iii"
    r"\boption[s]?\s*[a-d]\b",  # "option a/b/c/d"
    r"\banswers?\s*:\s*[a-d]\b",  # "answers: a"
    r"\bwhich(?:\s+\w+){0,2}\s+following\b",  # "which of the following" (stopwords may be gone)
    r"\bselect(?:\s+\w+){0,2}\s+(answer|option)\b",  # "select the best/correct answer/option"
    r"\b(?:none|all)(?:\s+\w+){0,2}\s+above\b",  # "none/all of the above"
    r"\bboth\s+[a-d]\s+(?:and\s+)?[a-d]\b",  # "both A and B" (allow missing 'and')
    r"(?:\btrue\b|\bfalse\b)(?:\W+(?:true|false)){2,}",  # repeated TF grids  (FIXED: no stray space)
    # --- Lettered/numbered list boilerplate ---
    r"\b(a|b|c|d)\b[^a-z0-9]+(i{1,4}|[1-4])\b[^a-z0-9]+(only|true|false)\b",
    r"\b(a|b|c|d)\)\s*(i{1,4}|[1-4])\b",  # "a) i", "b) 2"
    r"\(\s*(i{1,4}|[1-4])\s*\)\s*(and|or)\s*\(\s*(i{1,4}|[1-4])\s*\)",
    # --- “Correct answer is …” (allow missing stopwords) ---
    r"\bcorrect\s+answer(?:\s+\w+){0,2}\s+is\b",
    # --- Logic / philosophy boilerplate ---
    r"\blogically\s+equivalent\b.*\bcontradictory\b",
    r"\bneither\b.*\bnor\b.*\bconsistent\b",
    # --- Generic math boilerplate (robust to dropped stopwords) ---
    r"\bfind\s+value\s+x\b",  # "find the value of x"
    r"\bsolve\s+(?:for\s+)?x\b",  # "solve for x" / "solve x"
    r"\bsimplify\s+expression\b",  # "simplify the expression"
    r"\bremainder\s+\w*\s+divided\b",  # "remainder when ... divided by ..."
    r"\bwhat\s+probability\b",  # "what is the probability"
    r"\bcompute\s+(sum|product|difference|quotient)\b",
    r"\bequation\s+(line|circle|parabola)\b",  # "equation of the line/circle/parabola"
    r"\bif\s+(?:and\s+)?only\s+if\b",  # "if and only if"
    # --- Generic exam phrasing (robust to dropped 'is/are/the/of') ---
    r"\bstatements?\b.*\b(true|false)\b",  # "statements ... true/false"
    r"\bwhich\s+statements?\b.*\btrue\b",
    r"\baccording(?:\s+\w+){0,2}\s+passage\b",  # "according to the passage"
]


NGRAM_REJECT_REGEXES = [re.compile(p, re.IGNORECASE) for p in NGRAM_REJECT_PATTERNS]




def _read_hash_mask_pairs(path):
    pairs = []
    with open(path, "rb") as f:
        data = f.read()
    rec_size = 16
    for i in range(0, len(data), rec_size):
        h, m = struct.unpack("<QQ", data[i : i + rec_size])
        pairs.append((h, m))
    return pairs

def report_leak_per_benchmark(attr_sidecar_path: str, hits_dir: str, sources_json_path: str, out_dir: str):
    # 1) Denominator: total unique hashes per source from the index sidecar
    print("Loading benchmark n-gram totals (denominator)...", file=sys.stderr)
    total_per_src = defaultdict(int)
    for h, mask in _read_hash_mask_pairs(attr_sidecar_path):
        mm = mask
        while mm:
            sid = (mm & -mm).bit_length() - 1  # index of lowest set bit
            total_per_src[sid] += 1
            mm &= mm - 1

    # 2) Numerator: track leaked hashes per source *file*
    # This structure maps: { sid -> { filename -> {hash, ...} } }
    print(f"Aggregating leaked n-grams from *.hits.bin in {hits_dir}...", file=sys.stderr)
    per_benchmark_source = defaultdict(lambda: defaultdict(set))
    
    all_files = glob.glob(os.path.join(hits_dir, "*.hits.bin"))
    if not all_files:
        print(f"WARNING: No *.hits.bin files found in {hits_dir}. Cannot generate report.", file=sys.stderr)
        return

    for fn in all_files:
        # Get a clean name for the file, e.g., "file_A" from "path/to/file_A.hits.bin"
        file_key = os.path.basename(fn).replace(".hits.bin", "")
        
        for h, mask in _read_hash_mask_pairs(fn):
            mm = mask
            while mm:
                sid = (mm & -mm).bit_length() - 1
                # Store the hash 'h' as coming from 'file_key' for benchmark 'sid'
                per_benchmark_source[sid][file_key].add(h)
                mm &= mm - 1

    # 3) Compute the global union of leaked hashes for the main report
    # { sid -> {all_leaked_hashes} }
    global_leaked_hashes = defaultdict(set)
    for sid, file_map in per_benchmark_source.items():
        for filename, hash_set in file_map.items():
            global_leaked_hashes[sid].update(hash_set)

    # { sid -> total_leaked_count }
    leaked_per_src = {sid: len(s) for sid, s in global_leaked_hashes.items()}
    print(f"Found leaks for {len(leaked_per_src)} benchmarks across {len(all_files)} files.", file=sys.stderr)

    # 4) Load legend to map sid -> name (created at build time)
    with open(sources_json_path, "r") as f:
        legend = json.load(f)
    id_to_name = legend.get("id_to_name") or []
    # Fallback if your earlier build wrote just a list:
    if isinstance(id_to_name, list) and not legend.get("name_to_id") and not id_to_name:
        try:
            id_to_name = json.load(open(sources_json_path))  # handle legacy format
        except Exception:
            pass

    # 5) Build report rows for the *main* high-level report
    rows = []
    all_sids = set(total_per_src.keys()) | set(leaked_per_src.keys())
    for sid in sorted(all_sids):
        name = id_to_name[sid] if sid < len(id_to_name) else f"sid_{sid}"
        denom = total_per_src.get(sid, 0)
        num = leaked_per_src.get(sid, 0) # Total unique leaked n-grams
        pct = (num / denom) if denom else 0.0
        rows.append(
            {
                "source_id": sid,
                "benchmark": name,
                "total_unique_ngrams_in_index": denom,
                "unique_ngrams_leaked": num,
                "leak_pct": pct,
            }
        )

    # 6) Write *main* report artifacts
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    main_report_path = os.path.join(out_dir, "_LEAK_BY_BENCHMARK.json")
    with open(main_report_path, "w") as f:
        json.dump(rows, f, indent=2)

    if pd:
        try:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "_LEAK_BY_BENCHMARK.csv"), index=False)
        except Exception:
            pass

    # 7) Print a compact summary of the *main* report
    print("\n" + "="*50, file=sys.stderr)
    print("Overall Per-benchmark leak (% of indexed unique 13-grams):", file=sys.stderr)
    for r in rows:
        print(
            f"  {r['benchmark']:<20}  {r['unique_ngrams_leaked']:>9}/{r['total_unique_ngrams_in_index']:<9}  {100 * r['leak_pct']:.2f}%",
            file=sys.stderr,
        )
        
    # 8) NEW: Build and write per-benchmark, per-file breakdown
    print("\n" + "="*50, file=sys.stderr)
    print("Per-benchmark leak contribution by source file (Top 5):", file=sys.stderr)
    
    breakdown_report = {}
    # Use the same sorted 'all_sids' as the main report for consistent ordering
    for sid in sorted(all_sids):
        name = id_to_name[sid] if sid < len(id_to_name) else f"sid_{sid}"
        
        total_benchmark_ngrams = total_per_src.get(sid, 0)
        total_leaked_count = leaked_per_src.get(sid, 0) # from global_leaked_hashes
        
        if total_leaked_count == 0:
            print(f"  {name:<20}  -- No leaks found. --", file=sys.stderr)
            continue

        print(f"  {name:<20}  (Total Unique Leaks: {total_leaked_count} n-grams)", file=sys.stderr)
        
        file_contributions = []
        # per_benchmark_source[sid] = { filename -> {hash, ...} }
        for filename, hash_set in per_benchmark_source.get(sid, {}).items():
            # This counts all unique n-grams for this benchmark *found in this file*
            file_leak_count = len(hash_set)
            if file_leak_count == 0:
                continue
                
            pct_of_total_leak = (file_leak_count / total_leaked_count) if total_leaked_count else 0.0
            
            file_contributions.append({
                "file": filename,
                "leaked_ngrams_from_file": file_leak_count,
                "pct_of_total_leak_for_this_benchmark": pct_of_total_leak,
            })
        
        # Sort high to low by the number of leaked n-grams
        file_contributions.sort(key=lambda x: x["leaked_ngrams_from_file"], reverse=True)
        
        breakdown_report[name] = {
            "total_leaked_ngrams_in_benchmark": total_leaked_count,
            "total_ngrams_in_benchmark": total_benchmark_ngrams,
            "overall_leak_pct": (total_leaked_count / total_benchmark_ngrams) if total_benchmark_ngrams else 0.0,
            "contributors": file_contributions
        }

        # Print top 5 contributors to stderr
        for i, r in enumerate(file_contributions[:5]):
            print(
                f"    {i+1}. {r['file']:<30} {r['leaked_ngrams_from_file']:>9} n-grams ({r['pct_of_total_leak_for_this_benchmark']:>7.2%})",
                file=sys.stderr
            )
        if len(file_contributions) > 5:
            print(f"    ... and {len(file_contributions) - 5} other files.", file=sys.stderr)

    # Write the new breakdown report to a separate JSON
    breakdown_path = os.path.join(out_dir, "_LEAK_BREAKDOWN_BY_FILE.json")
    with open(breakdown_path, "w") as f:
        json.dump(breakdown_report, f, indent=2)
    print(f"\nWrote detailed breakdown to: {breakdown_path}", file=sys.stderr)
    print("="*50, file=sys.stderr)


def ngram_allowed(ngram: str) -> bool:
    """Return True if this n-gram should be KEPT (i.e., not a ubiquitous template)."""
    if not ngram or ngram.isspace():
        return False
    for rx in NGRAM_REJECT_REGEXES:
        if rx.search(ngram):
            return False
    return True


def normalize_text(s: str) -> List[str]:
    if not s:
        return []
    s = unicodedata.normalize("NFKC", s).lower()

    # Old logic:
    # return WORD_RE.findall(s)

    # --- NEW FILTERING LOGIC ---
    # This prevents n-grams like "1 2 3 4 5..."
    tokens = WORD_RE.findall(s)
    filtered_tokens = []
    for t in tokens:
        if t not in STOP_WORDS:
            filtered_tokens.append(t)
    return filtered_tokens


def iter_ngrams(words: List[str], n: int = 13) -> Iterator[str]:
    L = len(words)
    if L < n:
        return
    for i in range(L - n + 1):
        yield " ".join(words[i : i + n])


def hash_ngram(ng: str) -> int:
    # 8-byte (64-bit) hash
    return int.from_bytes(hashlib.blake2b(ng.encode("utf-8"), digest_size=8).digest(), "little", signed=False)


# -----------------------------
# HF benchmark loaders → texts
# (Used only for build-index)
# -----------------------------


def _concat_fields(obj: Dict[str, Any], keys: List[str]) -> str:
    parts = []
    for k in keys:
        v = obj.get(k, "")
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            parts.extend([str(x) for x in v if x])
        elif isinstance(v, dict):
            if "text" in v and isinstance(v["text"], list):
                parts.extend([str(x) for x in v["text"] if x])
            elif "labels" in v and isinstance(v["labels"], list):
                parts.extend([str(x) for x in v["labels"] if x])
            else:
                parts.append(json.dumps(v, ensure_ascii=False))
        else:
            parts.append(str(v))
    return " ".join(parts).strip()


def load_mmlu_texts(is_train: bool) -> Iterator[str]:
    base_cache = os.path.expanduser("~/.cache/huggingface/datasets")
    mmlu_caches = [os.path.join(base_cache, "hendrycks_test"), os.path.join(base_cache, "cais___mmlu")]
    for path in mmlu_caches:
        try:
            if os.path.exists(path):
                print(f"  [MMLU] Aggressively clearing cache: {path}", file=sys.stderr)
                # shutil.rmtree(path)
        except Exception as e:
            print(f"  [MMLU] Warning: Failed to clear cache {path} (Error: {e})", file=sys.stderr)
    try:
        print("  [MMLU] Attempting primary: 'hendrycks_test'...", file=sys.stderr)
        mmlu_subjects = get_dataset_config_names("hendrycks_test")
        if "all" in mmlu_subjects:
            mmlu_subjects.remove("all")
        print(f"  [MMLU] Found {len(mmlu_subjects)} subjects for hendrycks_test. Loading each...", file=sys.stderr)
        for i, subject in enumerate(mmlu_subjects):
            if (i + 1) % 10 == 0:
                print(f"  [MMLU] Loading subject {i + 1}/{len(mmlu_subjects)}: {subject}", file=sys.stderr)
            time.sleep(1)
            ds_subject_dict = load_dataset("hendrycks_test", subject, trust_remote_code=True)
            for split_name, split_dataset in ds_subject_dict.items():
                if (split_name in ["test"] and not is_train) or (split_name in ["train"] and is_train):
                    for row in split_dataset:
                        yield _concat_fields(row, ["question", "choices", "answer"])
    except Exception as e:
        print(f"  [MMLU] Primary 'hendrycks_test' failed (Error: {e}). Using fallback 'cais/mmlu'.", file=sys.stderr)
        try:
            print("  [MMLU-Fallback] Fetching config list for cais/mmlu...", file=sys.stderr)
            cais_subjects = get_dataset_config_names("cais/mmlu")
            if "all" in cais_subjects:
                cais_subjects.remove("all")
            if "auxiliary_train" in cais_subjects:
                cais_subjects.remove("auxiliary_train")
            print(
                f"  [MMLU-Fallback] Found {len(cais_subjects)} subjects for cais/mmlu. Loading each...", file=sys.stderr
            )
            for i, subject in enumerate(cais_subjects):
                try:
                    if (i + 1) % 5 == 0:
                        print(
                            f"  [MMLU-Fallback] Loading subject {i + 1}/{len(cais_subjects)}: {subject}",
                            file=sys.stderr,
                        )
                        # break # TODO: temp
                    time.sleep(1)
                    ds_subject_dict = load_dataset("cais/mmlu", subject, trust_remote_code=True)
                    for split_name, split_dataset in ds_subject_dict.items():
                        if (split_name in ["test"] and not is_train) or (split_name in ["train"] and is_train):
                            for row in split_dataset:
                                yield _concat_fields(row, ["question", "choices", "answer"])
                except Exception as e2:
                    print(
                        f"  [MMLU-Fallback] WARNING: Failed to load subject '{subject}' (Error: {e2}). Skipping.",
                        file=sys.stderr,
                    )
                    continue
        except Exception as e3:
            print(
                f"  [MMLU-Fallback] CRITICAL: Failed to load 'cais/mmlu' (Error: {e3}). MMLU will be skipped.",
                file=sys.stderr,
            )
            pass


def load_copa_texts() -> Iterator[str]:
    ds = load_dataset("super_glue", "copa")
    for split_name, split_dataset in ds.items():
        if split_name in ["test"]:
            for row in split_dataset:
                yield _concat_fields(row, ["premise", "choice1", "choice2", "question", "label"])


def load_lambada_texts() -> Iterator[str]:
    ds = load_dataset("EleutherAI/lambada_openai")
    for split_name, split_dataset in ds.items():
        if split_name in ["test"]:
            for row in split_dataset:
                yield _concat_fields(row, ["text"])


def load_openbookqa_texts() -> Iterator[str]:
    ds = load_dataset("allenai/openbookqa", "main")
    for split_name, split_dataset in ds.items():
        if split_name in ["test"]:
            for row in split_dataset:
                yield _concat_fields(row, ["question_stem", "choices", "answerKey", "fact1"])


def load_winogrande_texts() -> Iterator[str]:
    ds = load_dataset("winogrande", "winogrande_xl")
    for split_name, split_dataset in ds.items():
        if split_name in ["test"]:
            for row in split_dataset:
                yield _concat_fields(row, ["sentence", "option1", "option2", "answer"])


def load_humaneval_texts() -> Iterator[str]:
    """Loads text from OpenAI HumanEval"""
    try:
        ds = load_dataset("openai_humaneval")
        for split in ds.keys():  # Only has a 'test' split
            for row in ds[split]:
                # fields: prompt, canonical_solution, test
                yield _concat_fields(row, ["prompt", "canonical_solution", "test"])
    except Exception as e:
        print(f"  [HumanEval] ERROR: Failed to load openai_humaneval: {e}. Skipping.", file=sys.stderr)
        pass


def load_mbpp_texts() -> Iterator[str]:
    """Loads text from the MBPP (Mostly Basic Python Problems) dataset"""
    try:
        ds = load_dataset("mbpp")
        for split in ["test"]:
            for row in ds[split]:
                # fields: text, code, test_list (list of strings)
                yield _concat_fields(row, ["text", "code", "test_list"])
    except Exception as e:
        print(f"  [MBPP] ERROR: Failed to load mbpp: {e}. Skipping.", file=sys.stderr)
        pass


def load_math_texts() -> Iterator[str]:
    """Loads text from the Hendrycks' Competition MATH dataset"""
    try:
        ds = load_dataset("hendrycks/competition_math")
        for split in ["test"]:
            for row in ds[split]:
                # fields: problem, solution
                yield _concat_fields(row, ["problem", "solution"])
    except Exception as e:
        print(f"  [MATH] ERROR: Failed to load hendrycks/competition_math: {e}. Skipping.", file=sys.stderr)
        pass


def load_if_eval_texts() -> Iterator[str]:
    """Loads text from the IF-Eval benchmark (hosted in alpaca_farm)"""
    try:
        ds = load_dataset("tatsu-lab/alpaca_farm", "if_eval")
        for split in ds.keys():  # only 'validation' split
            for row in ds[split]:
                # fields: instruction, input, output
                yield _concat_fields(row, ["instruction", "input", "output"])
    except Exception as e:
        print(f"  [IF-Eval] ERROR: Failed to load tatsu-lab/alpaca_farm:if_eval: {e}. Skipping.", file=sys.stderr)
        pass


def load_arc_texts() -> Iterator[str]:
    configs = ["ARC-Challenge", "ARC-Easy"]
    for config_name in configs:
        try:
            print(f"  [ARC] Loading config: {config_name}", file=sys.stderr)
            ds_config = load_dataset("ai2_arc", config_name, trust_remote_code=True)
            for split_name, split_dataset in ds_config.items():
                if split_name.lower() == "test":
                    for row in split_dataset:
                        yield _concat_fields(row, ["question", "choices", "answerKey"])
        except Exception as e:
            print(f"  [ARC] WARNING: Failed to load config '{config_name}' (Error: {e}). Skipping.", file=sys.stderr)
            continue


def load_boolq_texts() -> Iterator[str]:
    ds = load_dataset("super_glue", "boolq")
    for split_name, split_dataset in ds.items():
        if split_name.lower() == "test":
            for row in split_dataset:
                yield _concat_fields(row, ["passage", "question", "label"])


def load_hellaswag_texts() -> Iterator[str]:
    ds = load_dataset("hellaswag")
    for split_name, split_dataset in ds.items():
        if split_name in ["test"]:
            for row in split_dataset:
                yield _concat_fields(row, ["ctx", "endings", "label"])


def load_piqa_texts() -> Iterator[str]:
    ds = load_dataset("piqa")
    for split_name, split_dataset in ds.items():
        if split_name.lower() == "test":
            for row in split_dataset:
                yield _concat_fields(row, ["goal", "sol1", "sol2", "label"])


def load_simpleqa_texts() -> Iterator[str]:
    ds = load_dataset("lighteval/SimpleQA")
    for split in ["test"]:  # ds.keys():
        for row in ds[split]:
            yield _concat_fields(row, ["problem"])


def load_coding_texts() -> Iterator[str]:
    ds = load_dataset("openai/openai_humaneval")["test"]
    for row in ds:
        yield _concat_fields(row, ["prompt", "solution"])


def load_ifeval_texts() -> Iterator[str]:
    ds = load_dataset("facebook/Multi-IF")["train"]
    for row in ds:
        yield _concat_fields(row, ["turn_1_prompt"])
    ds = load_dataset("google/IFEval")["train"]
    for row in ds:
        yield _concat_fields(row, ["prompt"])


def load_gsm8k_texts() -> Iterator[str]:
    ds = load_dataset("openai/gsm8k", "main")["test"]
    for row in ds:
        yield _concat_fields(row, ["question", "answer"])


def load_musr_texts() -> Iterator[str]:
    ds = load_dataset("TAUR-Lab/MuSR")
    for split in ["murder_mysteries", "object_placements", "team_allocation"]:
        for row in ds[split]:
            yield _concat_fields(row, ["question", "choices", "answer_choice"])


def load_bbh_texts() -> Iterator[str]:
    splits = [
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "navigate",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
    ]
    for split in splits:
        ds = load_dataset("lighteval/bbh", split)["train"]
        for row in ds:
            yield _concat_fields(row, ["input", "choices", "target_idx"])


def load_alert_texts() -> Iterator[str]:
    for split in ["alert", "alert_adversarial"]:
        ds = load_dataset("Babelscape/ALERT", split)["test"]
        for row in ds:
            yield _concat_fields(row, ["prompt"])


def load_gpqa_texts() -> Iterator[str]:
    for split in ["gpqa_extended", "gpqa_main", "gpqa_diamond", "gpqa_experts"]:
        ds = load_dataset("Idavidrein/gpqa", split)["train"]
        for row in ds:
            yield _concat_fields(row, ["Question", "Correct Answer"])


# def load_math_texts() -> Iterator[str]:
#     ds = load_dataset("DigitalLearningGmbH/MATH-lighteval")
#     for split in ["test"]:
#         for row in ds[split]:
#             yield _concat_fields(row, ["problem", "solution"])


# def load_mbpp_texts() -> Iterator[str]:
#     ds = load_dataset("Muennighoff/mbpp", trust_remote_code=True)
#     for split in ["test"]:
#         for row in ds[split]:
#             yield _concat_fields(row, ["text", "code"])


def load_aime2024_texts() -> Iterator[str]:
    ds = load_dataset("HuggingFaceH4/aime_2024")["train"]
    for row in ds:
        yield _concat_fields(row, ["problem", "solution"])


def load_aime2025_texts() -> Iterator[str]:
    ds = load_dataset("MathArena/aime_2025")["train"]
    for row in ds:
        yield _concat_fields(row, ["problem", "answer"])


def load_hmmt_feb_2025_texts() -> Iterator[str]:
    ds = load_dataset("MathArena/hmmt_feb_2025")["train"]
    for row in ds:
        yield _concat_fields(row, ["problem", "answer"])


def load_usamo_texts() -> Iterator[str]:
    ds = load_dataset("MathArena/usamo_2025")["train"]
    for row in ds:
        yield _concat_fields(row, ["problem", "sample_solution"])


def load_brumo_texts() -> Iterator[str]:
    ds = load_dataset("MathArena/brumo_2025")["train"]
    for row in ds:
        yield _concat_fields(row, ["problem", "answer"])


def load_math500_texts() -> Iterator[str]:
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    for row in ds:
        yield _concat_fields(row, ["problem", "answer"])


# def load_code_texts() -> Iterator[str]:
#     ds = load_dataset("livecodebench/code_generation_lite", "v4")
#     for split in ds.keys():
#         for row in ds[split]:
#             yield _concat_fields(row, ["question_content"])
#     ds = load_dataset("livecodebench/code_generation_lite", "v4_v5")
#     for split in ds.keys():
#         for row in ds[split]:
#             yield _concat_fields(row, ["question_content"])


def load_mmlu_redux_texts() -> Iterator[str]:
    splits = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]
    for split in splits:
        ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", split)["test"]
        for row in ds:
            yield _concat_fields(row, ["question", "choices", "answer"])


def load_mmmlu_texts() -> Iterator[str]:
    splits = [
        "default",
        "AR_XY",
        "BN_BD",
        "DE_DE",
        "ES_LA",
        "FR_FR",
        "HI_IN",
        "ID_ID",
        "IT_IT",
        "JA_JP",
        "KO_KR",
        "PT_BR",
        "SW_KE",
        "YO_NG",
        "ZH_CN",
    ]
    for split in splits:
        ds = load_dataset("openai/MMMLU", split)["test"]
        for row in ds:
            yield _concat_fields(row, ["Question", "Answer"])


def load_mmlu_pro_texts() -> Iterator[str]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    # for split in ["test"]:
    for row in ds:
        yield _concat_fields(row, ["question", "options", "answer"])


def load_commonsense_qa_texts() -> Iterator[str]:
    ds = load_dataset("tau/commonsense_qa")["test"]
    for row in ds:
        yield _concat_fields(row, ["question", "choices", "answerKey"])


def load_do_not_answer_texts() -> Iterator[str]:
    ds = load_dataset("LibrAI/do-not-answer")["train"]
    for row in ds:
        yield _concat_fields(row, ["question"])


# def load_toxigen_texts() -> Iterator[str]:
#     for split in ["train", "annotations", "annotated"]:
#         ds = load_dataset("toxigen/toxigen-data", split)['train']
#         for row in ds:
#             yield _concat_fields(row, ["text"])
#     splits = ['hate_asian_1k', 'hate_bisexual_1k', 'hate_black_1k', 'hate_chinese_1k',
#               'hate_immigrant_1k', 'hate_jewish_1k', 'hate_latino_1k', 'hate_lgbtq_1k', 'hate_mental_disability_1k',
#               'hate_mexican_1k', 'hate_middle_east_1k', 'hate_muslim_1k', 'hate_native_american_1k',
#               'hate_physical_disability_1k', 'hate_trans_1k', 'hate_women_1k', 'neutral_asian_1k',
#               'neutral_bisexual_1k', 'neutral_black_1k', 'neutral_chinese_1k', 'neutral_immigrant_1k',
#               'neutral_jewish_1k', 'neutral_latino_1k', 'neutral_lgbtq_1k', 'neutral_mental_disability_1k',
#               'neutral_mexican_1k', 'neutral_middle_east_1k',
#               'neutral_muslim_1k', 'neutral_native_american_1k', 'neutral_physical_disability_1k', 'neutral_women_1k']
#     for split in splits:
#         ds = load_dataset("toxigen/toxigen-data", "prompts")['split']
#         for row in ds:
#             yield _concat_fields(row, ["text"])


def load_advbench_texts() -> Iterator[str]:
    ds = load_dataset("walledai/AdvBench")["train"]
    for row in ds:
        yield _concat_fields(row, ["prompt", "target"])


# def load_social_iqa_texts() -> Iterator[str]:
#     ds = load_dataset("allenai/social_i_qa")
#     for split in ds.keys():
#         for row in ds[split]:
#             yield _concat_fields(row, ["question", "answerA", "answerB", "answerC", "label"])

"""
TIGER-Lab/MMLU-Pro
Viewer
•
Updated 25 days ago
•
12.1k
•
49.9k
•
379
Note Split: test
"""


def get_benchmark_loaders():
    loaders = [
        ("Ifeval", load_ifeval_texts),
        ("ARC", load_arc_texts),
        ("COPA", load_copa_texts),
        ("LAMBADA", load_lambada_texts),
        ("OpenBookQA", load_openbookqa_texts),
        ("Winogrande", load_winogrande_texts),
        ("BoolQ", load_boolq_texts),
        ("HellaSwag", load_hellaswag_texts),
        ("PIQA", load_piqa_texts),
        ("Gsm8k", load_gsm8k_texts),
        ("ALERT", load_alert_texts),
        ("GPQA", load_gpqa_texts),
        ("MATH", load_math_texts),
        ("MBPP", load_mbpp_texts),
        ("humaneval", load_humaneval_texts),
        ("SimpleQA", load_simpleqa_texts),
        ("CommonsenseQA", load_commonsense_qa_texts),
        ("DoNotAnswer", load_do_not_answer_texts),
        # ("MATH500", load_math500_texts),
  
        # ("MMLU", load_mmlu_texts),
        #      ("MuSR", load_musr_texts),
        #      ("BBH", load_bbh_texts),
        #      ("AIME2024", load_aime2024_texts),
        #      ("AIME2025", load_aime2025_texts),
        #      ("HMMT_Feb_2025", load_hmmt_feb_2025_texts),
        #      ("USAMO", load_usamo_texts),
        #      ("BRUMO", load_brumo_texts),
        #      ("Code", load_code_texts),
        #      ("MMLU_Redux", load_mmlu_redux_texts),
        #      ("MMMLU", load_mmmlu_texts),
        #      ("MMLU_Pro", load_mmlu_pro_texts),
        # ("SocialIQA", load_social_iqa_texts),
        # ("ToxiGen", load_toxigen_texts),
    ]

    return loaders


def load_all_benchmark_texts() -> Iterator[str]:
    disable_caching()
    loaders = get_benchmark_loaders()
    for name, fn in loaders:
        print(f"[build-index] Downloading & indexing: {name}", file=sys.stderr)
        for text in fn():
            yield text


# -----------------------------
# Build index (Python)
# -----------------------------


def build_index_hf(out_path: str, ngram: int = 13) -> None:
    disable_caching()

    def _texts_to_ngram_set(text_iter: Iterator[str], ngram: int, prefix: str) -> Tuple[set, int]:
        ngram_set = set()
        items = 0
        print(f"[{prefix}] Starting text iteration...", file=sys.stderr)
        for text in text_iter:
            toks = normalize_text(text)
            for ng in iter_ngrams(toks, n=ngram):
                if not ngram_allowed(ng):
                    continue
                ngram_set.add(hash_ngram(ng))
            items += 1
            if items % 10000 == 0:
                print(f"[{prefix}] ...processed items={items:,}, unique n-grams={len(ngram_set):,}", file=sys.stderr)
        print(f"[{prefix}] DONE. items={items:,}, unique n-grams={len(ngram_set):,}", file=sys.stderr)
        return ngram_set, items

    # ---------- Stage 1: MMLU train/eval ----------
    print("[build-index] Stage 1: MMLU train/eval...", file=sys.stderr)
    mmlu_train_ngrams, _ = _texts_to_ngram_set(load_mmlu_texts(is_train=True), ngram, "MMLU-Train")
    mmlu_eval_ngrams, _ = _texts_to_ngram_set(load_mmlu_texts(is_train=False), ngram, "MMLU-Eval")
    final_mmlu_ngrams = mmlu_eval_ngrams - mmlu_train_ngrams
    print(f"[build-index] MMLU after train subtraction: {len(final_mmlu_ngrams):,}", file=sys.stderr)

    # ---------- Stage 2: Other eval benchmarks with attribution ----------
    print("[build-index] Stage 2: Attribution over other eval benchmarks...", file=sys.stderr)
    name_to_id: Dict[str, int] = {}
    id_to_name: List[str] = []

    def _sid(name: str) -> int:
        if name not in name_to_id:
            name_to_id[name] = len(id_to_name)
            id_to_name.append(name)
        return name_to_id[name]

    # hash -> 64-bit bitmask of sources (allows up to 64 benchmarks)
    hash2mask: Dict[int, int] = {}

    # (a) add MMLU (with train subtraction) as one source
    mmlu_sid = _sid("MMLU")
    for h in final_mmlu_ngrams:
        hash2mask[h] = hash2mask.get(h, 0) | (1 << mmlu_sid)

    # (b) add the rest
    for name, loader in get_benchmark_loaders():
        print(f"[build-index]   indexing {name}...", file=sys.stderr)
        sid = _sid(name)
        item_count = 0
        for text in loader():
            toks = normalize_text(text)
            for ng in iter_ngrams(toks, n=ngram):
                if not ngram_allowed(ng):
                    continue
                h = hash_ngram(ng)
                hash2mask[h] = hash2mask.get(h, 0) | (1 << sid)
            item_count += 1
            if item_count % 10000 == 0:
                print(f"[build-index]   {name}: items={item_count:,}", file=sys.stderr)

    final_ngram_set = set(hash2mask.keys())
    print(f"[build-index] Total unique 13-grams (all evals): {len(final_ngram_set):,}", file=sys.stderr)

    # ---------- Stage 3: (optional) common-corpus subtraction ----------
    # You currently have common_corpus disabled. If you re-enable later,
    # use the same 9-gram helper you already wrote and remove by mapping hash2mask.

    # ---------- Stage 4: save artifacts ----------
    # 4a) Pickle index (unchanged API for your native converter)
    with open(out_path, "wb") as f:
        pickle.dump({"ngram": ngram, "hashes": final_ngram_set}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[build-index] Wrote pickle -> {out_path}", file=sys.stderr)

    # 4b) Binary attribution sidecar: <u64 hash, u64 mask> pairs
    attr_path = out_path + ".attr"
    with open(attr_path, "wb") as bf:
        for h, mask in hash2mask.items():
            bf.write(struct.pack("<QQ", h, mask))
    print(f"[build-index] Wrote attribution map -> {attr_path}", file=sys.stderr)

    # 4c) Source legend
    meta_path = out_path + ".sources.json"
    with open(meta_path, "w") as mf:
        json.dump({"name_to_id": name_to_id, "id_to_name": id_to_name}, mf, indent=2)
    print(f"[build-index] Wrote source legend -> {meta_path}", file=sys.stderr)

    per_source_total = defaultdict(int)
    seen_by_src = defaultdict(set)
    for h, mask in hash2mask.items():
        m = mask
        while m:
            sid = (m & -m).bit_length() - 1  # trailing bit index
            seen_by_src[sid].add(h)
            m &= m - 1
    for sid, s in seen_by_src.items():
        per_source_total[sid] = len(s)

    # when writing meta:
    meta_path = out_path + ".sources.json"
    with open(meta_path, "w") as mf:
        json.dump(
            {
                "name_to_id": name_to_id,
                "id_to_name": id_to_name,
                "per_source_total": {str(k): v for k, v in per_source_total.items()},
            },
            mf,
            indent=2,
        )


# -----------------------------
# Scanning logic (File-based, Rust worker)
# -----------------------------


@dataclass
class ScanConfig:
    text_key: str
    id_key: Optional[str]
    index_path: str  # Path to .native file
    out_dir: str
    min_hits: int
    ngram: int
    min_coverage: float


def scan_one_file_wrapper(file_path: str, cfg: ScanConfig) -> Dict[str, Any]:
    base = os.path.basename(file_path)
    stem = re.sub(r"\.(parquet|jsonl|jsonl\.gz|jsonl\.zst)$", "", base)

    hits_dir = os.path.join(cfg.out_dir, "per_source_hits")
    os.makedirs(hits_dir, exist_ok=True)
    per_hits_path = os.path.join(hits_dir, f"{stem}.hits.bin")

    summary_path = os.path.join(cfg.out_dir, "summaries", f"{stem}.summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            if "file" in summary and "scanned" in summary:
                print(f"[scan:skip] File already processed: {base}", file=sys.stderr)
                return summary
        except Exception:
            pass

    try:
        scanned, contaminated, json_results = scan_file_rust(
            file_path=file_path,
            text_key=cfg.text_key,
            id_key=cfg.id_key,
            index_path=cfg.index_path,
            n=cfg.ngram,
            min_hits=cfg.min_hits,
            min_coverage=cfg.min_coverage,
            out_hits_path=per_hits_path,
        )
    except Exception as e:
        print(f"[scan:ERROR] Failed to process file {base}: {e}", file=sys.stderr)
        return {"file": base, "scanned": 0, "contaminated": 0, "contam_rate": 0.0, "error": str(e)}

    # ---- per-source aggregation (doc-level) ----

    per_src_hits: Dict[int, int] = defaultdict(int)
    per_src_docs_any: Dict[int, set] = defaultdict(set)
    per_src_docs_primary: Dict[int, set] = defaultdict(set)

    for line in json_results:
        rec = json.loads(line)
        doc_id = rec["doc_id"]
        pairs = rec.get("src_hits", [])
        if not pairs:
            continue
        # count any
        local_max = 0
        for sid, cnt in pairs:
            sid_i = int(sid)
            cnt_i = int(cnt)
            per_src_hits[sid_i] += cnt_i
            per_src_docs_any[sid_i].add(doc_id)
            if cnt_i > local_max:
                local_max = cnt_i
        # primary winners (ties allowed)
        for sid, cnt in pairs:
            if int(cnt) == local_max:
                per_src_docs_primary[int(sid)].add(doc_id)

    per_source = {
        str(sid): {
            "hits": int(per_src_hits[sid]),
            "docs_any": len(per_src_docs_any[sid]),
            "docs_primary": len(per_src_docs_primary[sid]),
        }
        for sid in set(list(per_src_hits.keys()) + list(per_src_docs_any.keys()) + list(per_src_docs_primary.keys()))
    }

    # ---- write artifacts ----
    out_path = os.path.join(cfg.out_dir, "contaminated_docs", f"{stem}.decontam.ndjson")
    if contaminated > 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for line in json_results:
                f.write(line + "\n")

    summary = {
        "file": base,
        "scanned": scanned,
        "contaminated": contaminated,
        "contam_rate": (contaminated / scanned) if scanned else 0.0,
        "per_source": per_source,  # <- NEW
    }

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f)

    if contaminated > 0:
        print(
            f"[scan:DONE] {base} | Scanned: {scanned:,} | Contaminated: {contaminated:,} ({summary['contam_rate']:.3%})",
            file=sys.stderr,
        )

    return summary


# -----------------------------
# CLI Commands
# -----------------------------


def do_build_index(ns: argparse.Namespace) -> None:
    build_index_hf(out_path=ns.out_index, ngram=ns.ngram)


def do_build_index_native(ns: argparse.Namespace) -> None:
    print(f"Loading pickle index from: {ns.in_pickle}", file=sys.stderr)
    try:
        # This calls the Rust function to handle conversion
        build_index_native_rust(in_pickle_path=ns.in_pickle, out_native_path=ns.out_native)
        print(f"Wrote native index to: {ns.out_native}", file=sys.stderr)
    except Exception as e:
        if "No such file or directory" in str(e):
            print(f"ERROR: Pickle file not found: {ns.in_pickle}", file=sys.stderr)
            print("Run the 'build-index' command first.", file=sys.stderr)
        else:
            print(f"ERROR: Failed to build native index: {e}", file=sys.stderr)
        sys.exit(1)

        # ---- copy attribution sidecars from the pickle to the native basename ----
        # ---- copy attribution sidecars from the pickle basename to the native basename ----
    try:
        # Absolute, to avoid empty dirname edge-cases
        in_pickle_abs = os.path.abspath(ns.in_pickle)
        out_native_abs = os.path.abspath(ns.out_native)

        for suf in (".attr", ".sources.json"):
            src = in_pickle_abs + suf
            dst = out_native_abs + suf

            if not os.path.exists(src):
                print(f"(note) sidecar missing (expected but optional): {src}", file=sys.stderr)
                continue

            dst_dir = os.path.dirname(dst)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)

            shutil.copyfile(src, dst)
            print(f"Copied sidecar {src} -> {dst}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: failed to copy sidecars: {e}", file=sys.stderr)


def do_download_dataset(ns: argparse.Namespace) -> None:
    print(f"Downloading '{ns.hf_dataset}' to cache dir: {ns.cache_dir}", file=sys.stderr)
    disable_caching()
    try:
        load_dataset(
            ns.hf_dataset,
            split="train",  # Assume train split for now
            cache_dir=ns.cache_dir,
            streaming=False,  # <-- This forces download
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print("Download complete.", file=sys.stderr)
    print(f"Files are located in subdirectories of: {ns.cache_dir}", file=sys.stderr)
    print("Please construct a glob to find the .parquet files, e.g.:")
    print(f'  "{ns.cache_dir}/ontocord___mixture_vitae-300_bt/*/*/*.parquet"')


def do_scan(ns: argparse.Namespace) -> None:
    files = sorted(glob.glob(ns.input_glob))
    if not files:
        print(f"ERROR: No shard files matched the glob pattern: {ns.input_glob}", file=sys.stderr)
        print("Please check the path to your cached dataset files.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(files):,} files to scan.", file=sys.stderr)

    cfg = ScanConfig(
        text_key=ns.text_key,
        id_key=ns.id_key,
        index_path=ns.index,
        out_dir=ns.out_dir,
        min_hits=ns.min_hits,
        ngram=ns.ngram,
        min_coverage=ns.min_cov,
    )

    # Use partial to freeze the 'cfg' argument for the worker
    scan_func = partial(scan_one_file_wrapper, cfg=cfg)

    start_method = "fork" if sys.platform != "win32" else "spawn"

    total_scanned = 0
    total_contaminated = 0
    all_summaries = []

    print(f"Starting pool with {ns.workers} workers...", file=sys.stderr)

    with get_context(start_method).Pool(processes=ns.workers) as pool:
        # imap_unordered is good for load balancing
        for summary in pool.imap_unordered(scan_func, files):
            if "error" not in summary:
                total_scanned += summary.get("scanned", 0)
                total_contaminated += summary.get("contaminated", 0)
                all_summaries.append(summary)
            else:
                print(f"[scan:ERROR] File failed: {summary['file']} ({summary['error']})", file=sys.stderr)

            if len(all_summaries) % 100 == 0 and len(all_summaries) > 0:
                rate = (total_contaminated / total_scanned) if total_scanned else 0.0
                print(
                    f"[scan:PROGRESS] Files processed: {len(all_summaries):,}/{len(files):,} | Total Scanned: ~{total_scanned:,} | Total Contam: {total_contaminated:,} ({rate:.4%})",
                    file=sys.stderr,
                )

    # --- Final Aggregation ---
    print("[scan] All files processed. Aggregating results...", file=sys.stderr)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    final_summary_path = os.path.join(cfg.out_dir, "_FINAL_SUMMARY.json")
    final_csv_path = os.path.join(cfg.out_dir, "_FINAL_SUMMARY.csv")

    rate = (total_contaminated / total_scanned) if total_scanned else 0.0
    final_summary = {
        "total_files_scanned": len(all_summaries),
        "total_docs_scanned": total_scanned,
        "total_contaminated": total_contaminated,
        "contamination_rate": rate,
    }

    with open(final_summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    # Write CSV summary
    if pd and all_summaries:
        try:
            pd.DataFrame(all_summaries).to_csv(final_csv_path, index=False)
            print(f"Wrote aggregate CSV to: {final_csv_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to write CSV summary: {e}", file=sys.stderr)

    print("[scan] DONE.", file=sys.stderr)
    print(json.dumps(final_summary, indent=2), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="HF n-gram decontamination tool (Rust-accelerated).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- build-index ---
    b = sub.add_parser("build-index", help="Download HF benchmarks and build 13-gram index (.pkl).")
    b.add_argument("--ngram", type=int, default=13, help="Word n-gram size (default 13).")
    b.add_argument("--out-index", required=True, help="Output pickle path for index (e.g., index.pkl).")
    b.set_defaults(func=do_build_index)

    # --- build-index-native ---
    bn = sub.add_parser("build-index-native", help="Convert .pkl index to fast .native binary format.")
    bn.add_argument("--in-pickle", required=True, help="Input pickle index (from build-index).")
    bn.add_argument("--out-native", required=True, help="Output path for the native index (e.g., index.native).")
    bn.set_defaults(func=do_build_index_native)

    # --- download-dataset ---
    dl = sub.add_parser("download-dataset", help="Download a HF dataset to a local cache directory.")
    dl.add_argument("--hf-dataset", required=True, help="HF dataset name (e.g., 'ontocord/MixtureVitae-300BT').")
    dl.add_argument("--cache-dir", required=True, help="Directory to save the dataset cache (e.g., ./hf_cache).")
    dl.set_defaults(func=do_download_dataset)

    # --- scan ---
    s = sub.add_parser("scan", help="Scan downloaded dataset shards using the fast Rust worker.")
    s.add_argument(
        "--input-glob", required=True, help="Glob for cached shards (e.g., './hf_cache/path/to/*/*.parquet')."
    )
    s.add_argument("--text-key", default="text", help="Field name containing document text (default: text).")
    s.add_argument(
        "--id-key", default=None, help="Optional field name for a stable doc id (otherwise hashed from text)."
    )
    s.add_argument("--index", required=True, help="Path to the .native index (from build-index-native).")
    s.add_argument("--ngram", type=int, default=13, help="Word n-gram size; must match the index (default: 13).")
    s.add_argument("--min-hits", type=int, default=3, help="Min distinct n-gram overlaps to flag a doc (default: 1).")
    s.add_argument("--out-dir", required=True, help="Directory for per-file outputs and summaries.")
    s.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="Parallel workers (default: CPU count).")
    s.add_argument(
        "--min-cov",
        type=float,
        default=0.001,
        help="Minimum coverage of unique 13-grams to flag a doc (e.g., 0.001 = 0.1%)",
    )
    s.set_defaults(func=do_scan)

    rep = sub.add_parser("report-leak", help="Compute per-benchmark leak %% from per-file hits and index sidecar.")
    rep.add_argument(
        "--attr",
        required=True,
        help="Path to the attribution sidecar written at build time (e.g., index_13gram.pkl.attr).",
    )
    rep.add_argument("--hits-dir", required=True, help="Directory with *.hits.bin files (written during scan).")
    rep.add_argument(
        "--legend", required=True, help="Path to the source legend JSON (e.g., index_13gram.pkl.sources.json)."
    )
    rep.add_argument("--out-dir", required=True, help="Where to write the leak report JSON/CSV.")
    rep.set_defaults(func=lambda ns: report_leak_per_benchmark(ns.attr, ns.hits_dir, ns.legend, ns.out_dir))

    ns = ap.parse_args()
    ns.func(ns)


if __name__ == "__main__":
    main()
   