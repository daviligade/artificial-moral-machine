#!/usr/bin/env python3
"""Analyze the no-persona ("Experiment A'") baseline runs on all three models
(\texttt{gpt-4.1-nano}, \texttt{DeepSeek V3.1}, \texttt{Llama 3.3 70B}) and
write the per-dimension aggregate sparing-rate CSVs.

Computes:
  - For each model: aggregate count of (spared_positive, total) per AMCE
    dimension across all responses to the 5,356 trolley-style dilemmas,
    with no demographic preamble.
  - 8 rows per file (one per AMCE dimension).

Outputs:
  - data/per_dim_amce_nopersona_nano.csv
  - data/per_dim_amce_nopersona_deepseek.csv
  - data/per_dim_amce_nopersona_llama.csv

# ─────────────────────────────────────────────────────────────────────────────
# NOTE FOR EXTERNAL USERS
# ─────────────────────────────────────────────────────────────────────────────
# This script regenerates the per_dim_amce_nopersona_*.csv files by parsing
# raw OpenAI/Fireworks Batch API JSONL outputs. The raw JSONL files (totalling
# ~6 GB across the three no-persona runs) are NOT redistributed in this OSF
# repository. If you want to run this script, you must either:
#   (a) reproduce the experiment yourself or
#   (b) contact the authors after acceptance for the raw batches.
#
# The released per_dim_amce_nopersona_*.csv files (in data/) are the outputs
# this script produces, and feed Table 9 (no-persona vs with-persona) in the
# paper. Each k=20 prompt variant per dilemma was independently sampled at
# T=0 with the same prompt-phrasing variation as the persona-conditioned
# main experiment.
#
# AMCE-dimension classification and Age_old direction convention match
# regenerate_per_cohort_amce.py exactly: Age_old is the spare-elderly rate
# on dilemmas where exactly one side is tagged as elderly.
# ─────────────────────────────────────────────────────────────────────────────

Usage: python3 analyze_nopersona_experiments.py
"""

import os, json, re, glob, time
import pandas as pd
from collections import defaultdict

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(BASE_DIR, "data")
DILEMMAS   = os.path.join(DATA_DIR, "dilemmas.csv")

# Raw-JSONL directories. EDIT THESE TO YOUR LOCAL PATHS before running.
# These are the locations of the raw Batch-API JSONL files for each no-persona
# experiment. They are NOT redistributed.
NO_PERSONA_DIRS = {
    'nano':     "data/experiment_A_prime_nano_no_persona",
    'deepseek': "data/experiment_A_prime_deepseekv3p1_no_persona",
    'llama':    "data/experiment_A_prime_llama3_70b_no_persona",
}

# Custom-id regex for parsing per-model: must capture dilemma_id (group 1).
ID_PATTERNS = {
    'nano':     r"expA_prime_nano_dilemma_(\d+)__agent_(\d+)",
    'deepseek': r"expA_prime_deepseekv3p1_dilemma_(\d+)__agent_(\d+)",
    'llama':    r"expA_prime_llama_dilemma_(\d+)__agent_(\d+)",
}

OUTPUT_CSVS = {
    'nano':     os.path.join(DATA_DIR, "per_dim_amce_nopersona_nano.csv"),
    'deepseek': os.path.join(DATA_DIR, "per_dim_amce_nopersona_deepseek.csv"),
    'llama':    os.path.join(DATA_DIR, "per_dim_amce_nopersona_llama.csv"),
}

# ─── Tag + dimension logic (kept in sync with regenerate_per_cohort_amce.py)
def tag(d):
    """Tag a character description for AMCE dimensions."""
    t = set()
    d = d.lower()
    t.add('non-human' if 'pet' in d else 'human')
    if 'doctor' in d or 'executive' in d: t.add('high-status')
    if 'criminal' in d: t.add('low-status')
    if 'child' in d or 'young' in d: t.add('young')
    if 'old' in d or 'elderly' in d: t.add('old')
    if any(w in d for w in ['woman', 'women', 'female', 'girl', 'pregnant']): t.add('female')
    if any(w in d for w in [' man', ' men ', 'male']) and 'female' not in d: t.add('male')
    if d.endswith(' man') or d.endswith(' men') or d.endswith(' man.') or d.endswith(' men.'):
        t.add('male')
    if 'jaywalking' in d: t.add('jaywalking')
    if 'legally' in d: t.add('legal')
    if 'pregnant' in d or 'child' in d or 'children' in d: t.add('fragile')
    return frozenset(t)


def amce_dims(ta, tb):
    """Return list of (dimension_name, positive_is_first: bool) for a pair."""
    dims = []
    if ('young' in ta) != ('young' in tb): dims.append(('Age',            'young' in ta))
    if ('old'   in ta) != ('old'   in tb): dims.append(('Age_old',        'old' in ta))
    if ('human' in ta) != ('human' in tb): dims.append(('Species',        'human' in ta))
    if ('female' in ta) != ('female' in tb): dims.append(('Gender',       'female' in ta))
    if ('high-status' in ta) != ('high-status' in tb): dims.append(('Status', 'high-status' in ta))
    if ('legal' in ta) != ('legal' in tb): dims.append(('Law_legal',      'legal' in ta))
    if ('jaywalking' in ta) != ('jaywalking' in tb): dims.append(('Law_jaywalking', 'jaywalking' not in ta))
    if ('fragile' in ta) != ('fragile' in tb): dims.append(('Fragile',    'fragile' in ta))
    return dims


# ─── Load dilemma pair-tag mapping ─────────────────────────────────────────
print(f"Loading dilemma definitions from {DILEMMAS}...")
dilemmas = pd.read_csv(DILEMMAS)
dilemma_info = {}
for _, r in dilemmas.iterrows():
    c0, c1 = r['canonical_choice_0'], r['canonical_choice_1']
    dilemma_info[r['dilemma_id']] = (c0, c1, amce_dims(tag(c0), tag(c1)))
print(f"  {len(dilemma_info):,} dilemmas indexed")


# ─── Generic parsers ───────────────────────────────────────────────────────
re_choice = re.compile(r'I\s*choose\s*([AB])', re.IGNORECASE)


def parse_outputs(out_glob, id_pattern, provider):
    """Parse output JSONLs. provider in {'openai', 'fireworks'} controls
    where the message content lives in the response object."""
    re_id = re.compile(id_pattern)
    rows = []
    n_total, n_invalid = 0, 0
    for f in sorted(glob.glob(out_glob)):
        with open(f) as fh:
            for line in fh:
                n_total += 1
                try:
                    d = json.loads(line)
                    m = re_id.search(d['custom_id'])
                    if not m:
                        continue
                    if provider == 'openai':
                        content = d['response']['body']['choices'][0]['message']['content']
                    else:  # fireworks (DeepSeek/Llama)
                        content = d['response']['choices'][0]['message']['content']
                    m_c = re_choice.search(content)
                    if not m_c:
                        n_invalid += 1
                        continue
                    rows.append((d['custom_id'], int(m.group(1)), m_c.group(1).upper()))
                except Exception:
                    n_invalid += 1
    return rows, n_total, n_invalid


def parse_inputs(in_glob, id_pattern):
    """Parse input JSONLs: return {custom_id: (dilemma_id, a_is_canon0)}."""
    re_id = re.compile(id_pattern)
    out = {}
    for f in sorted(glob.glob(in_glob)):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                cid = d['custom_id']
                m = re_id.search(cid)
                if not m:
                    continue
                dil = int(m.group(1))
                c0 = dilemma_info[dil][0]
                # OpenAI puts the user prompt at d['body']['messages'][0]['content'];
                # for Fireworks the input file may already use the same shape we built
                # for the OpenAI request. Both should expose 'Choice A:' in the prompt.
                try:
                    body = d.get('body') or d
                    msg  = body['messages'][0]['content']
                    ca = msg.split('Choice A:')[1].split('\n')[0].strip()
                except Exception:
                    ca = c0  # fallback: assume canonical order
                a_is_canon0 = (ca == c0) or (ca.rstrip('.') == c0.rstrip('.'))
                out[cid] = (dil, a_is_canon0)
    return out


# ─── Aggregate one model's no-persona run ──────────────────────────────────
def aggregate_one_model(model_key, exp_dir, id_pattern, out_path):
    print()
    print("=" * 80)
    print(f"NO-PERSONA — {model_key}")
    print("=" * 80)

    if not os.path.isdir(exp_dir):
        print(f"  Raw JSONL directory not found: {exp_dir}")
        print(f"  Skipping {model_key} (set NO_PERSONA_DIRS to your local path to run).")
        return

    provider = 'openai' if model_key == 'nano' else 'fireworks'

    print(f"\nReading inputs from {exp_dir} ...")
    t0 = time.time()
    inputs = parse_inputs(f"{exp_dir}/*.jsonl", id_pattern)
    print(f"  {len(inputs):,} inputs in {time.time()-t0:.1f}s")

    print("Reading outputs...")
    t0 = time.time()
    out_glob = f"{exp_dir}/merged_output/*.jsonl"
    if not glob.glob(out_glob):
        # Some setups co-locate outputs with inputs; try a fallback pattern.
        out_glob = f"{exp_dir}/*output*.jsonl"
    outputs, n_total, n_invalid = parse_outputs(out_glob, id_pattern, provider)
    print(f"  {len(outputs):,}/{n_total:,} valid responses ({n_invalid:,} invalid) in {time.time()-t0:.1f}s")

    # Aggregate per AMCE dimension
    print("Aggregating per AMCE dimension...")
    counts = defaultdict(lambda: {'spared_positive': 0, 'total': 0})
    for cid, dil, choice in outputs:
        if cid not in inputs:
            continue
        _, a_is_canon0 = inputs[cid]
        c0, c1, dims = dilemma_info[dil]
        # Decide which canonical character was SPARED.
        # Prompt format is "Choice A: kill X / Choice B: kill Y", so "I choose A"
        # means the model chose to KILL the side labeled A in the prompt — i.e.,
        # spared the OTHER side.
        #   - If a_is_canon0 == True  (no flip): 'A' kills c0, spares c1  → spared_canon0 = False
        #   - If a_is_canon0 == False (flipped): 'A' kills c1, spares c0  → spared_canon0 = True
        spared_canon0 = (choice == 'A') != a_is_canon0
        for dim_name, positive_is_first in dims:
            counts[dim_name]['total'] += 1
            # 'positive_is_first' is True if c0 holds the positive side
            if positive_is_first == spared_canon0:
                counts[dim_name]['spared_positive'] += 1

    # Write CSV in the same shape as the existing per_dim_amce_nopersona_*.csv:
    # amce_dimension,spared_positive,total,rate
    rows = []
    for dim in sorted(counts.keys()):
        sp, tot = counts[dim]['spared_positive'], counts[dim]['total']
        rate = sp / tot if tot > 0 else 0.0
        rows.append((dim, sp, tot, rate))
    df = pd.DataFrame(rows, columns=['amce_dimension', 'spared_positive', 'total', 'rate'])
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(df.to_string(index=False))


# ─── Main ──────────────────────────────────────────────────────────────────
for model_key in ['nano', 'deepseek', 'llama']:
    aggregate_one_model(
        model_key,
        exp_dir=NO_PERSONA_DIRS[model_key],
        id_pattern=ID_PATTERNS[model_key],
        out_path=OUTPUT_CSVS[model_key],
    )

print("\nDone.")
