#!/usr/bin/env python3
"""Analyze Llama 3.3 70B (Fireworks) experiments: A' (no-persona) and C (persona, 1 agent).

Structurally identical to analyze_deepseek_experiments.py; only the directory and
custom_id patterns differ. Computes:
  - Aggregate AMCE sparing rates for both
  - η² variance decomposition for C (persona)
  - Political monotonicity for C
  - Side-by-side comparison with gpt-4.1-nano and DeepSeek V3.1
"""

# ─────────────────────────────────────────────────────────────────────────────
# NOTE FOR EXTERNAL USERS
# ─────────────────────────────────────────────────────────────────────────────
# This script regenerates the per_cohort_amce_expC_*.csv file by parsing raw
# OpenAI/Fireworks Batch API JSONL outputs. The raw JSONL files (totalling ~3.6 GB
# per model) are NOT redistributed in this OSF repository. If you want to run
# this script, you must either:
#   (a) reproduce the experiment yourself or
#   (b) contact the authors after acceptance for the raw batches.
#
# The released per_cohort_amce_expC_deepseek.csv and per_cohort_amce_expC_llama.csv
# (in data/) are the outputs this script produces, and are sufficient to
# reproduce every numeric claim via verify_crossmodel.py.
# ─────────────────────────────────────────────────────────────────────────────

import os, json, re, glob, time
import pandas as pd
import numpy as np
from collections import defaultdict

DILEMMAS = "../data/dilemmas.csv"
dilemmas = pd.read_csv(DILEMMAS)

def tag(d):
    t = set(); d = d.lower()
    t.add('non-human' if 'pet' in d else 'human')
    if 'doctor' in d or 'executive' in d: t.add('high-status')
    if 'criminal' in d: t.add('low-status')
    if 'child' in d or 'young' in d: t.add('young')
    if 'old' in d or 'elderly' in d: t.add('old')
    if any(w in d for w in ['woman', 'women', 'female', 'girl', 'pregnant']): t.add('female')
    if any(w in d for w in [' man', ' men ', 'male']) and 'female' not in d: t.add('male')
    if d.endswith(' man') or d.endswith(' men') or d.endswith(' man.') or d.endswith(' men.'): t.add('male')
    if 'jaywalking' in d: t.add('jaywalking')
    if 'legally' in d: t.add('legal')
    if 'pregnant' in d or 'child' in d or 'children' in d: t.add('fragile')
    return frozenset(t)

def amce_dims(ta, tb):
    dims = []
    if ('young' in ta) != ('young' in tb): dims.append(('Age', 'young' in ta))
    if ('old' in ta) != ('old' in tb): dims.append(('Age_old', 'old' in ta))
    if ('human' in ta) != ('human' in tb): dims.append(('Species', 'human' in ta))
    if ('female' in ta) != ('female' in tb): dims.append(('Gender', 'female' in ta))
    if ('high-status' in ta) != ('high-status' in tb): dims.append(('Status', 'high-status' in ta))
    if ('legal' in ta) != ('legal' in tb): dims.append(('Law_legal', 'legal' in ta))
    if ('jaywalking' in ta) != ('jaywalking' in tb): dims.append(('Law_jaywalking', 'jaywalking' not in ta))
    if ('fragile' in ta) != ('fragile' in tb): dims.append(('Fragile', 'fragile' in ta))
    return dims

dilemma_info = {}
for _, r in dilemmas.iterrows():
    c0, c1 = r['canonical_choice_0'], r['canonical_choice_1']
    dilemma_info[r['dilemma_id']] = (c0, c1, amce_dims(tag(c0), tag(c1)))

re_choice = re.compile(r'I\s*choose\s*([AB])', re.IGNORECASE)

def parse_outputs_fireworks(out_glob, id_pattern):
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
                    if not m: continue
                    content = d['response']['choices'][0]['message']['content']
                    m_c = re_choice.search(content)
                    if not m_c:
                        n_invalid += 1
                        continue
                    choice = m_c.group(1).upper()
                    rows.append((d['custom_id'], m, choice))
                except Exception:
                    n_invalid += 1
    return rows, n_total, n_invalid

def parse_inputs(in_glob, id_pattern):
    re_id = re.compile(id_pattern)
    out = {}
    for f in sorted(glob.glob(in_glob)):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                cid = d['custom_id']
                m = re_id.search(cid)
                if not m: continue
                dil = int(m.group(1))
                c0 = dilemma_info[dil][0]
                ca = d['body']['messages'][0]['content'].split('Choice A:')[1].split('\n')[0].strip()
                a_is_canon0 = (ca == c0) or (ca.rstrip('.') == c0.rstrip('.'))
                out[cid] = (dil, m, a_is_canon0)
    return out

# ==========================================
# A. Experiment A' Llama (no-persona)
# ==========================================
print("=" * 80)
print("EXPERIMENT A' Llama — no-persona on Llama 3.3 70B (Fireworks)")
print("=" * 80)
DIR_A_L = "../data/experiment_A_prime_llama33fw_no_persona"# HERE YOU SHOULD PUT THE PATH TO THE RAW JSONLs

print("\nReading inputs...")
t0 = time.time()
inputs_A_L = parse_inputs(f"{DIR_A_L}/experiment_A_prime_llama33fw_batch_*.jsonl",
                          r"expA_prime_llama33fw_dilemma_(\d+)__agent_(\d+)")
print(f"  {len(inputs_A_L):,} inputs in {time.time()-t0:.0f}s")

print("Reading outputs...")
t0 = time.time()
outputs_A_L, n_total, n_invalid = parse_outputs_fireworks(
    f"{DIR_A_L}/batch_downloads/*.jsonl",
    r"expA_prime_llama33fw_dilemma_(\d+)__agent_(\d+)")
print(f"  {n_total:,} parsed ({n_invalid} invalid) in {time.time()-t0:.0f}s")

counts_A_L = defaultdict(lambda: {'spare_pos':0, 'total':0})
for cid, m_out, choice in outputs_A_L:
    dil_id = int(m_out.group(1))
    if cid not in inputs_A_L: continue
    _, _, a_is_canon0 = inputs_A_L[cid]
    killed_c0 = (choice == 'A') if a_is_canon0 else (choice == 'B')
    spared_c0 = not killed_c0
    for dim_name, pos_first in dilemma_info[dil_id][2]:
        counts_A_L[dim_name]['total'] += 1
        if pos_first == spared_c0:
            counts_A_L[dim_name]['spare_pos'] += 1

# ==========================================
# B. Experiment C Llama (persona, 1 agent)
# ==========================================
print()
print("=" * 80)
print("EXPERIMENT C — Llama 3.3 70B persona, 1 agent per cell (~3M queries)")
print("=" * 80)
DIR_C = "../data/experiment_C_llama33fw_1agent"# HERE YOU SHOULD PUT THE PATH TO THE RAW JSONLs

print("\nReading inputs...")
t0 = time.time()
inputs_C = parse_inputs(f"{DIR_C}/llama33fw_1agent_batch_*.jsonl",
                        r"expC_llama33fw__dilemma_(\d+)__cohort_(\d+)_([\w-]+)__agent_(\d+)")
print(f"  {len(inputs_C):,} inputs in {time.time()-t0:.0f}s")

print("Reading outputs...")
t0 = time.time()
outputs_C, n_total_C, n_invalid_C = parse_outputs_fireworks(
    f"{DIR_C}/batch_downloads/*.jsonl",
    r"expC_llama33fw__dilemma_(\d+)__cohort_(\d+)_([\w-]+)__agent_(\d+)")
print(f"  {n_total_C:,} parsed ({n_invalid_C} invalid) in {time.time()-t0:.0f}s")

print("\nAggregating per (cohort, dimension)...")
counts_C = defaultdict(lambda: {'spared_positive':0, 'total':0})
sig_for_cohort = {}
for cid, m_out, choice in outputs_C:
    dil_id = int(m_out.group(1))
    coh = int(m_out.group(2))
    sig = m_out.group(3)
    sig_for_cohort[coh] = sig
    if cid not in inputs_C: continue
    _, _, a_is_canon0 = inputs_C[cid]
    killed_c0 = (choice == 'A') if a_is_canon0 else (choice == 'B')
    spared_c0 = not killed_c0
    for dim_name, pos_first in dilemma_info[dil_id][2]:
        key = (coh, dim_name)
        counts_C[key]['total'] += 1
        if pos_first == spared_c0:
            counts_C[key]['spared_positive'] += 1

SIG_DECODE = {
    'age': {'e':'elderly','y':'young'},
    'gender': {'m':'man','w':'woman'},
    'education': {'he':'high','le':'low'},
    'income': {'hi':'high','li':'low'},
    'politics': {'sc':'conservative','sp':'progressive'},
    'religion': {'dr':'religious','n':'non-religious'},
}
GEO_BY_POS = ['African','East Asian','Eastern European','Middle Eastern',
              'North American','Oceanian','South American','South Asian','Western European']
def decode(coh, sig):
    parts = sig.split('-')
    return {
        'age': SIG_DECODE['age'].get(parts[0]),
        'gender': SIG_DECODE['gender'].get(parts[1]),
        'education': SIG_DECODE['education'].get(parts[2]),
        'income': SIG_DECODE['income'].get(parts[3]),
        'politics': SIG_DECODE['politics'].get(parts[4]),
        'religion': SIG_DECODE['religion'].get(parts[5]),
        'geography': GEO_BY_POS[coh % 9],
    }

rows = []
for (coh, dim), c in counts_C.items():
    sig = sig_for_cohort.get(coh, '')
    demo = decode(coh, sig)
    rows.append({'cohort_m':coh, **demo, 'amce_dimension':dim,
                 'spared_positive':c['spared_positive'], 'total':c['total']})
df_C = pd.DataFrame(rows)
print(f"  Per-cohort rows: {len(df_C):,} (expected 4,608 = 576 × 8)")

OUT_CSV = "_per_cohort_amce_expC_llama.csv"
df_C.to_csv(OUT_CSV, index=False)
print(f"  Saved {OUT_CSV}")

# ===== REPORT =====
print()
print("=" * 110)
print("AGGREGATE AMCE RATES — 5-way comparison")
print("=" * 110)
ORIG_NANO_PERSONA = {'Age':0.4815,'Age_old':0.4563,'Fragile':0.4945,'Gender':0.4826,
                     'Law_jaywalking':0.4611,'Law_legal':0.4516,'Species':0.4297,'Status':0.6187}
NANO_NOPERSONA = {'Age':0.4442,'Age_old':0.5108,'Fragile':0.4819,'Gender':0.5720,
                  'Law_jaywalking':0.4150,'Law_legal':0.4198,'Species':0.4449,'Status':0.6282}
HUMAN_AMCE = {'Age':0.497,'Gender':0.119,'Species':0.585,'Status':0.345,'Law_legal':0.353}
HUMAN_RATE = {k: 0.5 + v/2 for k, v in HUMAN_AMCE.items()}

print(f"{'Dim':<18}{'nano-no-pers':>14}{'nano-persona':>14}{'Llama-no-pers':>16}{'Llama-persona':>16}{'human':>10}")
print("-" * 90)
agg_A_L = {}
agg_C_L = {}
for dim in sorted(counts_A_L.keys()):
    a_rate = counts_A_L[dim]['spare_pos'] / counts_A_L[dim]['total']
    agg_A_L[dim] = a_rate
    c_sub = df_C[df_C['amce_dimension']==dim]
    c_rate = c_sub['spared_positive'].sum() / c_sub['total'].sum()
    agg_C_L[dim] = c_rate
    nano_np = NANO_NOPERSONA.get(dim)
    nano_p = ORIG_NANO_PERSONA.get(dim)
    hum = HUMAN_RATE.get(dim)
    print(f"{dim:<18}"
          f"{(f'{nano_np:.4f}' if nano_np else '—'):>14}"
          f"{(f'{nano_p:.4f}' if nano_p else '—'):>14}"
          f"{a_rate:>15.4f} "
          f"{c_rate:>15.4f} "
          f"{(f'{hum:.4f}' if hum is not None else '—'):>9}")

# Direction match counts vs humans
print(f"\nDirection match vs humans (rate > 0.5 ↔ AMCE > 0):")
match_nano_p = 0; match_llama_p = 0; match_llama_a = 0; n_dims = 0
for dim in HUMAN_RATE:
    human_pos = HUMAN_RATE[dim] > 0.5
    if dim not in ORIG_NANO_PERSONA: continue
    n_dims += 1
    if (ORIG_NANO_PERSONA[dim] > 0.5) == human_pos: match_nano_p += 1
    if (agg_C_L.get(dim, 0.5) > 0.5) == human_pos: match_llama_p += 1
    if (agg_A_L.get(dim, 0.5) > 0.5) == human_pos: match_llama_a += 1
print(f"  gpt-4.1-nano persona:  {match_nano_p}/{n_dims} match")
print(f"  Llama 3.3 70B persona: {match_llama_p}/{n_dims} match")
print(f"  Llama 3.3 70B no-pers: {match_llama_a}/{n_dims} match")

# η² for C (Llama persona)
print()
print("=" * 110)
print("η² VARIANCE DECOMPOSITION — Llama 3.3 70B (Exp C) vs DeepSeek vs gpt-4.1-nano (original)")
print("=" * 110)
demo_cols = ['age','gender','education','income','politics','religion','geography']
dims_list = sorted(df_C['amce_dimension'].unique())
pivot = df_C.pivot_table(index='cohort_m', columns='amce_dimension',
                          values='spared_positive', aggfunc='sum')
pivot_t = df_C.pivot_table(index='cohort_m', columns='amce_dimension',
                           values='total', aggfunc='sum')
rate_matrix = pivot / pivot_t
demo_df = df_C.drop_duplicates('cohort_m')[['cohort_m']+demo_cols].set_index('cohort_m')
rate_with_demo = rate_matrix.join(demo_df)

def eta_sq(data, factor, dim):
    groups = data.groupby(factor)[dim]
    grand = data[dim].mean()
    ss_b = sum(len(g) * (g.mean() - grand)**2 for _, g in groups)
    ss_t = ((data[dim] - grand)**2).sum()
    return ss_b/ss_t if ss_t > 0 else 0

ORIG_NANO_ETA = {'politics':0.342,'religion':0.134,'education':0.097,
                 'age':0.081,'gender':0.078,'geography':0.047,'income':0.039}
DEEPSEEK_ETA  = {'politics':0.371,'religion':0.103,'education':0.066,
                 'age':0.069,'gender':0.041,'geography':0.051,'income':0.045}

print(f"\n{'Factor':<14}{'η² Llama':>14}{'η² DeepSeek':>14}{'η² nano':>14}{'L − nano':>14}")
print("-" * 70)
llama_etas = {}
for f in ['politics','religion','education','age','gender','geography','income']:
    etas = [eta_sq(rate_with_demo, f, d) for d in dims_list]
    mean_l = float(np.mean(etas))
    llama_etas[f] = mean_l
    ds = DEEPSEEK_ETA[f]
    orig = ORIG_NANO_ETA[f]
    print(f"{f:<14}{mean_l:>13.4f} {ds:>13.4f} {orig:>13.4f} {mean_l-orig:>+13.4f}")

print(f"\nPolitics/Geography ratio:")
print(f"  gpt-4.1-nano:    {ORIG_NANO_ETA['politics']/ORIG_NANO_ETA['geography']:.2f}×")
print(f"  DeepSeek V3.1:   {DEEPSEEK_ETA['politics']/DEEPSEEK_ETA['geography']:.2f}×")
print(f"  Llama 3.3 70B:   {llama_etas['politics']/llama_etas['geography']:.2f}×")

# Caricature zones
print()
print("=" * 110)
print("CARICATURE ZONES (η² > 0.2 per cell) — Llama 3.3 70B")
print("=" * 110)
n_caric = 0
n_pol = 0
per_cell_eta = []
for f in demo_cols:
    for d in dims_list:
        e = eta_sq(rate_with_demo, f, d)
        per_cell_eta.append((f, d, e))
        if e > 0.2:
            n_caric += 1
            if f == 'politics':
                n_pol += 1
            print(f"  {f:<12} × {d:<18}  η²={e:.3f}")
print(f"\nTotal caricature zones: {n_caric} (vs nano 10, DeepSeek 13)")
print(f"  Politics-driven:      {n_pol} (vs nano 6, DeepSeek 7)")

# Monotonicity
print()
print("=" * 110)
print("POLITICAL MONOTONICITY — Llama 3.3 70B (Exp C)")
print("=" * 110)
merge_cols = [c for c in demo_cols if c != 'politics']
rwd = rate_with_demo.reset_index()
con = rwd[rwd['politics']=='conservative']
prog = rwd[rwd['politics']=='progressive']
merged = con.merge(prog, on=merge_cols, suffixes=('_con','_pro'))
print(f"Matched pairs: {len(merged)}")
ORIG_NANO_MONO = {'Age':100.0,'Age_old':100.0,'Fragile':100.0,'Gender':100.0,
                  'Status':100.0,'Species':92.7,'Law_legal':52.1,'Law_jaywalking':52.6}
DEEPSEEK_MONO = {'Age':99.0,'Age_old':99.7,'Fragile':100.0,'Gender':99.0,
                 'Status':99.7,'Species':99.0,'Law_legal':99.3,'Law_jaywalking':99.0}
print(f"\n{'Dim':<18}{'Mono % Llama':>14}{'Direction':>22}{'Mono % DS':>14}{'Mono % nano':>14}")
print("-" * 80)
for dim in dims_list:
    con_h = (merged[f'{dim}_con']>merged[f'{dim}_pro']).sum()
    pro_h = (merged[f'{dim}_con']<merged[f'{dim}_pro']).sum()
    tot = con_h + pro_h
    mono = max(con_h,pro_h)/tot*100 if tot else 0
    dirn = "Conservative higher" if con_h>pro_h else ("Progressive higher" if pro_h>con_h else "tied")
    ds = DEEPSEEK_MONO.get(dim, 0)
    on = ORIG_NANO_MONO.get(dim, 0)
    print(f"{dim:<18}{mono:>12.1f}%  {dirn:>22}{ds:>12.1f}% {on:>12.1f}%")

# PCA on 5 core dims using correlation matrix
print()
print("=" * 110)
print("DIMENSIONAL COLLAPSE — PCA on 5 core dims, correlation matrix")
print("=" * 110)
from numpy.linalg import svd
core5 = ['Age', 'Gender', 'Species', 'Status', 'Law_legal']
M = rate_with_demo[core5].values
# Correlation-matrix PCA: standardize each dim to unit variance
M_z = (M - M.mean(axis=0, keepdims=True)) / M.std(axis=0, keepdims=True, ddof=0)
_, S, _ = svd(M_z, full_matrices=False)
var = (S ** 2) / (S ** 2).sum()
print(f"PC variance explained (Llama 3.3 70B, correlation matrix):")
for i, v in enumerate(var):
    print(f"  PC{i+1}: {v*100:.1f}%")
print(f"  (nano: PC1=68.4%; DeepSeek: PC1=53.4%)")
# Also report covariance-matrix PCA for transparency
M_c = M - M.mean(axis=0, keepdims=True)
_, Sc, _ = svd(M_c, full_matrices=False)
varc = (Sc ** 2) / (Sc ** 2).sum()
print(f"\n(for reference) Same data with covariance matrix (no standardization):")
print(f"  PC1: {varc[0]*100:.1f}%   PC2: {varc[1]*100:.1f}%   "
      "— sensitive to differing dimension variances; not the protocol used in the paper.")

print()
print("Done.")
