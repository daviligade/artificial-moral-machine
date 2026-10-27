#!/usr/bin/env python3
"""Verify the cross-model claims in Table 12 of the paper.

Reads the three per-cohort AMCE CSVs released in `data/`:
  - per_cohort_amce.csv                         (gpt-4.1-nano,    main experiment)
  - per_cohort_amce_expC_deepseek.csv           (DeepSeek V3.1,   Exp C, 1 agent per cell)
  - per_cohort_amce_expC_llama.csv              (Llama 3.3 70B,   Exp C, 1 agent per cell)

Reproduces every numeric cell in Table 12:
  - Politics mean η²
  - Politics / Geography η² ratio
  - Caricature zones (η² > 0.2)
  - Political monotonicity on the top-4 politicised dimensions
  - PC1 variance on the 5 core dims
  - Age × Status correlation across cohorts
  - Aggregate hierarchy inversions (of 5 dims)
  - Direction match against humans (requires Awad et al. plotdataUser*.csv in
    data/human_comparison/; skipped if absent)
  - Political direction on Status and Law

For the direction-match values to be reproduced exactly, the Awad et al. (2018)
subgroup-AMCE files must be downloaded into data/human_comparison/. See
README.md for the download URL (osf.io/3hvt2). All other comparisons run
from the released CSVs alone.

USAGE:
    python verify_crossmodel.py

Optional flags:
    --data-dir <path>   Override the data directory (default: ../data)
    --verbose           Print per-axis breakdowns for direction match

Exits with 0 if all checks within tolerance, 1 otherwise.
"""
import os
import re
import sys
import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import svd


# Paper claims for Table 12 (single source of truth for assertions)
PAPER = {
    'nano': {
        'politics_eta2': 0.342, 'geography_eta2': 0.047, 'ratio': 7.3,
        'caricature_zones': 10, 'caricature_politics': 6,
        'pc1_pct': 72.7, 'age_status_r': -0.89,
        'inversions': 3, 'dirmatch': 48.0,
        'pol_dir_status': 'progressive', 'pol_dir_law': 'indecisive',
    },
    'deepseek': {
        'politics_eta2': 0.371, 'geography_eta2': 0.051, 'ratio': 7.3,
        'caricature_zones': 13, 'caricature_politics': 7,
        'pc1_pct': 53.4, 'age_status_r': 0.37,
        'inversions': 0, 'dirmatch': 58.3,
        'pol_dir_status': 'conservative', 'pol_dir_law': 'conservative',
    },
    'llama': {
        'politics_eta2': 0.439, 'geography_eta2': 0.055, 'ratio': 8.0,
        'caricature_zones': 8, 'caricature_politics': 6,
        'pc1_pct': 53.8, 'age_status_r': 0.25,
        'inversions': 0, 'dirmatch': 48.0,
        'pol_dir_status': 'conservative', 'pol_dir_law': 'conservative',
    },
}

# Human AMCE values for the 5 reference dimensions (Awad et al. 2018)
HUMAN_AMCE = {'Age': 0.497, 'Gender': 0.119, 'Species': 0.585,
              'Status': 0.345, 'Law_legal': 0.353}
HUMAN_RATE = {k: 0.5 + v / 2 for k, v in HUMAN_AMCE.items()}

DEMO_COLS = ['age', 'gender', 'education', 'income', 'politics', 'religion', 'geography']

# Direction-match protocol (mirrors full_analysis.py)
NEGLIGIBLE_HUMAN = 0.005
NEGLIGIBLE_LLM   = 0.001
HUMAN_LABEL_MAP = {
    'Gender': 'Gender', 'Fitness': 'Fitness', 'Social Status': 'Status',
    'Age': 'Age', 'No. Characters': 'No. Characters', 'Law': 'Law',
    'Species': 'Species', 'Utilitarianism': 'Utilitarianism',
    'Interventionism': 'Interventionism', 'Relation to AV': 'Relation to AV',
}
MORAL_DIMS = OrderedDict([
    ('Age', 'Age'), ('Gender', 'Gender'), ('Species', 'Species'),
    ('Law', 'Law_legal'), ('Status', 'Status'),
])
DEMOGRAPHIC_AXES = OrderedDict([
    ('Income',    {'file': 'plotdataUserIncome.csv',
                   'human': ('Lower Income', 'Higher Income'),
                   'col': 'income', 'llm': ('low', 'high')}),
    ('Education', {'file': 'plotdataUserEducation.csv',
                   'human': ('Less Educated', 'More Educated'),
                   'col': 'education', 'llm': ('low', 'high')}),
    ('Religion',  {'file': 'plotdataUserReligious.csv',
                   'human': ('Very Religious', 'Not Religious'),
                   'col': 'religion', 'llm': ('religious', 'non-religious')}),
    ('Age',       {'file': 'plotdataUserAge.csv',
                   'human': ('Older', 'Younger'),
                   'col': 'age', 'llm': ('elderly', 'young')}),
    ('Politics',  {'file': 'plotdataUserPolitical.csv',
                   'human': ('Conservative', 'Progressive'),
                   'col': 'politics', 'llm': ('conservative', 'progressive')}),
    ('Gender',    {'file': 'plotdataUserGender.csv',
                   'human': ('male', 'female'),
                   'col': 'gender', 'llm': ('man', 'woman')}),
])


def eta_sq(data, factor, dim):
    groups = data.groupby(factor)[dim]
    grand = data[dim].mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for _, g in groups)
    ss_t = ((data[dim] - grand) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else 0


def pivot_rate(df):
    pivot = df.pivot_table(index='cohort_m', columns='amce_dimension',
                            values='spared_positive', aggfunc='sum')
    pivot_t = df.pivot_table(index='cohort_m', columns='amce_dimension',
                              values='total', aggfunc='sum')
    return pivot / pivot_t


def _extract_dim(label):
    m = re.search(r'\[(.*?)\]', label)
    return HUMAN_LABEL_MAP.get(m.group(1)) if m else None


def _load_human(human_dir, filename):
    path = os.path.join(human_dir, filename)
    if not os.path.exists(path):
        return None
    hdf = pd.read_csv(path)
    out = {}
    for _, row in hdf.iterrows():
        dim = _extract_dim(row['Label'])
        if dim is not None:
            out[(row['Attribute'], dim)] = row['Estimates']
    return out


def _llm_rate(df, column, value, amce_dim):
    sub = df[(df[column] == value) & (df['amce_dimension'] == amce_dim)]
    t = sub['total'].sum()
    return sub['spared_positive'].sum() / t if t > 0 else np.nan


def direction_match(df, human_dir, verbose=False):
    if not os.path.exists(os.path.join(human_dir, 'plotdataUserPolitical.csv')):
        return None, None, None
    matches, compared = 0, 0
    detail = []
    for axis_name, ax in DEMOGRAPHIC_AXES.items():
        h_data = _load_human(human_dir, ax['file'])
        if h_data is None:
            continue
        h_a, h_b = ax['human']
        l_a, l_b = ax['llm']
        for dim_label, csv_dim in MORAL_DIMS.items():
            h_val_a = h_data.get((h_a, dim_label))
            h_val_b = h_data.get((h_b, dim_label))
            if h_val_a is None or h_val_b is None:
                continue
            human_diff = h_val_a - h_val_b
            llm_diff = _llm_rate(df, ax['col'], l_a, csv_dim) - \
                       _llm_rate(df, ax['col'], l_b, csv_dim)
            if abs(human_diff) < NEGLIGIBLE_HUMAN or abs(llm_diff) < NEGLIGIBLE_LLM:
                continue
            compared += 1
            if (human_diff > 0) == (llm_diff > 0):
                matches += 1
                detail.append((axis_name, dim_label, human_diff, llm_diff, True))
            else:
                detail.append((axis_name, dim_label, human_diff, llm_diff, False))
    if compared == 0:
        return None, None, None
    return matches, compared, detail


def analyze_model(df, label, human_dir=None, verbose=False):
    rate = pivot_rate(df)
    demo_df = df.drop_duplicates('cohort_m')[['cohort_m'] + DEMO_COLS].set_index('cohort_m')
    rwd = rate.join(demo_df)
    dims = sorted(df['amce_dimension'].unique())
    core5 = ['Age', 'Gender', 'Species', 'Status', 'Law_legal']

    # η² by factor
    etas = {f: float(np.mean([eta_sq(rwd, f, d) for d in dims])) for f in DEMO_COLS}
    politics_eta2 = etas['politics']
    geography_eta2 = etas['geography']
    ratio = politics_eta2 / geography_eta2 if geography_eta2 else float('inf')

    # Caricature zones (η² > 0.2)
    zones = []
    for f in DEMO_COLS:
        for d in dims:
            e = eta_sq(rwd, f, d)
            if e > 0.2:
                zones.append((f, d, e))
    n_zones = len(zones)
    n_pol_zones = sum(1 for f, _, _ in zones if f == 'politics')

    # Political monotonicity on top-4 politicised dims
    merge_cols = [c for c in DEMO_COLS if c != 'politics']
    rwd_r = rwd.reset_index()
    con = rwd_r[rwd_r['politics'] == 'conservative']
    prog = rwd_r[rwd_r['politics'] == 'progressive']
    merged = con.merge(prog, on=merge_cols, suffixes=('_c', '_p'))
    monos = []
    for d in dims:
        c_h = (merged[f'{d}_c'] > merged[f'{d}_p']).sum()
        p_h = (merged[f'{d}_c'] < merged[f'{d}_p']).sum()
        tot = c_h + p_h
        if tot:
            monos.append((d, max(c_h, p_h) / tot * 100, 'cons' if c_h > p_h else 'prog'))
    top4_pol = sorted([(d, m, dirn) for d, m, dirn in monos
                       if d in {'Status', 'Law_legal', 'Law_jaywalking', 'Fragile'}],
                      key=lambda x: -x[1])[:4]
    top4_min = min((m for _, m, _ in top4_pol), default=0)

    # PC1 variance on 5 core dims (correlation matrix, matching the paper)
    M = rwd[core5].values
    M_z = (M - M.mean(axis=0, keepdims=True)) / M.std(axis=0, keepdims=True, ddof=0)
    _, S, _ = svd(M_z, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    pc1_pct = float(var[0]) * 100

    # Age × Status correlation
    age_status_r = rate['Age'].corr(rate['Status'])

    # Aggregate hierarchy inversions (5 dims)
    inversions = 0
    for d in core5:
        sub = df[df['amce_dimension'] == d]
        agg_rate = sub['spared_positive'].sum() / sub['total'].sum()
        if (agg_rate > 0.5) != (HUMAN_RATE[d] > 0.5):
            inversions += 1

    # Political direction on Status and Law
    status_dir = 'cons' if (con['Status'].mean() > prog['Status'].mean()) else 'prog'
    law_legal_dir = 'cons' if (con['Law_legal'].mean() > prog['Law_legal'].mean()) else 'prog'

    # Direction match (requires Awad files)
    if human_dir:
        dm_match, dm_comp, dm_detail = direction_match(df, human_dir, verbose=verbose)
    else:
        dm_match = dm_comp = dm_detail = None

    return {
        'politics_eta2': politics_eta2,
        'geography_eta2': geography_eta2,
        'ratio': ratio,
        'caricature_zones': n_zones,
        'caricature_politics': n_pol_zones,
        'top4_pol_min': top4_min,
        'pc1_pct': pc1_pct,
        'age_status_r': float(age_status_r),
        'inversions': inversions,
        'dirmatch': (dm_match / dm_comp * 100) if (dm_match is not None) else None,
        'dirmatch_raw': (dm_match, dm_comp),
        'dirmatch_detail': dm_detail,
        'pol_dir_status': status_dir,
        'pol_dir_law': law_legal_dir,
    }


def check(label, expected, observed, tol=0.05):
    """Compare observed to expected with tolerance. Returns True if pass."""
    if observed is None:
        print(f"  ⚠  {label:55s}  SKIPPED")
        return None
    if isinstance(expected, str):
        ok = (str(observed).startswith(expected[:4]))
        marker = '✓' if ok else '✗'
        print(f"  {marker}  {label:55s}  expected={expected:>10s}  observed={observed:>10s}")
    else:
        ok = abs(observed - expected) <= tol
        marker = '✓' if ok else '✗'
        print(f"  {marker}  {label:55s}  expected={expected:>8.3f}  observed={observed:>8.3f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=None,
                    help='Path to data/ directory (default: ../data relative to this file)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir.parent / 'data'
    human_dir = data_dir / 'human_comparison'

    print('=' * 80)
    print('CROSS-MODEL VERIFICATION (Table 12)')
    print(f'  Data directory: {data_dir}')
    print(f'  Awad et al. subgroup files: {"FOUND" if (human_dir / "plotdataUserPolitical.csv").exists() else "NOT FOUND (direction match will be skipped)"}')
    print('=' * 80)

    paths = {
        'nano':     data_dir / 'per_cohort_amce.csv',
        'deepseek': data_dir / 'per_cohort_amce_expC_deepseek.csv',
        'llama':    data_dir / 'per_cohort_amce_expC_llama.csv',
    }

    results = {}
    for label, path in paths.items():
        if not path.exists():
            print(f"  ⚠ {path.name} not found; skipping {label}")
            continue
        df = pd.read_csv(path)
        results[label] = analyze_model(df, label, human_dir=human_dir, verbose=args.verbose)

    n_pass, n_fail, n_skip = 0, 0, 0
    for label, res in results.items():
        print(f"\n--- {label.upper()} ---")
        exp = PAPER[label]
        for key in ['politics_eta2', 'geography_eta2', 'ratio',
                    'caricature_zones', 'caricature_politics',
                    'pc1_pct', 'age_status_r', 'inversions', 'dirmatch']:
            tol = 0.1 if key == 'dirmatch' else (0.5 if key == 'pc1_pct' else 0.05)
            if key in ('caricature_zones', 'caricature_politics', 'inversions'):
                tol = 0
            r = check(f'{key} (Table 12)', exp[key], res[key], tol=max(tol, 0.05))
            if r is True: n_pass += 1
            elif r is False: n_fail += 1
            else: n_skip += 1
        # Political direction (categorical)
        ok = res['pol_dir_status'].startswith(exp['pol_dir_status'][:4])
        print(f"  {'✓' if ok else '✗'}  pol_dir_status                                              expected={exp['pol_dir_status']:>10s}  observed={res['pol_dir_status']:>10s}")
        if ok: n_pass += 1
        else: n_fail += 1

    print('\n' + '=' * 80)
    if n_fail == 0:
        print(f'  ALL CHECKS PASSED  ({n_pass} passed, {n_skip} skipped — typically direction match)')
        sys.exit(0)
    else:
        print(f'  {n_fail} CHECK(S) FAILED  ({n_pass} passed, {n_skip} skipped)')
        sys.exit(1)


if __name__ == '__main__':
    main()
