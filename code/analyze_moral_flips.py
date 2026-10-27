#!/usr/bin/env python3
"""
Reproduce the moral-flips analysis from the released parquet.

Produces:
    1. Significant contrast counts per demographic axis (Fisher's Exact Test
       with Benjamini-Hochberg FDR correction, α = 0.05, requiring sparing
       preference to reverse across groups; reported in the paper:
       Politics 740, Religion 556, Age 491, Gender 470, Education 346,
       Income 312).
    2. Top moral flips by |Δ| (paper Table 8).
    3. The illustrative criminal-jaywalking vs. young-legal example shown
       in Figure 5 of the paper.

Definitions:
  - For each (dilemma_id, demographic_axis) pair:
      * Split cohorts into the two groups defined by the axis (binary axes:
        politics, religion, age, gender, education, income).
      * Aggregate spare-canonical_choice_0 counts and totals across all
        cohorts in each group.
      * Compute group_rate = spare_0_count / total.
  - "Moral flip": the two group rates lie on opposite sides of 0.5.
  - "Significant contrast": Fisher's exact p < threshold after BH-FDR
    correction at α = 0.05, performed PER axis (separate FDR family for
    each axis).
  - Top flips: rank by |group1_rate − group2_rate| over all dilemma×axis
    cells that are both significant AND a moral flip.

Geography (9-valued, not binary) is excluded from this script's per-axis
moral-flip analysis. Two reasons: (i) the binary-axis flip definition
(group A spares one side, group B spares the other) does not generalise
to a 9-valued factor without an arbitrary choice of pairwise contrast or
summary statistic; (ii) geography's η² = 0.047 in Table 2 already
establishes that it trails every binary axis. The script logs the geography 
level count (STEP 1) for transparency but does not produce per-dilemma 
geography test statistics.

Inputs (relative to script):
  ../data/merged_dilemma_data_POLARS.parquet (61M rows)
  ../data/per_cohort_amce.csv                (576 cohorts × 8 dims; cohort demographics)
  ../data/dilemmas.csv                       (5,356 dilemma texts; for top-flip labelling)

Outputs (stdout + analyze_moral_flips_results.txt):
  - Counts per axis (significant + moral flips)
  - Top moral flips by |Δ|
  - Criminal-jaywalking vs. young-legal example (per-cohort breakdown)

Cost: ~32K Fisher's exact tests + one streaming pass over the parquet.
"""
import os, time
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import fisher_exact
from collections import defaultdict

OUT = "analyze_moral_flips_results.txt"
PARQUET = "../data/merged_dilemma_data_POLARS.parquet"
COHORTS_CSV = "../data/per_cohort_amce.csv"
DILEMMAS_CSV = "../data/dilemmas.csv"

log_lines = []
def L(s=""):
    print(s); log_lines.append(s)

def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR. Returns boolean array of significance."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    # find largest k such that ranked[k] <= (k+1)/n * alpha
    crit = np.arange(1, n+1) / n * alpha
    below = ranked <= crit
    if not below.any():
        sig_idx = np.array([], dtype=int)
    else:
        kmax = np.where(below)[0].max()
        sig_idx = order[:kmax+1]
    sig = np.zeros(n, dtype=bool)
    sig[sig_idx] = True
    return sig

# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Load cohort demographic table
# ──────────────────────────────────────────────────────────────────────
L("=" * 78)
L("STEP 1 — Load cohort demographics")
L("=" * 78)
demo_cols = ['age','gender','education','income','politics','religion','geography']
coh = (pd.read_csv(COHORTS_CSV)[['cohort_m'] + demo_cols]
         .drop_duplicates('cohort_m')
         .rename(columns={'cohort_m':'cohort_id'}))
L(f"Loaded {len(coh)} cohorts")
for c in demo_cols:
    vals = sorted(coh[c].unique())
    L(f"  {c}: {vals}")

# Binary axes (have exactly 2 values)
binary_axes = [c for c in demo_cols if coh[c].nunique() == 2]
L(f"Binary axes for paired comparisons: {binary_axes}")
L(f"Geography (multi-valued): {coh['geography'].nunique()} levels")

# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Aggregate parquet to (dilemma_id, cohort_id, spare_0, total)
# ──────────────────────────────────────────────────────────────────────
L()
L("=" * 78)
L("STEP 2 — Aggregate parquet to per-cell counts (dilemma × cohort)")
L("=" * 78)
t0 = time.time()
agg = (pl.scan_parquet(PARQUET)
         .select(["dilemma_id","cohort_id","canonical_decision"])
         .filter(pl.col("canonical_decision").is_in(["A","B"]))
         .with_columns((pl.col("canonical_decision")=="B").cast(pl.Int32).alias("spare_0"))
         .group_by(["dilemma_id","cohort_id"])
         .agg([pl.col("spare_0").sum().alias("spare_0"),
               pl.col("spare_0").count().alias("total")])
         .collect(engine="streaming"))
L(f"Aggregated in {time.time()-t0:.1f}s; {len(agg):,} cells")

# Convert to pandas for ease
agg_df = agg.to_pandas()
agg_df = agg_df.merge(coh, on='cohort_id', how='left')

# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Per-dilemma, per-axis: compute Fisher's exact + group rates
# ──────────────────────────────────────────────────────────────────────
L()
L("=" * 78)
L("STEP 3 — Compute Fisher's exact + spare-rates per (dilemma × binary axis)")
L("=" * 78)

# For each axis, build a frame: dilemma_id, group, spare_0, total
records = []  # (dilemma_id, axis, val1, val2, r1, r2, p, n1, n2)
t0 = time.time()
for axis in binary_axes:
    grouped = (agg_df.groupby(['dilemma_id', axis])
                     .agg({'spare_0':'sum', 'total':'sum'})
                     .reset_index())
    pivot = grouped.pivot(index='dilemma_id', columns=axis, values=['spare_0','total'])
    vals = sorted(coh[axis].unique())
    v1, v2 = vals[0], vals[1]
    s1 = pivot[('spare_0', v1)].values
    s2 = pivot[('spare_0', v2)].values
    n1 = pivot[('total', v1)].values
    n2 = pivot[('total', v2)].values
    dilemma_ids = pivot.index.values

    # spare-0 rates
    r1 = s1 / np.where(n1>0, n1, 1)
    r2 = s2 / np.where(n2>0, n2, 1)

    # Fisher's exact per dilemma
    pvals = np.empty(len(dilemma_ids))
    for i in range(len(dilemma_ids)):
        # 2x2: rows = group, cols = (spare_0, kill_0)
        table = np.array([[s1[i], n1[i] - s1[i]],
                          [s2[i], n2[i] - s2[i]]])
        try:
            _, p = fisher_exact(table)
        except Exception:
            p = 1.0
        pvals[i] = p

    sig_mask = bh_fdr(pvals, alpha=0.05)

    for i, did in enumerate(dilemma_ids):
        records.append({
            'dilemma_id': int(did),
            'axis': axis,
            'val1': v1, 'val2': v2,
            'r1': float(r1[i]), 'r2': float(r2[i]),
            'n1': int(n1[i]), 'n2': int(n2[i]),
            'p': float(pvals[i]),
            'significant': bool(sig_mask[i]),
            'delta': float(r1[i] - r2[i]),
        })
    elapsed = time.time() - t0
    L(f"  {axis}: {len(dilemma_ids)} dilemmas, {sig_mask.sum()} significant after BH-FDR  (cumulative {elapsed:.0f}s)")

flips_df = pd.DataFrame(records)
L(f"\nTotal records (dilemma × axis): {len(flips_df):,}")

# ──────────────────────────────────────────────────────────────────────
# STEP 4 — Counts: significant contrasts and moral flips per axis
# ──────────────────────────────────────────────────────────────────────
L()
L("=" * 78)
L("STEP 4 — Significant contrasts and moral flips per axis")
L("=" * 78)

# Moral flip = rates on opposite sides of 0.5
flips_df['flip'] = ((flips_df['r1'] > 0.5) & (flips_df['r2'] < 0.5)) | \
                   ((flips_df['r1'] < 0.5) & (flips_df['r2'] > 0.5))

L(f"\n{'Axis':<14}{'Sig':>8}{'Flips':>8}{'Sig+Flip':>10}")
L("-" * 40)
total_sig = 0; total_flips = 0; total_sig_flip = 0
for axis in binary_axes:
    sub = flips_df[flips_df['axis']==axis]
    nsig = int(sub['significant'].sum())
    nflip = int(sub['flip'].sum())
    nsf = int((sub['significant'] & sub['flip']).sum())
    total_sig += nsig; total_flips += nflip; total_sig_flip += nsf
    L(f"{axis:<14}{nsig:>8,}{nflip:>8,}{nsf:>10,}")
L("-" * 40)
L(f"{'TOTAL':<14}{total_sig:>8,}{total_flips:>8,}{total_sig_flip:>10,}")
L()
L("Paper: (binary axes, no effect-size threshold beyond Fisher/BH-FDR + the moral-flip criterion): "
  "Politics 740, Religion 556, Age 491, Gender 470, Education 346, Income 312")
L(f"  Sum = {740+556+491+470+346+312} (paper)")

# ──────────────────────────────────────────────────────────────────────
# STEP 5 — Top moral flips by |Δ|
# ──────────────────────────────────────────────────────────────────────
L()
L("=" * 78)
L("STEP 5 — Top moral flips by |Δ| (compare to paper Table 8)")
L("=" * 78)

# Filter to flips that are also significant
top = flips_df[flips_df['significant'] & flips_df['flip']].copy()
top['absdelta'] = top['delta'].abs()
top = top.sort_values('absdelta', ascending=False).head(20)

# Load dilemma texts to add labels
dilemmas = pd.read_csv(DILEMMAS_CSV)
top = top.merge(dilemmas, on='dilemma_id', how='left')

L(f"\n{'rank':<5}{'axis':<11}{'val1':<14}{'val2':<14}{'r1':>8}{'r2':>8}{'|Δ|':>8}  contrast")
L("-" * 130)
for rank, (_, r) in enumerate(top.iterrows(), 1):
    contrast = f"{r['canonical_choice_0'][:50]} | {r['canonical_choice_1'][:50]}"
    L(f"{rank:<5}{r['axis']:<11}{r['val1']:<14}{r['val2']:<14}{r['r1']:>7.3f} {r['r2']:>7.3f} {r['absdelta']:>7.3f}  {contrast}")
L()
L("Paper Table 8 top 5 (for reference; all politics-driven):")
L("  1. elite, multiple > female, young, legal, multiple        | prog: A=82.0  con: B=74.8  Δ=56.8  N=5,756")
L("  2. elite, male, legal, multiple > female, young, multiple  | prog: A=82.3  con: B=71.4  Δ=53.7  N=5,607")
L("  3. elite, legal, multiple > female, young, multiple        | prog: A=59.5  con: B=91.0  Δ=50.5  N=5,736")
L("  4. fragile (child), jaywalking, multiple > old, multiple   | con:  A=66.2  prog:B=83.6  Δ=49.8  N=5,760")
L("  5. fragile (child), multiple > elite, legal, multiple      | con:  A=84.4  prog:B=65.0  Δ=49.4  N=5,759")
L("Convention: A(%) is the spare-A rate in the Favored-A group; B(%) is the spare-B rate in the Favored-B group; Δ = A% + B% - 100.")

# ──────────────────────────────────────────────────────────────────────
# STEP 6 — Criminal-jaywalking vs young-legal example (Figure 5)
# ──────────────────────────────────────────────────────────────────────
L()
L("=" * 78)
L("STEP 6 — Criminal-jaywalking vs young-legal example (paper Figure 5)")
L("=" * 78)

# Find dilemmas matching: one side has criminal+jaywalking, other has young+legal
def has(d, terms):
    d = d.lower()
    return all(t in d for t in terms)

matched = []
for did, c0, c1 in dilemmas.itertuples(index=False):
    a_crim = has(c0, ['criminal','jaywalking'])
    b_crim = has(c1, ['criminal','jaywalking'])
    a_yl = has(c0, ['young','legally']) and 'criminal' not in c0.lower()
    b_yl = has(c1, ['young','legally']) and 'criminal' not in c1.lower()
    # also accept "child crossing legally" as young-legal (child has young tag)
    a_child_legal = has(c0, ['child','legally'])
    b_child_legal = has(c1, ['child','legally'])
    a_is_yl = a_yl or a_child_legal
    b_is_yl = b_yl or b_child_legal
    if a_crim and b_is_yl:
        matched.append((did, True))   # criminal first (canonical_choice_0)
    elif b_crim and a_is_yl:
        matched.append((did, False))  # criminal second
L(f"Matching dilemmas (criminal+jaywalking vs young/child+legal): {len(matched)}")

# Compute spare-criminal rate by politics
sub_pol = flips_df[(flips_df['axis']=='politics')].set_index('dilemma_id')
# Need raw counts to combine across dilemmas; use agg_df
matched_dids = [d for d, _ in matched]
m_agg = agg_df[agg_df['dilemma_id'].isin(matched_dids)].copy()
# For each row, "spare criminal" = spare_0 if criminal_first else (total - spare_0)
crim_first_map = {d: cf for d, cf in matched}
m_agg['crim_first'] = m_agg['dilemma_id'].map(crim_first_map)
m_agg['spare_crim'] = np.where(m_agg['crim_first'], m_agg['spare_0'], m_agg['total'] - m_agg['spare_0'])

# Aggregate by politics
res = m_agg.groupby('politics').agg({'spare_crim':'sum','total':'sum'}).reset_index()
res['rate'] = res['spare_crim'] / res['total']
L(f"\nAcross all {len(matched)} matched dilemmas:")
for _, r in res.iterrows():
    L(f"  {r['politics']}: spare-criminal rate = {r['rate']:.4f}  ({r['rate']*100:.2f}%)  n={r['total']:,}")

# Per-dilemma breakdown
L(f"\nPer-dilemma rates (showing all {len(matched)}):")
per_dil = (m_agg.groupby(['dilemma_id','politics'])
                .agg({'spare_crim':'sum','total':'sum'}).reset_index())
per_dil['rate'] = per_dil['spare_crim'] / per_dil['total']
piv = per_dil.pivot(index='dilemma_id', columns='politics', values='rate')
piv['n_per_group'] = per_dil.pivot(index='dilemma_id', columns='politics', values='total')['conservative']
piv['delta'] = piv['progressive'] - piv['conservative']
piv = piv.sort_values('delta', ascending=False)
for did, row in piv.iterrows():
    cd = next(d for d in dilemmas.itertuples(index=False) if d.dilemma_id == did)
    L(f"  d_id={did:>4}  con={row['conservative']:.3f}  prog={row['progressive']:.3f}  Δ={row['delta']:+.3f}  ({cd[1][:35]} | {cd[2][:35]})")

L()
L("Paper Figure 5: conservatives spare the legal pedestrians 65.1% (i.e., spare the criminals 34.9%), progressives spare the criminals 58.8%, n=5,760 per group; 24-pp gap on dilemma 4973 (matches the computed rates above).")

# ──────────────────────────────────────────────────────────────────────
# Write output
# ──────────────────────────────────────────────────────────────────────
with open(OUT, 'w') as f:
    f.write("\n".join(log_lines))
L(f"\nResults written to {OUT}")
