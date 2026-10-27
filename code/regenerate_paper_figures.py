#!/usr/bin/env python3
"""Regenerate all six figures of the paper from the released data.

  Figure 1: Variance decomposition — mean η² by demographic factor on gpt-4.1-nano.
  Figure 2: PCA biplot — PC1 vs PC2 for the 5 core Moral Machine dimensions.
  Figure 3: Direction-match heatmap — 6 axes × 5 dims vs Awad et al. humans.
            (REQUIRES Awad et al. plotdataUser*.csv files in data/human_comparison/;
             see README. The figure is skipped if the files are absent.)
  Figure 4: Aggregate preferences — nano sparing rates vs human reference.
  Figure 5: Politics example — criminal-jaywalking vs young-legal (dilemma 4973).
  Figure 6: Three-model cross-comparison — gpt-4.1-nano, DeepSeek V3.1, Llama 3.3 70B.

Inputs (relative to script location):
  ../data/per_cohort_amce.csv                           (Figures 1–4)
  ../data/per_cohort_amce_expC_deepseek.csv             (Figure 6)
  ../data/per_cohort_amce_expC_llama.csv                (Figure 6)
  ../data/human_comparison/plotdataUser*.csv            (Figure 3, optional)

Outputs (in ../figures/, created if needed):
  fig_variance_decomposition.{pdf,png}
  fig_pca_loadings.{pdf,png}
  fig_direction_match_heatmap.{pdf,png}        (if Awad files present)
  fig_aggregate_preferences.{pdf,png}
  politics__antisocial_jaywalking__vs__legal_young.{pdf,png}
  fig_crossmodel_comparison.{pdf,png}

Usage:  python regenerate_paper_figures.py
"""
import os, time
os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):
    time.tzset()
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrow, FancyBboxPatch
from numpy.linalg import svd


# ─── Style ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    11,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        110,
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.15,
})

COLOR_RED      = '#C44740'
COLOR_BLUE     = '#3878B5'
COLOR_ORG      = '#E47C32'
COLOR_GRY      = '#909090'
COLOR_GREEN    = '#5BA85B'
COLOR_LRED     = '#D87674'
COLOR_LGRY     = '#D5D5D5'
COLOR_PURPLE   = '#7E3D9C'
COLOR_DEEPSEEK = '#C44740'
COLOR_LLAMA    = '#2F8E3E'
COLOR_HUMAN    = '#909090'
COLOR_LIGHT    = '#E2E2E2'


# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')
HUMAN_DIR  = os.path.join(DATA_DIR, 'human_comparison')
OUT_DIR    = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, basename):
    """Save a figure as both .pdf and .png in OUT_DIR."""
    pdf = os.path.join(OUT_DIR, basename + '.pdf')
    png = os.path.join(OUT_DIR, basename + '.png')
    fig.savefig(pdf)
    fig.savefig(png)
    print(f'  saved {pdf}')
    print(f'  saved {png}')


# ─── Shared helpers ────────────────────────────────────────────────────────
def eta_sq(data, factor, dim):
    """Compute η² of `factor` on `dim` (one-way ANOVA proportion)."""
    groups = data.groupby(factor)[dim]
    grand = data[dim].mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for _, g in groups)
    ss_t = ((data[dim] - grand) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else 0


def pivot_rates(csv_path, demo_cols=('age', 'gender', 'education', 'income',
                                      'politics', 'religion', 'geography')):
    """Return rate matrix indexed by cohort_m with demographic columns joined."""
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index='cohort_m', columns='amce_dimension',
                            values='spared_positive', aggfunc='sum')
    pivot_t = df.pivot_table(index='cohort_m', columns='amce_dimension',
                              values='total', aggfunc='sum')
    rate = pivot / pivot_t
    demo = df.drop_duplicates('cohort_m').set_index('cohort_m')[list(demo_cols)]
    return rate.join(demo)


# ─── FIGURE 1 — Variance decomposition (two-model) ────────────────────────
def figure_1(nano_rate, ds_rate):
    """Side-by-side η² bars: gpt-4.1-nano (solid) vs DeepSeek V3.1 (hatched).
    Matches the paper's Figure 1 layout."""
    print('\n[Figure 1] Variance decomposition (nano + DeepSeek)…')
    factors_full = ['politics', 'religion', 'education', 'age',
                    'gender', 'geography', 'income']
    factor_names = ['Politics', 'Religion', 'Education', 'Age',
                    'Gender', 'Geography', 'Income']

    def _eta_all(rate):
        dims = sorted([c for c in rate.columns if c not in
                       {'age', 'gender', 'education', 'income',
                        'politics', 'religion', 'geography'}])
        return [float(np.mean([eta_sq(rate, f, d) for d in dims]))
                for f in factors_full]

    nano_eta = _eta_all(nano_rate)
    ds_eta   = _eta_all(ds_rate)
    print('  nano η²    :', dict(zip(factor_names, [f'{e:.3f}' for e in nano_eta])))
    print('  DeepSeek η²:', dict(zip(factor_names, [f'{e:.3f}' for e in ds_eta])))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    y = np.arange(len(factor_names))
    h = 0.36

    nano_colors = [COLOR_RED if f == 'Politics' else COLOR_BLUE for f in factor_names]
    ds_colors   = [COLOR_RED if f == 'Politics' else COLOR_BLUE for f in factor_names]

    ax.barh(y - h / 2, nano_eta, height=h, color=nano_colors,
            edgecolor='black', linewidth=0.6, label='_nolegend_')
    ax.barh(y + h / 2, ds_eta, height=h, color=ds_colors,
            edgecolor='black', linewidth=0.6, alpha=0.45, hatch='///',
            label='_nolegend_')

    # Annotations
    ax.text(min(nano_eta[0], ds_eta[0]) / 2, 0,
            'dominant in both LLMs', ha='center', va='center',
            color='white', fontsize=10, style='italic', alpha=0.95)
    geo_i = factor_names.index('Geography')
    ax.text(max(nano_eta[geo_i], ds_eta[geo_i]) + 0.013, geo_i,
            'dominant in humans', ha='left',
            va='center', color=COLOR_GRY, fontsize=10, style='italic')

    ax.set_yticks(y)
    ax.set_yticklabels(factor_names)
    ax.invert_yaxis()
    ax.set_xlabel(r'Mean $\eta^2$ across 8 AMCE dimensions')
    ax.set_xlim(0, 0.42)
    ax.set_xticks(np.arange(0, 0.45, 0.05))
    ax.grid(axis='x', linestyle=':', color='lightgray', alpha=0.7)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=COLOR_RED,  edgecolor='black', label='gpt-4.1-nano (politics)'),
        Patch(facecolor=COLOR_BLUE, edgecolor='black', label='gpt-4.1-nano (other)'),
        Patch(facecolor=COLOR_RED,  edgecolor='black', alpha=0.45, hatch='///',
              label='DeepSeek V3.1 (politics)'),
        Patch(facecolor=COLOR_BLUE, edgecolor='black', alpha=0.45, hatch='///',
              label='DeepSeek V3.1 (other)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right',
              frameon=True, framealpha=0.95, fontsize=9)

    plt.tight_layout()
    _save(fig, 'fig_variance_decomposition')
    plt.close(fig)


# ─── FIGURE 2 — PCA biplot ────────────────────────────────────────────────
def figure_2(rate_with_demo):
    print('\n[Figure 2] PCA biplot (5 core dimensions, correlation matrix)…')
    core5 = ['Age', 'Gender', 'Species', 'Status', 'Law_legal']
    M = rate_with_demo[core5].values
    Mz = (M - M.mean(0)) / M.std(0, ddof=0)
    U, S, Vt = svd(Mz, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    # Vt rows are PC vectors; columns correspond to dimensions in order
    # Loadings = correlations between each PC and each variable
    loadings = Vt.T * np.sqrt(S ** 2 / (len(M) - 1))
    # Normalize so PC1 loadings are in approx [-1, 1] (cosmetic)
    scale = 1.0 / max(abs(loadings[:, :2]).max(), 1e-9)
    loadings = loadings * scale
    pc1, pc2 = var[0] * 100, var[1] * 100
    print(f'  PC1 = {pc1:.1f}%, PC2 = {pc2:.1f}%')

    # Sign-flip so Age points left and Status points right (matches paper)
    age_idx = core5.index('Age')
    if loadings[age_idx, 0] > 0:
        loadings[:, 0] = -loadings[:, 0]
    if loadings[core5.index('Gender'), 1] < 0:
        loadings[:, 1] = -loadings[:, 1]

    pole_a = {'Age', 'Gender'}
    display = {'Age': 'Age', 'Gender': 'Gender', 'Species': 'Species',
               'Status': 'Status', 'Law_legal': 'Law'}

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.axhline(0, color='lightgray', lw=0.7, zorder=0)
    ax.axvline(0, color='lightgray', lw=0.7, zorder=0)

    for i, name in enumerate(core5):
        x, y = loadings[i, 0], loadings[i, 1]
        color = COLOR_RED if name in pole_a else COLOR_BLUE
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=18))
        # Label at arrow tip with offset
        offset = 0.08
        norm = np.hypot(x, y)
        lx = x + (x / norm) * offset if norm else x
        ly = y + (y / norm) * offset if norm else y
        ha = 'left' if x >= 0 else 'right'
        va = 'bottom' if y >= 0 else 'top'
        ax.text(lx, ly, display[name], color=color,
                ha=ha, va=va, fontsize=12, fontweight='bold')

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.55, 1.05)
    ax.set_xticks(np.arange(-1.0, 1.25, 0.5))
    ax.set_yticks(np.arange(-0.4, 1.05, 0.2))
    ax.set_xlabel(f'PC1 ({pc1:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pc2:.1f}% variance)')

    pole_handles = [
        mpl.lines.Line2D([0], [0], color=COLOR_RED, lw=2, label='Pole A (Age, Gender)'),
        mpl.lines.Line2D([0], [0], color=COLOR_BLUE, lw=2,
                          label='Pole B (Status, Species, Law)'),
    ]
    ax.legend(handles=pole_handles, loc='lower left', frameon=True, framealpha=0.95)

    plt.tight_layout()
    _save(fig, 'fig_pca_loadings')
    plt.close(fig)


# ─── FIGURE 3 — Direction match heatmap ───────────────────────────────────
HUMAN_LABEL_MAP = {
    'Gender': 'Gender', 'Fitness': 'Fitness', 'Social Status': 'Status',
    'Age': 'Age', 'No. Characters': 'No. Characters', 'Law': 'Law',
    'Species': 'Species', 'Utilitarianism': 'Utilitarianism',
    'Interventionism': 'Interventionism', 'Relation to AV': 'Relation to AV',
}
MORAL_DIMS = OrderedDict([
    ('Age', 'Age'), ('Gender', 'Gender'), ('Status', 'Status'),
    ('Species', 'Species'), ('Law', 'Law_legal'),
])
DEMOGRAPHIC_AXES = OrderedDict([
    ('Income',    {'file': 'plotdataUserIncome.csv',
                   'human': ('Lower Income', 'Higher Income'),
                   'col': 'income',    'llm': ('low', 'high')}),
    ('Education', {'file': 'plotdataUserEducation.csv',
                   'human': ('Less Educated', 'More Educated'),
                   'col': 'education', 'llm': ('low', 'high')}),
    ('Religion',  {'file': 'plotdataUserReligious.csv',
                   'human': ('Very Religious', 'Not Religious'),
                   'col': 'religion',  'llm': ('religious', 'non-religious')}),
    ('Age',       {'file': 'plotdataUserAge.csv',
                   'human': ('Older', 'Younger'),
                   'col': 'age',       'llm': ('elderly', 'young')}),
    ('Politics',  {'file': 'plotdataUserPolitical.csv',
                   'human': ('Conservative', 'Progressive'),
                   'col': 'politics',  'llm': ('conservative', 'progressive')}),
    ('Gender',    {'file': 'plotdataUserGender.csv',
                   'human': ('male', 'female'),
                   'col': 'gender',    'llm': ('man', 'woman')}),
])
NEGLIGIBLE_HUMAN = 0.005
NEGLIGIBLE_LLM   = 0.001


def _load_human(filename):
    fpath = os.path.join(HUMAN_DIR, filename)
    if not os.path.exists(fpath):
        return None
    hdf = pd.read_csv(fpath)
    out = {}
    for _, row in hdf.iterrows():
        m = re.search(r'\[(.*?)\]', row['Label'])
        dim = HUMAN_LABEL_MAP.get(m.group(1)) if m else None
        if dim:
            out[(row['Attribute'], dim)] = row['Estimates']
    return out


def figure_3(per_cohort_csv):
    print('\n[Figure 3] Direction-match heatmap…')
    if not os.path.exists(os.path.join(HUMAN_DIR, 'plotdataUserPolitical.csv')):
        print('  Awad et al. files not found in data/human_comparison/. Skipping Figure 3.')
        print('  To enable: download plotdataUser*.csv from osf.io/3hvt2 (see README).')
        return

    df = pd.read_csv(per_cohort_csv)
    rate_long = (df.groupby(['amce_dimension', 'politics', 'religion', 'age',
                              'gender', 'education', 'income'])
                  .apply(lambda g: g['spared_positive'].sum() / g['total'].sum())
                  .reset_index(name='rate'))

    def llm_rate(col, val, dim):
        sub = df[(df[col] == val) & (df['amce_dimension'] == dim)]
        t = sub['total'].sum()
        return sub['spared_positive'].sum() / t if t else np.nan

    # Build 6 × 5 verdict matrix
    rows_order = []
    verdict = []  # rows × cols of 'match' / 'mismatch' / 'trivial'
    match_rates = []
    for axis_name, ax_def in DEMOGRAPHIC_AXES.items():
        h_data = _load_human(ax_def['file'])
        if h_data is None:
            continue
        h_a, h_b = ax_def['human']
        l_a, l_b = ax_def['llm']
        col = ax_def['col']
        row = []
        m, c = 0, 0
        for dim_label, csv_dim in MORAL_DIMS.items():
            ha = h_data.get((h_a, dim_label))
            hb = h_data.get((h_b, dim_label))
            if ha is None or hb is None:
                row.append('trivial'); continue
            human_diff = ha - hb
            llm_diff = llm_rate(col, l_a, csv_dim) - llm_rate(col, l_b, csv_dim)
            if abs(human_diff) < NEGLIGIBLE_HUMAN or abs(llm_diff) < NEGLIGIBLE_LLM:
                row.append('trivial')
            elif (human_diff > 0) == (llm_diff > 0):
                row.append('match'); m += 1; c += 1
            else:
                row.append('mismatch'); c += 1
        rows_order.append(axis_name)
        verdict.append(row)
        match_rates.append((m, c))

    # Row order matches the paper's Figure 3:
    # Politics, Gender, Religion, Age, Education, Income.
    # Headline finding (Politics) is placed first, then remaining factors are
    # sorted ascending by match rate.
    paper_order = ['Politics', 'Gender', 'Religion', 'Age', 'Education', 'Income']
    idx_map = {name: i for i, name in enumerate(rows_order)}
    new_idx = [idx_map[n] for n in paper_order if n in idx_map]
    rows_order  = [rows_order[i]  for i in new_idx]
    verdict     = [verdict[i]     for i in new_idx]
    match_rates = [match_rates[i] for i in new_idx]
    cols = list(MORAL_DIMS.keys())

    color_map = {'match': COLOR_GREEN, 'mismatch': COLOR_LRED, 'trivial': COLOR_LGRY}
    marker_map = {'match': '✓', 'mismatch': '✗', 'trivial': '—'}

    fig, ax = plt.subplots(figsize=(8, 5.0))
    for i, row in enumerate(verdict):
        for j, v in enumerate(row):
            ax.add_patch(plt.Rectangle((j, len(verdict) - 1 - i), 1, 1,
                                        facecolor=color_map[v],
                                        edgecolor='white', linewidth=2))
            tc = 'white' if v != 'trivial' else '#666'
            ax.text(j + 0.5, len(verdict) - 1 - i + 0.5, marker_map[v],
                    ha='center', va='center',
                    fontsize=22, color=tc, fontweight='bold')

    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(verdict))
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols)
    yticks = []
    for (m, c) in match_rates:
        rate = m / c * 100 if c else 0
        yticks.append(f'({rate:.0f}%)')
    row_labels = [f'{name}  {tag}' for name, tag in zip(rows_order, yticks)]
    ax.set_yticks(np.arange(len(verdict)) + 0.5)
    ax.set_yticklabels(reversed(row_labels))
    ax.set_xlabel('Moral Dimension')
    ax.set_ylabel('Demographic Factor  (match rate)')
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Patch(facecolor=COLOR_GREEN, label='Match (✓)'),
        Patch(facecolor=COLOR_LRED,  label='Mismatch (✗)'),
        Patch(facecolor=COLOR_LGRY,  label='Trivial (—)'),
    ]
    ax.legend(handles=handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=10)

    plt.tight_layout()
    _save(fig, 'fig_direction_match_heatmap')
    plt.close(fig)


# ─── FIGURE 4 — Aggregate preferences ─────────────────────────────────────
def figure_4(per_cohort_csv):
    print('\n[Figure 4] Aggregate preferences (nano vs Moral Machine humans)…')
    df = pd.read_csv(per_cohort_csv)
    dims = ['Age', 'Gender', 'Species', 'Status', 'Law_legal']
    dim_labels = ['Age\n(spare young)', 'Gender\n(spare female)',
                  'Species\n(spare human)', 'Status\n(spare high-status)',
                  'Law\n(spare lawful)']
    nano_rates = []
    for d in dims:
        sub = df[df['amce_dimension'] == d]
        nano_rates.append(sub['spared_positive'].sum() / sub['total'].sum())

    # Human AMCEs from Awad et al. (2018), translated to rates as 0.5 + AMCE/2
    human_amce = {'Age': 0.497, 'Gender': 0.119, 'Species': 0.585,
                  'Status': 0.345, 'Law_legal': 0.353}
    human_rates = [0.5 + human_amce[d] / 2 for d in dims]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(dims))
    colors = [COLOR_BLUE if r >= 0.5 else COLOR_ORG for r in nano_rates]
    bars = ax.bar(x, nano_rates, color=colors, edgecolor='black',
                   linewidth=0.6, width=0.62, label='_nolegend_')

    for bar, r in zip(bars, nano_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{r:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    ax.scatter(x, human_rates, marker='D', s=130, color=COLOR_PURPLE,
                edgecolor='black', linewidth=0.6, zorder=3, label='_nolegend_')
    for xi, hr in zip(x, human_rates):
        ax.text(xi + 0.18, hr, f'{hr:.3f}', color=COLOR_PURPLE,
                fontsize=10, style='italic', va='center')

    ax.axhline(0.5, ls='--', color=COLOR_GRY, lw=1.0, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylabel('Sparing rate')
    ax.set_ylim(0.30, 0.95)
    ax.set_yticks(np.arange(0.30, 0.96, 0.05))
    ax.grid(axis='y', linestyle=':', color='lightgray', alpha=0.7)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=COLOR_BLUE, edgecolor='black', label='gpt-4.1-nano (above 0.5)'),
        Patch(facecolor=COLOR_ORG, edgecolor='black', label='gpt-4.1-nano (below 0.5)'),
        mpl.lines.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_PURPLE,
                          markeredgecolor='black', markersize=10, label='Human (Moral Machine)'),
        mpl.lines.Line2D([0], [0], color=COLOR_GRY, ls='--', lw=1.0, label='No preference (0.5)'),
    ]
    ax.legend(handles=legend_handles, loc='upper center',
              bbox_to_anchor=(0.5, 1.10), ncol=4,
              frameon=True, framealpha=0.95)

    plt.tight_layout()
    _save(fig, 'fig_aggregate_preferences')
    plt.close(fig)


# ─── FIGURE 5 — Politics example (dilemma 4973) ───────────────────────────
def figure_5():
    print('\n[Figure 5] Politics example (dilemma 4973)…')
    # Per-cohort rates for dilemma 4973 (reproducible from the parquet via
    # analyze_moral_flips.py; values match Figure 5 of the paper).
    con_spare_crim, prog_spare_crim, n = 0.34948, 0.58837, 5760
    delta_pp = abs(con_spare_crim - prog_spare_crim) * 100

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    groups = ['Conservative', 'Progressive']
    rates  = [con_spare_crim, prog_spare_crim]
    colors = [COLOR_RED, COLOR_BLUE]
    bars = ax.bar(groups, rates, color=colors, edgecolor='black',
                   linewidth=0.6, width=0.62)

    for bar, r in zip(bars, rates):
        s = int(round(r * n))
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02,
                f'{r*100:.1f}%\n({s:,} / {n:,})',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.axhline(0.5, ls='--', color=COLOR_GRY, lw=1.2)
    ax.text(1.5, 0.51, 'no preference (0.5)', ha='right', va='bottom',
            color=COLOR_GRY, style='italic', fontsize=9)
    ax.annotate('', xy=(1, prog_spare_crim), xytext=(1, con_spare_crim),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.4))
    ax.text(1.07, (con_spare_crim + prog_spare_crim) / 2,
             f'{delta_pp:.0f} pp\ngap',
             ha='left', va='center', fontweight='bold', fontsize=10)
    ax.text(1.5, 0.93, '↑ Spare the criminal\n(jaywalking)', ha='right', va='top',
             color=COLOR_GRY, style='italic', fontsize=9)
    ax.text(1.5, 0.10, '↓ Spare the legal\nyoung pedestrian', ha='right', va='bottom',
             color=COLOR_GRY, style='italic', fontsize=9)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylabel('Proportion sparing criminal (jaywalking)\nover young pedestrian (legal)')
    ax.grid(axis='y', linestyle=':', color='lightgray', alpha=0.7)
    ax.set_axisbelow(True)
    fig.text(0.5, -0.04,
             f'Symmetric-quantity dilemma (several criminals jaywalking vs. several '
             f'young men crossing legally; n={n:,} per group).\n'
             'Conservative LLM personas chose to kill the criminals (sparing legal '
             'pedestrians) more often, while progressives split closer to chance.',
             ha='center', va='top', fontsize=8.5,
             color=COLOR_GRY, style='italic')

    plt.tight_layout()
    _save(fig, 'politics__antisocial_jaywalking__vs__legal_young')
    plt.close(fig)


# ─── FIGURE 6 — Three-model cross-comparison ──────────────────────────────
def figure_6(nano_csv, ds_csv, ll_csv):
    print('\n[Figure 6] Three-model cross-comparison…')
    ETA = {
        'gpt-4.1-nano':  {'politics': 0.342, 'geography': 0.047},
        'DeepSeek V3.1': {'politics': 0.371, 'geography': 0.051},
        'Llama 3.3 70B': {'politics': 0.439, 'geography': 0.055},
    }
    RATIO = {m: e['politics'] / e['geography'] for m, e in ETA.items()}

    HUMAN_AMCE = {'Age': 0.497, 'Gender': 0.119, 'Species': 0.585,
                   'Status': 0.345, 'Law_legal': 0.353}
    HUMAN_AGG  = {k: 0.5 + v / 2 for k, v in HUMAN_AMCE.items()}

    def agg(csv_path, dims=('Age', 'Gender', 'Species', 'Law_legal', 'Status')):
        df = pd.read_csv(csv_path)
        return {d: df[df['amce_dimension'] == d]['spared_positive'].sum()
                  / df[df['amce_dimension'] == d]['total'].sum()
                for d in dims}

    NANO_AGG = agg(nano_csv)
    DS_AGG   = agg(ds_csv)
    LL_AGG   = agg(ll_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Left panel: master switch (politics + geography η² per model)
    models = ['gpt-4.1-nano', 'DeepSeek V3.1', 'Llama 3.3 70B']
    colors = [COLOR_BLUE, COLOR_DEEPSEEK, COLOR_LLAMA]
    x = np.arange(len(models))
    w = 0.36

    pol = [ETA[m]['politics']  for m in models]
    geo = [ETA[m]['geography'] for m in models]

    ax1.bar(x - w / 2, pol, w, color=colors, edgecolor='black', linewidth=0.6)
    ax1.bar(x + w / 2, geo, w, color=colors, edgecolor='black',
            linewidth=0.6, alpha=0.45, hatch='///')

    for i, m in enumerate(models):
        ax1.text(x[i] - w / 2, pol[i] + 0.012, f'{pol[i]:.3f}',
                  ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        ax1.text(x[i] + w / 2, geo[i] + 0.012, f'{geo[i]:.3f}',
                  ha='center', va='bottom', fontsize=9.5, color='#444')
        ax1.text(x[i], -0.045, f'{RATIO[m]:.1f}×', ha='center', va='top',
                  fontsize=10, fontweight='bold', color=colors[i])

    ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylabel(r'Mean $\eta^2$ across 8 AMCE dimensions')
    ax1.set_ylim(0, max(pol) * 1.18)
    ax1.set_yticks(np.arange(0, 0.55, 0.1))
    ax1.set_title('The master switch persists across model scales\n'
                  '(politics : geography ratio shown below bars)',
                  fontsize=10.5, pad=6)
    ax1.grid(axis='y', linestyle=':', color='lightgray', alpha=0.7)
    ax1.set_axisbelow(True)

    leg1 = [
        Patch(facecolor=COLOR_LIGHT, edgecolor='black', label='Politics $\\eta^2$'),
        Patch(facecolor=COLOR_LIGHT, edgecolor='black', alpha=0.45,
              hatch='///', label='Geography $\\eta^2$'),
    ]
    ax1.legend(handles=leg1, loc='upper left', frameon=True, framealpha=0.95)

    # Right panel: aggregate sparing on 5 human-reference dimensions
    dims = ['Age', 'Gender', 'Species', 'Law_legal', 'Status']
    dim_labels = ['Age\n(young)', 'Gender\n(female)', 'Species\n(human)',
                   'Law\n(legal)', 'Status\n(elite)']

    human  = [HUMAN_AGG[d] for d in dims]
    nano   = [NANO_AGG[d]  for d in dims]
    ds     = [DS_AGG[d]    for d in dims]
    llama  = [LL_AGG[d]    for d in dims]

    x2 = np.arange(len(dims))
    w2 = 0.20

    ax2.bar(x2 - 1.5 * w2, human, w2, color=COLOR_HUMAN, edgecolor='black',
            linewidth=0.6, label='Humans')
    ax2.bar(x2 - 0.5 * w2, nano,  w2, color=COLOR_BLUE, edgecolor='black',
            linewidth=0.6, label='gpt-4.1-nano')
    ax2.bar(x2 + 0.5 * w2, ds,    w2, color=COLOR_DEEPSEEK, edgecolor='black',
            linewidth=0.6, label='DeepSeek V3.1')
    ax2.bar(x2 + 1.5 * w2, llama, w2, color=COLOR_LLAMA, edgecolor='black',
            linewidth=0.6, label='Llama 3.3 70B')

    for i, d in enumerate(dims):
        human_pos = HUMAN_AGG[d] > 0.5
        for (val, xoff) in [(nano[i], -0.5 * w2), (ds[i], 0.5 * w2),
                             (llama[i], 1.5 * w2)]:
            if (val > 0.5) != human_pos:
                ax2.text(x2[i] + xoff, val + 0.02, '✗', ha='center', va='bottom',
                          color='#C44740', fontsize=12, fontweight='bold')

    ax2.axhline(0.5, ls='--', color='#666', lw=1.0, zorder=0)
    ax2.text(len(dims) - 0.5, 0.51, 'no preference (0.5)',
             ha='right', va='bottom', color='#666', style='italic', fontsize=8.5)

    ax2.set_xticks(x2); ax2.set_xticklabels(dim_labels, fontsize=9.5)
    ax2.set_ylabel('Aggregate sparing rate')
    ax2.set_ylim(0, 1.0); ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_title('Aggregate alignment with humans\n'
                  '(red ✗ = inversion: model on opposite side of 0.5)',
                  fontsize=10.5, pad=6)
    ax2.grid(axis='y', linestyle=':', color='lightgray', alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', ncol=1, frameon=True, framealpha=0.95)

    plt.tight_layout()
    _save(fig, 'fig_crossmodel_comparison')
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    nano_csv = os.path.join(DATA_DIR, 'per_cohort_amce.csv')
    ds_csv   = os.path.join(DATA_DIR, 'per_cohort_amce_expC_deepseek.csv')
    ll_csv   = os.path.join(DATA_DIR, 'per_cohort_amce_expC_llama.csv')

    print(f'Reading nano per-cohort AMCE from {nano_csv} ...')
    nano_rate = pivot_rates(nano_csv)
    print(f'  {len(nano_rate)} cohorts × {len([c for c in nano_rate.columns if c not in {"age","gender","education","income","politics","religion","geography"}])} AMCE dimensions')
    print(f'Reading DeepSeek per-cohort AMCE from {ds_csv} ...')
    ds_rate = pivot_rates(ds_csv)
    print(f'  {len(ds_rate)} cohorts × DeepSeek AMCE dimensions')

    figure_1(nano_rate, ds_rate)
    figure_2(nano_rate)
    figure_3(nano_csv)
    figure_4(nano_csv)
    figure_5()
    figure_6(nano_csv, ds_csv, ll_csv)

    print('\nAll figures regenerated.')


if __name__ == '__main__':
    main()
