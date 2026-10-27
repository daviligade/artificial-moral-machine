# The Artificial Moral Machine

This repository contains research code, small-to-medium datasets, analysis outputs, and supplementary materials for the published paper. **The complete archival dataset, including the two large Parquet files, is hosted on OSF**: https://doi.org/10.17605/OSF.IO/Z3WGQ. GitHub is a companion copy, not a replacement for the OSF archive.

**"The Artificial Moral Machine: Do LLMs Mirror Human Moral Diversity or Merely Caricature It?"**


## Overview

We present the largest-scale empirical test of algorithmic fidelity in the moral domain, conditioning `gpt-4.1-nano-2025-04-14` (the smallest model in the GPT-4.1 family) on 576 demographic cohort profiles across 5,356 trolley-style dilemmas and collecting over 61 million moral judgments. Our multivariate analysis framework documents a **political master switch**: persona conditioning collapses moral variance along a single political axis (politics η² = 0.342, 7.3× more than geography). Three robustness checks confirm and qualify this finding: (i) a no-persona baseline shows aggregate moral biases are largely model defaults; (ii) a qualifier-stripped re-run (politics/religion intensifiers removed) leaves political dominance intact (η² = 0.420); (iii) two cross-model replications on `DeepSeek V3.1` (671B-MoE) and `Llama 3.3 70B` (dense) reproduce — and on Llama, sharpen — the master switch (η² = 0.371 and **0.439**; Politics/Geography ratio = 7.3× and **8.0×**). The specific *direction* of the political caricature is model-dependent: both larger models converge on the typical real-world stereotype (conservatives spare elites and lawful pedestrians), whereas nano produces an internally-inverted caricature. Aggregate hierarchy inversion is nano-specific (3/5 inversions on nano → 0/5 in both larger models); dimensional collapse weakens substantially in both larger models (PC1, correlation matrix: 72.7% → 53.4% on DeepSeek, 53.8% on Llama) but remains substantial relative to a no-collapse baseline of ~20%.


## Where to find the two large Parquet files

The following files are intentionally **not included in this GitHub repository**. Obtain the real, nonempty versions from the [OSF archive](https://osf.io/z3wgq/files/osfstorage):

- `data/merged_dilemma_data_POLARS.parquet` — primary experiment, approximately 0.9 GB (size stated in the original OSF README).
- `data/merged_dilemma_data_POLARS__1agent_subsample.parquet` — smaller subsample, approximately 68 MB (size stated in the original OSF README).

Download each file from OSF, put it in the local `data/` directory with **exactly that filename**, and run the commands below from the repository root. **Do not upload zero-byte placeholder files**: the real data must be downloaded from OSF.

Scripts that only use the released CSV aggregate files may work without either large Parquet file. Regenerating aggregates from raw main-experiment responses and the moral-flips analysis requires the full main Parquet file.


### Hugging Face Datasets

The research datasets are also available on [Hugging Face](https://huggingface.co/datasets/davideliga/artificial-moral-machine), where they can be browsed and loaded directly using the Python `datasets` library.

Hugging Face provides an alternative distribution format for convenient programmatic access. The OSF repository remains the authoritative archival source.

**Dataset access:** https://huggingface.co/datasets/davideliga/artificial-moral-machine

**Archival DOI:** https://doi.org/10.17605/OSF.IO/Z3WGQ

**Associated publication:** For academic citation, please cite the associated scientific publication: https://doi.org/10.1609/aies.v9i3.43257


## Repository Structure

```
├── README.md
├── LICENSE                                                  # MIT license for original code
├── DATA_LICENSE.txt                                         # CC BY 4.0 notice for original datasets
├── CITATION.cff                                             # Citation metadata
├── data/
│   │
│   │   ── Experiment design (shared across all models) ──
│   ├── dilemmas.csv                                         #   5,356 unique moral dilemma pairs
│   ├── cohorts.csv                                          #   576 demographic cohort profiles
│   ├── situation_variants.csv                               #   67 situation phrasing variants
│   ├── decision_variants.csv                                #   6 decision instruction variants
│   │
│   │   ── gpt-4.1-nano (primary model) ──
│   ├── merged_dilemma_data_POLARS.parquet                   #   OSF ONLY: main experiment, raw responses (61.7M rows, ~0.9 GB)
│   ├── merged_dilemma_data_POLARS__1agent_subsample.parquet #   OSF ONLY: 1-agent-per-cell subsample of main (3.08M rows, 68 MB)
│   ├── merged_dilemma_data_POLARS__expB_nano_qualifier_stripped.parquet
│   │                                                        #   Exp B (qualifier-stripped) raw responses (3.08M rows)
│   ├── per_cohort_amce.csv                                  #   Main aggregates: 576 cohorts × 8 AMCE dimensions
│   ├── per_cohort_amce_expB_qualifier_stripped.csv          #   Exp B aggregates: 576 × 8
│   ├── per_dim_amce_nopersona_nano.csv                      #   No-persona baseline: 8 rows (per-dimension aggregate)
│   │
│   │   ── DeepSeek V3.1 (cross-model replication) ──
│   ├── merged_dilemma_data_POLARS__expC_deepseek.parquet    #   Exp C (persona, 1 agent per cell) raw (3.08M rows)
│   ├── per_cohort_amce_expC_deepseek.csv                    #   Exp C aggregates: 576 × 8
│   ├── per_dim_amce_nopersona_deepseek.csv                  #   No-persona baseline aggregates
│   │
│   │   ── Llama 3.3 70B (cross-model replication) ──
│   ├── merged_dilemma_data_POLARS__expC_llama.parquet       #   Exp C (persona, 1 agent per cell) raw (3.08M rows)
│   ├── per_cohort_amce_expC_llama.csv                       #   Exp C aggregates: 576 × 8
│   ├── per_dim_amce_nopersona_llama.csv                     #   No-persona baseline aggregates
│   │
│   └── human_comparison/
│       └── summary_overall_preferences.csv                  #   Third-party Takemoto data; see licensing exception
│
└── code/
    ├── full_analysis.py                                     #   Verifies every quantitative claim about gpt-4.1-nano against the data
    ├── verify_crossmodel.py                                 #   Reproduces Table 12 (cross-model) from the three per_cohort_amce CSVs
    ├── regenerate_per_cohort_amce.py                        #   Re-derives per_cohort_amce.csv from the main parquet
    ├── analyze_moral_flips.py                               #   Reproduces moral-flips analysis (Table 8) from the main parquet
    ├── regenerate_paper_figures.py                          #   Regenerates all six paper figures from the released data
    ├── analyze_deepseek_experiments.py                      #   Derives DeepSeek aggregates from raw batch JSONLs (raw not redistributed)
    ├── analyze_llama_experiments.py                         #   Derives Llama aggregates from raw batch JSONLs (raw not redistributed)
    ├── regenerate_per_cohort_amce_expB.py                   #   Regenerates the qualifier-stripped per-cohort AMCE CSV
    ├── analyze_nopersona_experiments.py                     #   Derives the three no-persona per-dim CSVs from raw batches (raw not redistributed)
    └── requirements.txt
```


## What Is Released, Per Experiment

The released data is organised by **model** (`gpt-4.1-nano`, `DeepSeek V3.1`, `Llama 3.3 70B`) and **experiment** (main / Exp B qualifier-stripped / Exp C persona / no-persona baseline). For each cell we release either *raw per-judgment responses* (a parquet, one row per LLM call), *aggregated per-cohort AMCE counts* (a per_cohort CSV with 576 × 8 = 4,608 rows), or *per-dimension aggregates* (an 8-row CSV):

| Model           | Experiment                            | Raw responses (parquet) | Per-cohort aggregates (CSV) | Per-dimension aggregates (CSV) |
|-----------------|---------------------------------------|--------------------------|------------------------------|--------------------------------|
| gpt-4.1-nano    | Main (20 agents/cell, 61.7M queries)  | ✓ full + 1-agent subset  | ✓ `per_cohort_amce.csv`      | (derivable from CSV)           |
| gpt-4.1-nano    | Exp B qualifier-stripped (1 agent/cell, 3M)  | ✓ `__expB_nano_qualifier_stripped.parquet` | ✓ `per_cohort_amce_expB_qualifier_stripped.csv` | (derivable) |
| gpt-4.1-nano    | No-persona baseline (107K queries)    | —                        | —                            | ✓ `per_dim_amce_nopersona_nano.csv` |
| DeepSeek V3.1   | Exp C persona (1 agent/cell, 3M)      | ✓ `__expC_deepseek.parquet` | ✓ `per_cohort_amce_expC_deepseek.csv` | (derivable) |
| DeepSeek V3.1   | No-persona baseline (107K queries)    | —                        | —                            | ✓ `per_dim_amce_nopersona_deepseek.csv` |
| Llama 3.3 70B   | Exp C persona (1 agent/cell, 3M)      | ✓ `__expC_llama.parquet` | ✓ `per_cohort_amce_expC_llama.csv` | (derivable) |
| Llama 3.3 70B   | No-persona baseline (107K queries)    | —                        | —                            | ✓ `per_dim_amce_nopersona_llama.csv` |

The analyses can be reproduced from the published OSF archive and the code in `code/` (see *Reproducing the Analysis* below).


## Dataset Description

### Raw response parquets

The OSF archive contains five parquet files; each contains raw per-judgment LLM responses. Three are included here and two large files must be downloaded separately.

**`merged_dilemma_data_POLARS.parquet`** (the full nano main experiment, 61,701,120 rows × 26 columns, ~0.9 GB). One row per LLM response:

| Column | Description |
|--------|-------------|
| `dilemma_id` | Unique identifier for the dilemma pair (0–5,355) |
| `cohort_id` | Unique identifier for the demographic cohort (0–575) |
| `agent_id` | Prompt-variant index within a cell (0–19) |
| `canonical_decision` | The LLM's binary choice (`A` or `B` in canonical order, or `None` if unparseable) |
| `raw_prompt` | The full prompt sent to the API (unique per agent) |
| `raw_output` | The raw API response text |
| `choice_a`, `choice_b` | Text descriptions of the two choices as presented (order randomised per cell) |
| `canonical_choice_0`, `canonical_choice_1` | Canonical (alphabetically sorted) choice descriptions |
| `cohort_signature` | String encoding of the 7-dimensional cohort profile |
| `age`, `gender`, `education`, `income`, `politics`, `religion`, `geography` | Individual demographic dimensions |
| `entity_A`, `entity_B` | Character descriptions for each side of the dilemma |
| `primary_contrast` | The primary moral dimension being tested |
| `tags_A_str`, `tags_B_str` | Morally salient tags for each character |
| `A_minus_B`, `B_minus_A`, `Shared_tags` | Tag set differences and intersection |

**`merged_dilemma_data_POLARS__1agent_subsample.parquet`** (3,085,056 rows × 26 columns, 68 MB). A deterministic 1-agent-per-cell subsample of the main experiment, same schema as above. Use this when you don't need the full 20-agent replication.

**`merged_dilemma_data_POLARS__expB_nano_qualifier_stripped.parquet`** (3,085,056 × 14), **`merged_dilemma_data_POLARS__expC_deepseek.parquet`** (3,085,056 × 14), and **`merged_dilemma_data_POLARS__expC_llama.parquet`** (3,085,056 × 14). The raw per-judgment responses for the three 1-agent-per-cell experiments. They use a **reduced schema** (14 columns) without the heaviest text fields (`raw_prompt`, `raw_output`, `entity_A/B`, `tags_*`) to keep file size small. Retained columns:

| Column | Description |
|--------|-------------|
| `dilemma_id`, `cohort_id`, `agent_id` | Same as main parquet |
| `canonical_decision` | The LLM's choice in canonical order (`A`, `B`, or `None`) |
| `canonical_choice_0`, `canonical_choice_1` | Canonical choice descriptions (joined from `dilemmas.csv`) |
| `cohort_signature`, `age`, `gender`, `education`, `income`, `politics`, `religion`, `geography` | Cohort metadata (joined from `cohorts.csv`) |

If you need the dropped text fields (e.g., to audit a specific raw response), they can be reconstructed by joining `cohort_id` with `cohorts.csv` (for prompts) and applying the tag-extraction logic in `analyze_moral_flips.py`.

### Per-cohort aggregate CSVs

`per_cohort_amce.csv`, `per_cohort_amce_expB_qualifier_stripped.csv`, `per_cohort_amce_expC_deepseek.csv`, and `per_cohort_amce_expC_llama.csv` each contain 4,608 rows = 576 cohorts × 8 AMCE dimensions:

| Column | Description |
|--------|-------------|
| `cohort_m` | Cohort identifier (0–575) |
| `age`, `gender`, `education`, `income`, `politics`, `religion`, `geography` | Cohort demographics |
| `amce_dimension` | One of: `Age`, `Age_old`, `Fragile`, `Gender`, `Law_jaywalking`, `Law_legal`, `Species`, `Status` |
| `spared_positive` | Count of responses that spared the positive side of the dimension |
| `total` | Total comparable responses for the cohort × dimension cell |

The aggregate sparing rate is `spared_positive / total`. These four files are sufficient to reproduce every numeric claim in Table 12, monotonicity, caricature zones, PCA, and the cohort-level direction-match analyses.

### No-persona aggregate CSVs

`per_dim_amce_nopersona_nano.csv`, `per_dim_amce_nopersona_deepseek.csv`, and `per_dim_amce_nopersona_llama.csv` each contain 8 rows (one per AMCE dimension), showing the aggregate sparing rate when the model is prompted **without** a demographic preamble:

| Column | Description |
|--------|-------------|
| `amce_dimension` | One of the 8 AMCE dimensions |
| `spared_positive` | Count of responses that spared the positive side |
| `total` | Total comparable responses |
| `rate` | `spared_positive / total` |

These reproduce Table 9 (nano with-persona vs. no-persona comparison) and the statement that DeepSeek and Llama no-persona aggregates align directionally with humans on all 5 reference dimensions.

### Experiment design files

These four files encode the complete combinatorial structure of the experiment. Together with the randomisation protocol described in the paper, they allow full replication with any LLM.

- **`dilemmas.csv`** (5,356 rows): each row is a unique moral dilemma, defined by a canonical pair of choices (alphabetically sorted). In the actual prompts, A/B ordering is randomised per (dilemma, cohort) cell.
- **`cohorts.csv`** (576 rows): each row is a unique demographic cohort profile with the persona prompt text and parsed demographic dimensions (age, gender, education, income, politics, religion, geography).
- **`situation_variants.csv`** (67 rows): the 67 situation phrasing variants (e.g., "Now consider a self-driving car encountering an unavoidable accident, with two options:"). Each prompt randomly draws one.
- **`decision_variants.csv`** (6 rows): the 6 decision instruction variants (e.g., "You are asked to make this decision. What should the self-driving car do?"). Each prompt randomly draws one.

A prompt is assembled as: `[persona] + [situation variant] + [Choice A: ...] + [Choice B: ...] + [decision variant] + [format instruction]`. For each (dilemma, cohort) cell, 20 such prompts are generated in the nano main experiment with independently sampled situation and decision variants (1 prompt per cell in Exp B and the Exp C runs).

### Human comparison data (`human_comparison/`)

`summary_overall_preferences.csv` contains aggregate AMCE values across multiple LLMs and humans from Takemoto (2024) and Ahmad & Takemoto (2025), available at [github.com/kztakemoto/mmllm](https://github.com/kztakemoto/mmllm) (MIT License).

The direction-match analyses in `full_analysis.py` and `verify_crossmodel.py` additionally require demographic subgroup AMCE estimates from the original Moral Machine experiment (Awad et al. 2018). These files are **not included** in this repository (the original OSF project does not specify a redistribution license). To enable the direction-match checks, download the following files from [osf.io/3hvt2](https://osf.io/3hvt2/) and place them in `data/human_comparison/`:

- `plotdataUserAge.csv`
- `plotdataUserEducation.csv`
- `plotdataUserGender.csv`
- `plotdataUserIncome.csv`
- `plotdataUserPolitical.csv`
- `plotdataUserReligious.csv`
- `country_cluster_map.csv`

If they are absent, both verification scripts gracefully skip the direction-match section and continue verifying everything else.


## Experiment Parameters

- **Primary model:** `gpt-4.1-nano-2025-04-14` (OpenAI Batch API; smallest model in the GPT-4.1 family).
- **Cross-model replications:** `DeepSeek V3.1` (671B-parameter MoE, ~37B active per forward pass) and `Llama 3.3 70B` (dense), both via Fireworks Batch API.
- **Temperature:** 0 (deterministic decoding) for all experiments.
- **Cohort profiles:** 576 (2 ages × 2 genders × 2 education levels × 2 income levels × 2 political orientations × 2 religious orientations × 9 geographic backgrounds).
- **Unique dilemmas:** 5,356 (from 104 unique characters, paired combinatorially as 104 choose 2).
- **Prompt variants per (dilemma, cohort) cell:** 20 in the main `gpt-4.1-nano` experiment; 1 in Exp B (qualifier-stripped, nano) and Exp C (DeepSeek, Llama).
- **Total API queries:** 61,701,120 (nano main) + 3,085,056 (nano Exp B) + 3,085,056 (DeepSeek Exp C) + 3,085,056 (Llama Exp C) + 107,120 × 3 (no-persona baselines) ≈ 71.3M judgments.


## Reproducing the Analysis

All analyses can be reproduced from the released parquet and CSV files alone (the raw batch JSONLs are not redistributed; the released parquets are derived from them).

```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. (Optional) Download Awad et al. human baseline data to enable direction-match checks.
#    Place plotdataUser*.csv and country_cluster_map.csv into data/human_comparison/.
#    See instructions above. Without these, the direction-match checks are skipped
#    and all other checks still run.

# 3. Verify every quantitative claim about gpt-4.1-nano against the data.
#    Checks aggregate AMCE rates, eta-squared, PCA, monotonicity, k-means clusters,
#    direction match, caricature zones, and prose claims (~74 numeric assertions).
python code/full_analysis.py

# 4. Verify every quantitative claim in Table 12 (three-model cross-comparison).
#    Reads the three per_cohort_amce CSVs (nano, DeepSeek, Llama) and checks
#    politics eta-squared, Politics/Geography ratio, caricature zones, monotonicity,
#    PC1, Age x Status correlation, aggregate inversions, direction match, etc.
python code/verify_crossmodel.py

# 5. (Optional) Re-derive per_cohort_amce.csv from the full parquet to confirm consistency.
#    NOTE: requires ~8 GB RAM to load 3 columns from the 61.7M-row parquet.
python code/regenerate_per_cohort_amce.py

# 6. (Optional) Reproduce the moral-flips analysis (Table 8, Figure 5).
python code/analyze_moral_flips.py

# 7. (Optional) Regenerate all six paper figures from the released data.
#    Figure 3 (direction-match heatmap) requires the Awad et al. files
#    from step 2; if they are absent, that figure is skipped and the
#    other five still render.
python code/regenerate_paper_figures.py
```

Steps 3 and 4 are the headline verification: they confirm every numeric claim in the paper against the released data. With the Awad et al. files installed (step 2), all 74 checks in `full_analysis.py` and all 30 checks in `verify_crossmodel.py` should pass with 0 failures. Without those files, the Awad-dependent direction-match checks (~29 in `full_analysis.py`, 3 in `verify_crossmodel.py`) are skipped and the rest still pass.


## Citation

If you use or build upon the datasets, code, or research materials provided in this repository, please cite our associated scientific publication:

> Liga, D., Yu, L., Slavkovik, M., Touileb, S., Liao, B., & van der Torre, L. (2026). **The Artificial Moral Machine: Do LLMs Mirror Human Moral Diversity or Merely Caricature It?** *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, **9**(3), 1678–1688. https://doi.org/10.1609/aies.v9i3.43257

**BibTeX (copy into your Overleaf `.bib` file):**

```bibtex
@inproceedings{liga2026artificial,
  author    = {Liga, Davide and Yu, Liuwen and Slavkovik, Marija and Touileb, Samia and Liao, Beishui and van der Torre, Leendert},
  title     = {The Artificial Moral Machine: Do {LLMs} Mirror Human Moral Diversity or Merely Caricature It?},
  booktitle = {Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society},
  volume    = {9},
  number    = {3},
  pages     = {1678--1688},
  year      = {2026},
  doi       = {10.1609/aies.v9i3.43257},
  url       = {https://doi.org/10.1609/aies.v9i3.43257}
}
```

You can also download this reference as [`paper.bib`](paper.bib) and cite it in LaTeX using `\cite{liga2026artificial}`.

For reproducibility and data provenance, we also encourage citing the archived dataset:

**OSF Dataset DOI:** https://doi.org/10.17605/OSF.IO/Z3WGQ

For the dataset's authoritative metadata, consult the [OSF record](https://doi.org/10.17605/OSF.IO/Z3WGQ).

GitHub's **Cite this repository** feature uses [`CITATION.cff`](CITATION.cff) to identify the paper as the preferred citation.

These academic citation requests are separate from the legal attribution and notice requirements of the applicable licenses.

## License

- **Original source code** in `code/`: [MIT License](LICENSE). Preserve its copyright and permission notices when redistributing code.
- **Original datasets** in `data/`: [Creative Commons Attribution 4.0 International (CC BY 4.0)](DATA_LICENSE.txt), except where third-party rights or a more specific notice apply. Attribution is required by CC BY 4.0 when its conditions apply.
- **Third-party materials:** `data/human_comparison/summary_overall_preferences.csv` originates from [kztakemoto/mmllm](https://github.com/kztakemoto/mmllm). The upstream project describes itself as MIT-licensed, but the license status of this particular CSV has not been independently confirmed. The file is **excluded** from our CC BY 4.0 grant; preserve applicable upstream notices and attribution. Other material from external providers is not relicensed by this notice.

The root `LICENSE` file grants MIT rights over the original source code only; it does **not** relicense datasets or third-party materials.


## Acknowledgments

This work uses human baseline data from:
- Awad, E. et al. (2018). The Moral Machine experiment. *Nature*, 563(7729), 59–64. Data available at [osf.io/3hvt2](https://osf.io/3hvt2/) (not redistributed here; see above for download instructions).
- Takemoto, K. (2024). The moral machine experiment on large language models. *Royal Society Open Science*, 11(2), 231393. Data available at [github.com/kztakemoto/mmllm](https://github.com/kztakemoto/mmllm) (MIT License; `summary_overall_preferences.csv` included with attribution).
