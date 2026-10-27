#!/usr/bin/env python3
"""
Regenerate per_cohort_amce_expB_qualifier_stripped.csv from the
qualifier-stripped parquet (Exp B: 576-cohort design re-run with milder
qualifiers --- "politically conservative" replacing "strongly conservative",
"religious" replacing "deeply religious"; k=1 agent per cell).

Reuses the same dimension-detection and aggregation logic as
regenerate_per_cohort_amce.py.

Pipeline:
  1. Reconstruct canonical dilemma pairs from the character generation logic
  2. Reconstruct cohort profiles (576 combinations)
  3. Read the parquet (only dilemma_id, cohort_id, canonical_decision columns)
  4. Compute AMCE dimensions per dilemma and aggregate per cohort
  5. Write per_cohort_amce.csv

Input:  data/merged_dilemma_data_POLARS__expB_nano_qualifier_stripped.parquet
Output: data/per_cohort_amce_expB_qualifier_stripped.csv

Requirements: numpy, pandas, polars, pyarrow
Usage: python3 regenerate_per_cohort_amce_expB.py
"""

import os
import sys
import re
import csv
import time
import numpy as np
from itertools import product, combinations
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Input
PARQUET_PATH = os.path.join(BASE_DIR, "data", "merged_dilemma_data_POLARS__expB_nano_qualifier_stripped.parquet")

# Outputs
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "per_cohort_amce_expB_qualifier_stripped.csv")

N_DILEMMAS = 5356
N_COHORTS = 576
N_AGENTS = 1  # Exp B used k=1 agent per cell

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Reconstruct canonical dilemma pairs
# ═══════════════════════════════════════════════════════════════════════════════

print("Step 1: Reconstructing canonical dilemma pairs...")

quantities = ["one", "several"]
genders = ["male", "female", ""]
ages = ["young", "old", ""]
social_characters = ["doctor", "executive", "criminal"]
fragile_characters = ["pregnant woman", "child"]
non_humans_characters = ["pet"]
legality = ["crossing the street legally", "jaywalking", ""]
actions = ['']
outcomes = ['kill']

plural_exceptions = {
    "pregnant woman": "pregnant women", "child": "children",
    "man": "men", "woman": "women", "pedestrian": "pedestrians",
    "pet": "pets",
}

def build_char_desc(quantity, gender, age, social, fragile, non_human, law):
    """Build a character description string (matching the original code logic)."""
    if non_human:
        base = non_human
        if quantity == "several":
            base = plural_exceptions.get(non_human, non_human + "s")
        desc = f"{quantity} {base}".strip()
    elif social:
        base = f"{gender} {social}".strip()
        if quantity == "several":
            if base.endswith("person"):
                base = base.replace("person", "people")
            elif not base.endswith('s'):
                base += "s"
        desc = f"{quantity} {base}".strip()
    elif fragile:
        base = fragile
        if quantity == "several":
            base = plural_exceptions.get(fragile, fragile + "s")
        desc = f"{quantity} {base}".strip()
    else:
        if gender and age:
            if gender == "male": noun = "man" if quantity == "one" else "men"
            elif gender == "female": noun = "woman" if quantity == "one" else "women"
            else: noun = "pedestrian" if quantity == "one" else "pedestrians"
            desc = f"{quantity} {age} {noun}".strip()
        elif gender:
            if gender == "male": noun = "man" if quantity == "one" else "men"
            elif gender == "female": noun = "woman" if quantity == "one" else "women"
            else: noun = "pedestrian" if quantity == "one" else "pedestrians"
            desc = f"{quantity} {noun}".strip()
        else:
            noun = "pedestrian" if quantity == "one" else "pedestrians"
            desc = f"{quantity} {noun}".strip()

    if law and non_human == "":
        verb = "is" if quantity == "one" else "are"
        desc += f" who {verb} {law}"

    return ' '.join(desc.split())

def tag_character(desc):
    """Tag a character description for AMCE dimensions."""
    tags = set()
    d = desc.lower()

    # Species
    tags.add('non-human' if 'pet' in d else 'human')

    # Age
    if 'child' in d or 'young' in d or 'boy' in d or 'girl' in d: tags.add('young')
    if 'old' in d or 'elderly' in d: tags.add('old')

    # Gender
    if any(w in d for w in ['woman', 'women', 'female', 'girl', 'pregnant']): tags.add('female')
    # 'male' is a substring of 'female'; guard against false positives on
    # 'female criminal' etc. by removing the bare 'male' rule.
    if any(w in d for w in [' man ', ' man', ' men ', 'boy']) or d.endswith(' man') or d.endswith(' men'):
        tags.add('male')

    # Status
    if 'doctor' in d or 'executive' in d: tags.add('high-status')
    if 'criminal' in d: tags.add('low-status')

    # Legality
    if 'jaywalking' in d: tags.add('jaywalking')
    if 'legally' in d: tags.add('legal')

    # Fragile
    if 'pregnant' in d or 'child' in d or 'children' in d: tags.add('fragile')

    # Quantity
    if d.startswith('several'): tags.add('several')
    if d.startswith('one '): tags.add('one')

    # Social role type
    if 'doctor' in d: tags.add('prosocial')
    if 'executive' in d: tags.add('elite')
    if 'criminal' in d: tags.add('antisocial')

    return frozenset(tags)

# Generate all choices
all_choices = []

# Social characters
for q, g, s, law in product(quantities, genders, social_characters, legality):
    desc = build_char_desc(q, g, '', s, '', '', law)
    full = f"kill {desc}"
    all_choices.append(full)

# Fragile characters
for q, frag, law in product(quantities, fragile_characters, legality):
    desc = build_char_desc(q, '', '', '', frag, '', law)
    full = f"kill {desc}"
    all_choices.append(full)

# Non-human
for q, nh in product(quantities, non_humans_characters):
    desc = build_char_desc(q, '', '', '', '', nh, '')
    full = f"kill {desc}"
    all_choices.append(full)

# Regular pedestrians
for q, g, a, law in product(quantities, genders, ages, legality):
    if not g: continue
    desc = build_char_desc(q, g, a, '', '', '', law)
    full = f"kill {desc}"
    all_choices.append(full)

all_choices = sorted(set(all_choices))

# Generate pairs
canonical_pairs = []
for a, b in combinations(all_choices, 2):
    if a != b:
        canonical_pairs.append((a, b))

canonical_pairs.sort(key=lambda x: (x[0].lower(), x[1].lower()))

print(f"  Generated {len(all_choices)} unique choices")
print(f"  Generated {len(canonical_pairs)} canonical pairs")

if len(canonical_pairs) != N_DILEMMAS:
    print(f"  WARNING: Expected {N_DILEMMAS} pairs, got {len(canonical_pairs)}")
    print(f"  Will use actual count: {len(canonical_pairs)}")
    N_DILEMMAS = len(canonical_pairs)

# Pre-compute tags for each canonical pair
pair_tags = []
for a_text, b_text in canonical_pairs:
    pair_tags.append((tag_character(a_text), tag_character(b_text)))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Reconstruct cohort profiles
# ═══════════════════════════════════════════════════════════════════════════════

print("Step 2: Reconstructing cohort profiles...")

age_cohort = ['young', 'elderly']
gender_cohort = ['man', 'woman']
education_cohort = ['low education', 'high education']
income_cohort = ['low income', 'high income']
political_cohort = ['strongly conservative', 'strongly progressive']
religious_cohort = ['non-religious', 'deeply religious']
geography_cohort = ['Eastern European', 'Western European', 'North American', 'South American',
                     'African', 'Oceanian', 'Middle Eastern', 'East Asian', 'South Asian']

cohort_profiles = sorted(list(product(
    age_cohort, gender_cohort, education_cohort, income_cohort,
    political_cohort, religious_cohort, geography_cohort
)))

assert len(cohort_profiles) == N_COHORTS, f"Expected {N_COHORTS} cohorts, got {len(cohort_profiles)}"

# Build cohort_m -> demographic values mapping
cohort_demographics = {}
for m, profile in enumerate(cohort_profiles):
    age, gender, edu, inc, pol, rel, geo = profile
    cohort_demographics[m] = {
        'age': 'young' if age == 'young' else 'elderly',
        'gender': 'man' if gender == 'man' else 'woman',
        'education': 'high' if 'high' in edu else 'low',
        'income': 'high' if 'high' in inc else 'low',
        'politics': 'conservative' if 'conservative' in pol else 'progressive',
        'religion': 'religious' if 'deeply' in rel else 'non-religious',
        'geography': geo,
    }

print(f"  {N_COHORTS} cohort profiles reconstructed")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Define AMCE contrast functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_amce_dimensions(tags_first, tags_second):
    """
    For a canonical pair, determine which AMCE dimensions are testable
    and which character is the "positive" side for each dimension.

    Returns: list of (dimension_name, positive_is_first: bool)
    """
    dims = []

    # Age: young > old
    if ('young' in tags_first) != ('young' in tags_second):
        dims.append(('Age', 'young' in tags_first))
    if ('old' in tags_first) != ('old' in tags_second):
        dims.append(('Age_old', 'old' in tags_first))

    # Species: human > non-human
    if ('human' in tags_first) != ('human' in tags_second):
        dims.append(('Species', 'human' in tags_first))

    # Gender: female > male
    if ('female' in tags_first) != ('female' in tags_second):
        dims.append(('Gender', 'female' in tags_first))

    # Status: high > low
    if ('high-status' in tags_first) != ('high-status' in tags_second):
        dims.append(('Status', 'high-status' in tags_first))

    # Law: legal > jaywalking
    if ('legal' in tags_first) != ('legal' in tags_second):
        dims.append(('Law_legal', 'legal' in tags_first))
    if ('jaywalking' in tags_first) != ('jaywalking' in tags_second):
        dims.append(('Law_jaywalking', 'jaywalking' not in tags_first))

    # Number: several > one
    if ('several' in tags_first) != ('several' in tags_second):
        dims.append(('Number', 'several' in tags_first))

    # Fragile
    if ('fragile' in tags_first) != ('fragile' in tags_second):
        dims.append(('Fragile', 'fragile' in tags_first))

    return dims

# Pre-compute AMCE dimensions for all canonical pairs
pair_amce = []
for i, (tags_a, tags_b) in enumerate(pair_tags):
    pair_amce.append(get_amce_dimensions(tags_a, tags_b))

# Count how many dilemmas test each dimension
dim_counts = defaultdict(int)
for dims in pair_amce:
    for dim_name, _ in dims:
        dim_counts[dim_name] += 1

print("\nAMCE dimensions coverage:")
for dim, count in sorted(dim_counts.items()):
    print(f"  {dim}: {count} dilemmas test this")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Read parquet and compute per-cohort AMCE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nStep 4: Reading parquet and computing per-cohort AMCE...")
print(f"  Parquet: {PARQUET_PATH}")

if not os.path.exists(PARQUET_PATH):
    print(f"  ERROR: Parquet file not found at {PARQUET_PATH}")
    print(f"  To generate it, run: python scripts/STEP_1.py")
    print(f"  (requires raw batch files in data/batch_inputs/ and data/batch_outputs/)")
    sys.exit(1)

import polars as pl

start_time = time.time()

# Read only the columns we need (memory-efficient: ~2 GB for 61.7M rows)
print("  Reading parquet columns: dilemma_id, cohort_id, canonical_decision...")
df = pl.read_parquet(
    PARQUET_PATH,
    columns=["dilemma_id", "cohort_id", "canonical_decision"]
)
print(f"  Loaded {len(df):,} rows in {time.time() - start_time:.1f}s")

# Filter valid decisions (A or B; exclude None/unparseable)
df = df.filter(pl.col("canonical_decision").is_in(["A", "B"]))
n_valid = len(df)
print(f"  Valid decisions: {n_valid:,}")

# In canonical ordering:
#   canonical_decision == "A" → model chose to kill canonical_choice_0 → spared canonical_choice_1
#   canonical_decision == "B" → model chose to kill canonical_choice_1 → spared canonical_choice_0
# So: spared_is_first (spared canonical[0]) iff canonical_decision == "B"

# Step 4a: Aggregate per (dilemma_id, cohort_id) to reduce from 61.7M to ~3M rows
print("  Aggregating per (dilemma_id, cohort_id)...")
agg = df.group_by(["dilemma_id", "cohort_id"]).agg([
    (pl.col("canonical_decision") == "B").sum().alias("count_spared_first"),
    pl.col("canonical_decision").count().alias("total_votes"),
])
print(f"  Aggregated to {len(agg):,} (dilemma, cohort) cells")

del df  # free memory

# Step 4b: Build AMCE dimension mapping
# For each (dilemma_id, amce_dim): whether positive_is_first
print("  Building AMCE dimension mapping...")
amce_records = []
for n, dims in enumerate(pair_amce):
    for dim_name, pos_first in dims:
        amce_records.append({
            "dilemma_id": n,
            "amce_dim": dim_name,
            "positive_is_first": pos_first,
        })

df_amce = pl.DataFrame(amce_records)
print(f"  AMCE mapping: {len(df_amce):,} (dilemma, dimension) entries")

# Step 4c: Join aggregated data with AMCE dimensions
print("  Joining with AMCE dimensions...")
joined = agg.join(df_amce, on="dilemma_id")
print(f"  Joined: {len(joined):,} rows")

del agg  # free memory

# Step 4d: Compute spared_positive per row
# If positive_is_first: spared_positive = count_spared_first
# If NOT positive_is_first: spared_positive = total_votes - count_spared_first
joined = joined.with_columns(
    pl.when(pl.col("positive_is_first"))
    .then(pl.col("count_spared_first"))
    .otherwise(pl.col("total_votes") - pl.col("count_spared_first"))
    .alias("spared_positive")
)

# Step 4e: Aggregate per (cohort_id, amce_dim)
print("  Final aggregation per (cohort_id, amce_dim)...")
result = joined.group_by(["cohort_id", "amce_dim"]).agg([
    pl.col("spared_positive").sum().alias("spared_positive"),
    pl.col("total_votes").sum().alias("total"),
]).sort(["cohort_id", "amce_dim"])

print(f"  Result: {len(result):,} cohort-dimension records")

elapsed = time.time() - start_time
print(f"  Parquet processing complete in {elapsed:.1f}s")

del joined  # free memory


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Write per-cohort data to CSV
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nStep 5: Writing per-cohort data to {OUTPUT_CSV}")

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['cohort_m', 'age', 'gender', 'education', 'income', 'politics',
                      'religion', 'geography', 'amce_dimension', 'spared_positive', 'total'])

    for row in result.iter_rows(named=True):
        m = row['cohort_id']
        demo = cohort_demographics[m]
        writer.writerow([
            m,
            demo['age'],
            demo['gender'],
            demo['education'],
            demo['income'],
            demo['politics'],
            demo['religion'],
            demo['geography'],
            row['amce_dim'],
            row['spared_positive'],
            row['total'],
        ])

print(f"  Per-cohort data written ({len(result)} records)")


# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nDone! per_cohort_amce.csv regenerated. "
      f"Total time: {(time.time() - start_time)/60:.1f} minutes")
