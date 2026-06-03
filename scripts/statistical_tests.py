"""
Steps 2-12 of the analysis pipeline: statistical tests.

Takes the raw annotation data (DATASET.csv) and the BERTScore output from
1_compute_bertscore.py (bertscore_results.csv), merges them into one long-form
table (one row per article per model), and runs the full set of tests behind the
poster:

    Step 2  - merge into long form (180 rows = 60 articles x 3 models)
    Step 3  - complete-omission summary
    Step 4  - descriptive statistics (per model, per topic/model)
    Step 5  - one-sample t-tests of compression gap vs 0
    Step 6  - one-way ANOVA across the four topics
    Step 7  - paired t-tests between models
    Step 8  - directional consistency across models
    Step 9  - correlation between word-count gap and BERTScore gap
    Step 10 - balanced-subset robustness check (the honesty analysis)
    Step 11 - outlier identification (>2 SD from model mean)
    Step 12 - save all result tables to CSV

Primary metric throughout is word-count compression gap (gap_sideA). BERTScore
gap is carried as a secondary check; its weak correlation with the word-count
gap (Step 9) is why it is demoted to a limitation rather than reported as a
result.

Note: the sign test (39/38/24 articles compressing the conservative side) and the
omission chi-square reported on the poster are computed in the charting code; the
directional evidence here is the one-sample t-tests (Step 5) and the directional
consistency tallies (Step 8).

Input:  DATASET.csv, bertscore_results.csv
Output: merged_results.csv, summary_by_model.csv, summary_by_topic_model.csv,
        statistical_tests.csv, directional_consistency.csv,
        balanced_subset_results.csv, outliers.csv

Usage:
    python 2_statistical_tests.py                 # uses ./data
    DATA_DIR=/path/to/data python 2_statistical_tests.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, ttest_1samp, ttest_rel

warnings.filterwarnings("ignore")

# Directory holding the input CSVs and receiving the output CSVs.
# Override with the DATA_DIR environment variable.
OUTPUT_DIR = os.environ.get("DATA_DIR", "data")

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

ds = pd.read_csv(os.path.join(OUTPUT_DIR, "DATASET.csv"))
bs_df = pd.read_csv(os.path.join(OUTPUT_DIR, "bertscore_results.csv"))

# Strip whitespace from all column names
ds.columns = ds.columns.str.strip()
bs_df.columns = bs_df.columns.str.strip()

# Normalise article ID column name
ds = ds.rename(columns={"Article ID": "article_id"})

print(f"DATASET.csv: {ds.shape[0]} rows, {ds.shape[1]} cols")
print(f"bertscore_results.csv: {bs_df.shape[0]} rows, {bs_df.shape[1]} cols")

# Clean DATASET column names for easier access.
# ('groc'/'Groc'/'Grock' are the raw-data spellings of Grok.)
ds = ds.rename(columns={
    "Article: Side A": "orig_sideA_wc",
    "Article: Side B": "orig_sideB_wc",
    "Side A Article Share": "orig_sideA_share",
    "Side B Article Share": "orig_sideB_share",
    "CLAUDE Side A": "claude_sideA_wc",
    "CLAUDE Side B": "claude_sideB_wc",
    "CLAUDE Side A Share": "claude_sideA_share",
    "CLAUDE Side B share": "claude_sideB_share",
    "Chat Side A": "chat_sideA_wc",
    "Chat Side B": "chat_sideB_wc",
    "Chat Side A Share": "chat_sideA_share",
    "Chat Side B Share:": "chat_sideB_share",
    "Groc Side A:": "grok_sideA_wc",
    "Groc Side B:": "grok_sideB_wc",
    "Groc Side A Share:": "grok_sideA_share",
    "Groc Side B Share:": "grok_sideB_share",
    "Claude Gap Side A": "claude_gap_sideA",
    "Claude Gap Side B": "claude_gap_sideB",
    "Chat Gap Side A": "chat_gap_sideA",
    "Chat Gap Side B": "chat_gap_sideB",
    "Groc Gap Side A": "grok_gap_sideA",
    "Grock Gap Side B": "grok_gap_sideB",
    "Imbalance Direction": "imbalance_direction",
})

# Normalise imbalance direction values
ds["imbalance_direction"] = ds["imbalance_direction"].str.strip()

# Force gap columns numeric
gap_cols = ["claude_gap_sideA", "claude_gap_sideB",
            "chat_gap_sideA", "chat_gap_sideB",
            "grok_gap_sideA", "grok_gap_sideB"]
for c in gap_cols:
    ds[c] = pd.to_numeric(ds[c], errors="coerce")


# ─────────────────────────────────────────────
# STEP 2 — MERGE
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 2 — MERGE")
print("=" * 70)

# Long-form dataset: one row per article per model
# Pull gap columns from DATASET per model
def get_model_gaps(row, model_key):
    return {
        "gap_sideA": row.get(f"{model_key}_gap_sideA", np.nan),
        "gap_sideB": row.get(f"{model_key}_gap_sideB", np.nan),
        "sideA_wc":  row.get(f"{model_key}_sideA_wc", np.nan),
        "sideB_wc":  row.get(f"{model_key}_sideB_wc", np.nan),
    }

model_map = {"claude": "claude", "chat": "chat", "groc": "grok"}

long_rows = []
for _, drow in ds.iterrows():
    for tx_model, ds_key in model_map.items():
        gaps = get_model_gaps(drow, ds_key)
        long_rows.append({
            "article_id": drow["article_id"],
            "article_name": drow["Article Name"],
            "topic": drow["Topic"],
            "source_name": drow["Source Name"],
            "model": tx_model,
            "model_display": ds_key if ds_key != "grok" else "grok",
            "orig_sideA_wc": drow["orig_sideA_wc"],
            "orig_sideB_wc": drow["orig_sideB_wc"],
            "orig_sideA_share": drow["orig_sideA_share"],
            "orig_sideB_share": drow["orig_sideB_share"],
            "imbalance_direction": drow["imbalance_direction"],
            "gap_sideA": gaps["gap_sideA"],
            "gap_sideB": gaps["gap_sideB"],
            "model_sideA_wc": gaps["sideA_wc"],
            "model_sideB_wc": gaps["sideB_wc"],
        })

long_ds = pd.DataFrame(long_rows)

# Merge with BERTScore results
merged = long_ds.merge(
    bs_df[["article_id", "model", "bertscore_sideA", "bertscore_sideB",
           "bertscore_gap", "complete_omission_sideA", "complete_omission_sideB"]],
    on=["article_id", "model"],
    how="left"
)

assert len(merged) == 180, f"Expected 180 rows, got {len(merged)}"
merged.to_csv(os.path.join(OUTPUT_DIR, "merged_results.csv"), index=False)
print(f"Saved merged_results.csv: {merged.shape}")

print("\n--- Step 2 Summary ---")
print(f"Merged dataframe: {merged.shape[0]} rows × {merged.shape[1]} cols")
print(f"Models: {sorted(merged['model'].unique())}")
print(f"Topics: {sorted(merged['topic'].unique())}")
print(f"Missing gap_sideA: {merged['gap_sideA'].isna().sum()}")
print(f"Missing bertscore_gap: {merged['bertscore_gap'].isna().sum()}")


# ─────────────────────────────────────────────
# STEP 3 — OMISSION SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 3 — OMISSION SUMMARY")
print("=" * 70)

omission_rows = []
for side in ["sideA", "sideB"]:
    col = f"complete_omission_{side}"
    for m in ["claude", "chat", "groc"]:
        sub = merged[merged["model"] == m]
        count = sub[col].sum()
        rate = count / 60
        omission_rows.append({
            "model": m,
            "side": side,
            "omission_count": int(count),
            "omission_rate": round(rate, 4),
        })

omission_summary = pd.DataFrame(omission_rows)
print("\nOmission counts per model per side:")
print(omission_summary.to_string(index=False))

print("\nArticles with complete omissions:")
for _, row in merged[merged["complete_omission_sideA"] | merged["complete_omission_sideB"]].iterrows():
    sides = []
    if row["complete_omission_sideA"]:
        sides.append("Side A")
    if row["complete_omission_sideB"]:
        sides.append("Side B")
    print(f"  article_id={row['article_id']:3d}  model={row['model']:6s}  "
          f"topic={row['topic']:12s}  omitted={','.join(sides)}  "
          f"'{row['article_name'][:55]}'")


# ─────────────────────────────────────────────
# STEP 4 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 4 — DESCRIPTIVE STATISTICS")
print("=" * 70)

def desc_stats(series):
    return {"mean": series.mean(), "sd": series.std(), "n": series.count()}

metrics = ["gap_sideA", "gap_sideB", "bertscore_gap"]
model_stats_rows = []
for m in ["claude", "chat", "groc"]:
    sub = merged[merged["model"] == m]
    row = {"model": m}
    for met in metrics:
        d = desc_stats(sub[met].dropna())
        row[f"{met}_mean"] = round(d["mean"], 4)
        row[f"{met}_sd"]   = round(d["sd"], 4)
        row[f"{met}_n"]    = int(d["n"])
    model_stats_rows.append(row)

model_stats = pd.DataFrame(model_stats_rows)
print("\nPer-model statistics:")
print(model_stats.to_string(index=False))

topic_model_rows = []
for m in ["claude", "chat", "groc"]:
    for t in sorted(merged["topic"].unique()):
        sub = merged[(merged["model"] == m) & (merged["topic"] == t)]
        row = {"model": m, "topic": t}
        for met in metrics:
            d = desc_stats(sub[met].dropna())
            row[f"{met}_mean"] = round(d["mean"], 4)
            row[f"{met}_sd"]   = round(d["sd"], 4)
            row[f"{met}_n"]    = int(d["n"])
            if d["n"] < 15:
                row[f"{met}_flag"] = "UNDERPOWERED (n<15)"
            else:
                row[f"{met}_flag"] = ""
        topic_model_rows.append(row)

topic_model_stats = pd.DataFrame(topic_model_rows)
print("\nPer-topic-model statistics (n<15 flagged):")
print(topic_model_stats.to_string(index=False))

print("\n--- Step 4 Summary ---")
for m in ["claude", "chat", "groc"]:
    r = model_stats[model_stats["model"] == m].iloc[0]
    print(f"  {m}: gap_sideA mean={r['gap_sideA_mean']:.4f} (SD={r['gap_sideA_sd']:.4f})  "
          f"gap_sideB mean={r['gap_sideB_mean']:.4f}  "
          f"bertscore_gap mean={r['bertscore_gap_mean']:.4f}")
underpowered = topic_model_stats[
    topic_model_stats[[c for c in topic_model_stats.columns if c.endswith("_flag")]].apply(
        lambda row: row.str.contains("UNDER").any(), axis=1
    )
]
if len(underpowered):
    print(f"\n  Underpowered cells (n<15): {len(underpowered)}")
    for _, row in underpowered.iterrows():
        print(f"    model={row['model']}  topic={row['topic']}  n={row['gap_sideA_n']}")


# ─────────────────────────────────────────────
# STEP 5 — ONE-SAMPLE T-TESTS
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 5 — ONE-SAMPLE T-TESTS (vs 0)")
print("=" * 70)

def one_sample_ttest(data, label):
    data = data.dropna()
    if len(data) < 2:
        return {"label": label, "n": len(data), "t": np.nan, "p": np.nan, "sig": "N/A"}
    t, p = ttest_1samp(data, 0)
    return {"label": label, "n": len(data), "t": round(t, 4), "p": round(p, 4),
            "sig": "YES" if p < 0.05 else "no"}

ttest_rows = []
for m in ["claude", "chat", "groc"]:
    sub = merged[merged["model"] == m]
    for met, col_A, col_B in [
        ("gap_sideA", "gap_sideA", "model_sideA_wc"),
        ("gap_sideB", "gap_sideB", "model_sideB_wc"),
        ("bertscore_gap", "bertscore_gap", None),
    ]:
        # Including all rows
        r_all = one_sample_ttest(sub[met], f"{m} {met} all")
        r_all["model"] = m
        r_all["metric"] = met
        r_all["version"] = "all"
        ttest_rows.append(r_all)

        # Excluding zero word count rows (only for gap_sideA / gap_sideB)
        if col_B is not None:
            nonzero = sub[sub[col_B] > 0][met]
        else:
            # For bertscore_gap, exclude omissions
            if met == "bertscore_gap":
                nonzero = sub[~sub["complete_omission_sideA"] & ~sub["complete_omission_sideB"]][met]
            else:
                nonzero = sub[met]
        r_nz = one_sample_ttest(nonzero, f"{m} {met} nonzero")
        r_nz["model"] = m
        r_nz["metric"] = met
        r_nz["version"] = "nonzero"
        ttest_rows.append(r_nz)

ttest_df = pd.DataFrame(ttest_rows)[["model", "metric", "version", "n", "t", "p", "sig"]]
print(ttest_df.to_string(index=False))

print("\n--- Step 5 Summary ---")
sig_all = ttest_df[ttest_df["sig"] == "YES"]
print(f"Significant results (p<0.05): {len(sig_all)} of {len(ttest_df)}")
for _, r in sig_all.iterrows():
    print(f"  {r['model']:6s} {r['metric']:20s} [{r['version']:7s}]  t={r['t']:.3f}  p={r['p']:.4f}")

# Check if results change between versions
for m in ["claude", "chat", "groc"]:
    for met in ["gap_sideA", "gap_sideB", "bertscore_gap"]:
        sub = ttest_df[(ttest_df["model"] == m) & (ttest_df["metric"] == met)]
        if len(sub) == 2:
            sigs = sub["sig"].tolist()
            if sigs[0] != sigs[1]:
                print(f"  ** Result CHANGES between versions: {m} {met}: {sigs}")


# ─────────────────────────────────────────────
# STEP 6 — ANOVA ACROSS TOPICS
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 6 — ONE-WAY ANOVA ACROSS TOPICS")
print("=" * 70)

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not available — Tukey HSD skipped.")

anova_rows = []
topics = sorted(merged["topic"].unique())

for m in ["claude", "chat", "groc"]:
    for met in ["gap_sideA", "bertscore_gap"]:
        sub = merged[merged["model"] == m]
        groups = [sub[sub["topic"] == t][met].dropna().values for t in topics]
        ns = [len(g) for g in groups]
        if all(len(g) >= 2 for g in groups):
            F, p = f_oneway(*groups)
        else:
            F, p = np.nan, np.nan

        tukey_result = ""
        if p < 0.05 and HAS_STATSMODELS:
            valid = sub[[met, "topic"]].dropna()
            try:
                tk = pairwise_tukeyhsd(valid[met], valid["topic"], alpha=0.05)
                tukey_result = str(tk.summary())
            except Exception as e:
                tukey_result = f"Tukey failed: {e}"

        row = {"model": m, "metric": met,
               "F": round(F, 4) if not np.isnan(F) else np.nan,
               "p": round(p, 4) if not np.isnan(p) else np.nan,
               "sig": "YES" if (not np.isnan(p) and p < 0.05) else "no"}
        for t, n in zip(topics, ns):
            row[f"n_{t}"] = n
            row[f"mean_{t}"] = round(sub[sub["topic"] == t][met].mean(), 4)
        anova_rows.append(row)

        print(f"\n{m.upper()} | {met} | F={row['F']} p={row['p']} {'***SIGNIFICANT***' if row['sig']=='YES' else ''}")
        for t in topics:
            n_cell = row.get(f"n_{t}", 0)
            mean_cell = row.get(f"mean_{t}", np.nan)
            direction = "Side B compressed more" if mean_cell > 0 else ("Side A compressed more" if mean_cell < 0 else "balanced")
            flag = " [n<15 FLAG]" if n_cell < 15 else ""
            print(f"  {t:15s} n={n_cell:2d}  mean={mean_cell:7.4f}  ({direction}){flag}")
        if tukey_result:
            print(f"  Tukey HSD:\n{tukey_result}")

anova_df = pd.DataFrame(anova_rows)

print("\n--- Step 6 Summary ---")
sig_anova = anova_df[anova_df["sig"] == "YES"]
print(f"Significant ANOVAs: {len(sig_anova)} of {len(anova_df)}")
for _, r in sig_anova.iterrows():
    print(f"  {r['model']} {r['metric']}: F={r['F']} p={r['p']}")


# ─────────────────────────────────────────────
# STEP 7 — PAIRED T-TESTS BETWEEN MODELS
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 7 — PAIRED T-TESTS BETWEEN MODELS")
print("=" * 70)

model_pairs = [("claude", "chat"), ("claude", "groc"), ("chat", "groc")]

paired_rows = []
for met in ["gap_sideA", "bertscore_gap"]:
    for m1, m2 in model_pairs:
        sub1 = merged[merged["model"] == m1].set_index("article_id")[met]
        sub2 = merged[merged["model"] == m2].set_index("article_id")[met]
        common = sub1.index.intersection(sub2.index)
        d1 = sub1.loc[common].dropna()
        d2 = sub2.loc[common].dropna()
        common2 = d1.index.intersection(d2.index)
        d1, d2 = d1.loc[common2], d2.loc[common2]

        if len(d1) >= 2:
            t, p = ttest_rel(d1, d2)
        else:
            t, p = np.nan, np.nan
        paired_rows.append({
            "metric": met, "model1": m1, "model2": m2, "version": "all",
            "n": len(d1), "t": round(t, 4), "p": round(p, 4),
            "sig": "YES" if (not np.isnan(p) and p < 0.05) else "no"
        })

        # Excluding zero/omission rows
        if met == "gap_sideA":
            nz1 = merged[(merged["model"] == m1) & (merged["model_sideA_wc"] > 0)].set_index("article_id")[met]
            nz2 = merged[(merged["model"] == m2) & (merged["model_sideA_wc"] > 0)].set_index("article_id")[met]
        else:
            nz1 = merged[(merged["model"] == m1) & ~merged["complete_omission_sideA"] & ~merged["complete_omission_sideB"]].set_index("article_id")[met]
            nz2 = merged[(merged["model"] == m2) & ~merged["complete_omission_sideA"] & ~merged["complete_omission_sideB"]].set_index("article_id")[met]
        c = nz1.index.intersection(nz2.index)
        nz1, nz2 = nz1.loc[c].dropna(), nz2.loc[c].dropna()
        c2 = nz1.index.intersection(nz2.index)
        nz1, nz2 = nz1.loc[c2], nz2.loc[c2]
        if len(nz1) >= 2:
            tnz, pnz = ttest_rel(nz1, nz2)
        else:
            tnz, pnz = np.nan, np.nan
        paired_rows.append({
            "metric": met, "model1": m1, "model2": m2, "version": "nonzero",
            "n": len(nz1), "t": round(tnz, 4), "p": round(pnz, 4),
            "sig": "YES" if (not np.isnan(pnz) and pnz < 0.05) else "no"
        })

paired_df = pd.DataFrame(paired_rows)
print(paired_df.to_string(index=False))

print("\n--- Step 7 Summary ---")
sig_paired = paired_df[paired_df["sig"] == "YES"]
print(f"Significant paired tests: {len(sig_paired)} of {len(paired_df)}")
for _, r in sig_paired.iterrows():
    print(f"  {r['model1']} vs {r['model2']}  {r['metric']:20s} [{r['version']}]  t={r['t']:.3f}  p={r['p']:.4f}")
for met in ["gap_sideA", "bertscore_gap"]:
    for m1, m2 in model_pairs:
        sub = paired_df[(paired_df["metric"] == met) & (paired_df["model1"] == m1) & (paired_df["model2"] == m2)]
        if len(sub) == 2:
            sigs = sub["sig"].tolist()
            if sigs[0] != sigs[1]:
                print(f"  ** CHANGES between versions: {m1} vs {m2} on {met}: {sigs}")


# ─────────────────────────────────────────────
# STEP 8 — DIRECTIONAL CONSISTENCY
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 8 — DIRECTIONAL CONSISTENCY ACROSS MODELS")
print("=" * 70)

pivot = merged.pivot(index="article_id", columns="model", values="gap_sideA")
pivot = pivot.dropna()

def dominant_side(row):
    """Which side got compressed more (negative gap_sideA = Side A compressed more)."""
    return "sideA" if row < 0 else ("sideB" if row > 0 else "balanced")

for m in ["claude", "chat", "groc"]:
    pivot[f"dir_{m}"] = pivot[m].apply(dominant_side)

pivot["all_agree"] = (
    (pivot["dir_claude"] == pivot["dir_chat"]) &
    (pivot["dir_chat"] == pivot["dir_groc"])
)
pivot["two_agree"] = (
    (pivot["dir_claude"] == pivot["dir_chat"]) |
    (pivot["dir_claude"] == pivot["dir_groc"]) |
    (pivot["dir_chat"] == pivot["dir_groc"])
)

n_articles = len(pivot)
all_agree_pct = pivot["all_agree"].mean() * 100
two_agree_pct = pivot["two_agree"].mean() * 100

print(f"\nAll three models agree on direction: {pivot['all_agree'].sum()}/{n_articles} = {all_agree_pct:.1f}%")
print(f"At least two models agree:          {pivot['two_agree'].sum()}/{n_articles} = {two_agree_pct:.1f}%")

# Merge topic info for by-topic breakdown
pivot_topic = pivot.merge(ds[["article_id", "Topic"]], on="article_id")
print("\nBy topic:")
dir_rows = []
for t in sorted(pivot_topic["Topic"].unique()):
    sub_t = pivot_topic[pivot_topic["Topic"] == t]
    n_t = len(sub_t)
    aa = sub_t["all_agree"].mean() * 100
    ta = sub_t["two_agree"].mean() * 100
    print(f"  {t:15s} n={n_t:2d}  all agree={aa:.1f}%  two agree={ta:.1f}%")
    dir_rows.append({
        "topic": t, "n": n_t,
        "all_agree_pct": round(aa, 1), "two_agree_pct": round(ta, 1)
    })

dir_df = pd.DataFrame(dir_rows)

print("\n--- Step 8 Summary ---")
print(f"Overall directional consistency: {all_agree_pct:.1f}% all-three-agree, "
      f"{two_agree_pct:.1f}% at-least-two-agree. "
      f"{'High consistency suggests structural rather than model-specific asymmetry.' if all_agree_pct > 50 else 'Low consistency suggests model-specific rather than structural asymmetry.'}")


# ─────────────────────────────────────────────
# STEP 9 — CORRELATION BETWEEN METRICS
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 9 — CORRELATION: COMPRESSION GAP vs BERTSCORE GAP")
print("=" * 70)

corr_rows = []

# Overall
valid = merged[["gap_sideA", "bertscore_gap"]].dropna()
r, p = stats.pearsonr(valid["gap_sideA"], valid["bertscore_gap"])
corr_rows.append({"subset": "all", "model": "all", "r": round(r, 4), "p": round(p, 4),
                  "n": len(valid), "sig": "YES" if p < 0.05 else "no"})
print(f"All rows: r={r:.4f}  p={p:.4f}  n={len(valid)}")

# Per model
for m in ["claude", "chat", "groc"]:
    sub = merged[merged["model"] == m][["gap_sideA", "bertscore_gap"]].dropna()
    r, p = stats.pearsonr(sub["gap_sideA"], sub["bertscore_gap"])
    corr_rows.append({"subset": "per_model", "model": m, "r": round(r, 4), "p": round(p, 4),
                      "n": len(sub), "sig": "YES" if p < 0.05 else "no"})
    print(f"  {m:6s}: r={r:.4f}  p={p:.4f}  n={len(sub)}")

corr_df = pd.DataFrame(corr_rows)

print("\n--- Step 9 Summary ---")
overall_r = corr_df[corr_df["subset"] == "all"]["r"].iloc[0]
print(f"Overall Pearson r between compression gap Side A and BERTScore gap = {overall_r:.4f}. "
      f"{'Strong positive correlation validates that both metrics capture the same phenomenon.' if abs(overall_r) > 0.5 else 'Moderate/weak correlation suggests the metrics capture partially different aspects of compression bias.'}")


# ─────────────────────────────────────────────
# STEP 10 — BALANCED SUBSET ANALYSIS
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 10 — BALANCED SUBSET ANALYSIS")
print("=" * 70)

bal = merged[merged["imbalance_direction"].str.lower() == "balanced"].copy()
print(f"Balanced subset: {len(bal)} rows ({len(bal)//3} articles, {len(bal['model'].unique())} models each)")

if len(bal) == 0:
    print("No balanced articles — skipping.")
else:
    # Step 5 rerun on balanced
    print("\n[Balanced] One-sample t-tests:")
    bal_ttest_rows = []
    for m in ["claude", "chat", "groc"]:
        sub = bal[bal["model"] == m]
        for met in ["gap_sideA", "gap_sideB", "bertscore_gap"]:
            r = one_sample_ttest(sub[met], f"{m} {met}")
            r["model"] = m
            r["metric"] = met
            bal_ttest_rows.append(r)
    bal_ttest = pd.DataFrame(bal_ttest_rows)
    print(bal_ttest[["model", "metric", "n", "t", "p", "sig"]].to_string(index=False))

    print("\n[Balanced] ANOVA across topics:")
    bal_anova_rows = []
    bal_topics = sorted(bal["topic"].unique())
    for m in ["claude", "chat", "groc"]:
        for met in ["gap_sideA", "bertscore_gap"]:
            sub = bal[bal["model"] == m]
            groups = [sub[sub["topic"] == t][met].dropna().values for t in bal_topics]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) >= 2:
                F, p = f_oneway(*groups)
                sig = "YES" if p < 0.05 else "no"
            else:
                F, p, sig = np.nan, np.nan, "N/A"
            bal_anova_rows.append({"model": m, "metric": met,
                                    "F": round(F, 4) if not np.isnan(F) else np.nan,
                                    "p": round(p, 4) if not np.isnan(p) else np.nan, "sig": sig})
    bal_anova = pd.DataFrame(bal_anova_rows)
    print(bal_anova.to_string(index=False))

    print("\n[Balanced] Paired t-tests between models:")
    bal_paired_rows = []
    for met in ["gap_sideA", "bertscore_gap"]:
        for m1, m2 in model_pairs:
            s1 = bal[bal["model"] == m1].set_index("article_id")[met]
            s2 = bal[bal["model"] == m2].set_index("article_id")[met]
            c = s1.index.intersection(s2.index)
            d1, d2 = s1.loc[c].dropna(), s2.loc[c].dropna()
            c2 = d1.index.intersection(d2.index)
            d1, d2 = d1.loc[c2], d2.loc[c2]
            if len(d1) >= 2:
                t, p = ttest_rel(d1, d2)
                sig = "YES" if p < 0.05 else "no"
            else:
                t, p, sig = np.nan, np.nan, "N/A"
            bal_paired_rows.append({
                "metric": met, "model1": m1, "model2": m2,
                "n": len(d1), "t": round(t, 4), "p": round(p, 4), "sig": sig
            })
    bal_paired = pd.DataFrame(bal_paired_rows)
    print(bal_paired.to_string(index=False))

    print("\n[Balanced] Directional consistency:")
    bal_pivot = bal.pivot(index="article_id", columns="model", values="gap_sideA").dropna()
    for m in ["claude", "chat", "groc"]:
        if m in bal_pivot.columns:
            bal_pivot[f"dir_{m}"] = bal_pivot[m].apply(dominant_side)
    if all(f"dir_{m}" in bal_pivot.columns for m in ["claude", "chat", "groc"]):
        bal_pivot["all_agree"] = (
            (bal_pivot["dir_claude"] == bal_pivot["dir_chat"]) &
            (bal_pivot["dir_chat"] == bal_pivot["dir_groc"])
        )
        bal_pivot["two_agree"] = (
            (bal_pivot["dir_claude"] == bal_pivot["dir_chat"]) |
            (bal_pivot["dir_claude"] == bal_pivot["dir_groc"]) |
            (bal_pivot["dir_chat"] == bal_pivot["dir_groc"])
        )
        nb = len(bal_pivot)
        print(f"  All agree: {bal_pivot['all_agree'].sum()}/{nb} = {bal_pivot['all_agree'].mean()*100:.1f}%")
        print(f"  Two agree: {bal_pivot['two_agree'].sum()}/{nb} = {bal_pivot['two_agree'].mean()*100:.1f}%")

    print("\n--- Step 10 Summary ---")
    full_sig_A = set(ttest_df[(ttest_df["metric"] == "gap_sideA") & (ttest_df["sig"] == "YES") & (ttest_df["version"] == "all")]["model"].tolist())
    bal_sig_A  = set(bal_ttest[(bal_ttest["metric"] == "gap_sideA") & (bal_ttest["sig"] == "YES")]["model"].tolist()) if len(bal_ttest) > 0 else set()
    print(f"  Full corpus sig (gap_sideA t-test): {full_sig_A}")
    print(f"  Balanced subset sig (gap_sideA t-test): {bal_sig_A}")
    disappeared = full_sig_A - bal_sig_A
    if disappeared:
        print(f"  Effects that disappear in balanced subset: {disappeared} → may reflect faithful compression of already-skewed articles")
    remained = full_sig_A & bal_sig_A
    if remained:
        print(f"  Effects that HOLD in balanced subset: {remained} → suggests model-introduced bias even in balanced articles")

    bal_results = {
        "ttest": bal_ttest if len(bal_ttest) > 0 else pd.DataFrame(),
        "anova": bal_anova,
        "paired": bal_paired,
    }


# ─────────────────────────────────────────────
# STEP 11 — OUTLIER IDENTIFICATION
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 11 — OUTLIER IDENTIFICATION (>2 SD from model mean)")
print("=" * 70)

outlier_rows = []
for m in ["claude", "chat", "groc"]:
    sub = merged[merged["model"] == m].copy()
    mu = sub["gap_sideA"].mean()
    sd = sub["gap_sideA"].std()
    threshold = 2 * sd
    outliers = sub[abs(sub["gap_sideA"] - mu) > threshold]
    for _, row in outliers.iterrows():
        outlier_rows.append({
            "article_id": row["article_id"],
            "article_name": row["article_name"],
            "topic": row["topic"],
            "model": m,
            "gap_sideA": round(row["gap_sideA"], 4),
            "model_mean": round(mu, 4),
            "sds_from_mean": round((row["gap_sideA"] - mu) / sd, 2),
            "imbalance_direction": row["imbalance_direction"],
        })

outlier_df = pd.DataFrame(outlier_rows)
if len(outlier_df):
    print(outlier_df.to_string(index=False))
else:
    print("No outliers found.")

print(f"\n--- Step 11 Summary ---")
print(f"Total outliers (>2 SD on gap_sideA): {len(outlier_df)}")
for m in ["claude", "chat", "groc"]:
    n = len(outlier_df[outlier_df["model"] == m]) if len(outlier_df) else 0
    print(f"  {m}: {n} outlier(s)")


# ─────────────────────────────────────────────
# STEP 12 — SAVE ALL RESULTS TABLES
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 12 — SAVING RESULT TABLES")
print("=" * 70)

model_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_by_model.csv"), index=False)
topic_model_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_by_topic_model.csv"), index=False)

stat_tables = [ttest_df, anova_df, paired_df, corr_df]
stat_combined = pd.concat(stat_tables, ignore_index=True, sort=False)
stat_combined.to_csv(os.path.join(OUTPUT_DIR, "statistical_tests.csv"), index=False)

dir_summary = pd.DataFrame([{
    "scope": "overall",
    "topic": "all",
    "n_articles": n_articles,
    "all_agree_count": int(pivot["all_agree"].sum()),
    "all_agree_pct": round(all_agree_pct, 1),
    "two_agree_count": int(pivot["two_agree"].sum()),
    "two_agree_pct": round(two_agree_pct, 1),
}] + [{
    "scope": "by_topic",
    "topic": r["topic"],
    "n_articles": r["n"],
    "all_agree_count": None,
    "all_agree_pct": r["all_agree_pct"],
    "two_agree_count": None,
    "two_agree_pct": r["two_agree_pct"],
} for _, r in dir_df.iterrows()])
dir_summary.to_csv(os.path.join(OUTPUT_DIR, "directional_consistency.csv"), index=False)

if len(bal) > 0:
    bal_out = pd.concat([bal_ttest, bal_anova, bal_paired], ignore_index=True, sort=False)
    bal_out.to_csv(os.path.join(OUTPUT_DIR, "balanced_subset_results.csv"), index=False)

if len(outlier_df):
    outlier_df.to_csv(os.path.join(OUTPUT_DIR, "outliers.csv"), index=False)

print("Saved files:")
saved = ["bertscore_results.csv", "merged_results.csv", "summary_by_model.csv",
         "summary_by_topic_model.csv", "statistical_tests.csv",
         "directional_consistency.csv", "balanced_subset_results.csv", "outliers.csv"]
for f in saved:
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(path):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} NOT FOUND")


# ─────────────────────────────────────────────
# OVERALL SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)

print(f"""
This analysis examined whether three AI summarization models (Claude, ChatGPT/Chat,
and Grok) introduced systematic compression bias when summarizing {ds.shape[0]} politically
balanced news articles across multiple topics.

KEY FINDINGS:

1. OMISSIONS: {merged['complete_omission_sideA'].sum()} complete Side A omissions and
   {merged['complete_omission_sideB'].sum()} complete Side B omissions were detected across all
   models. These represent cases where a model produced zero sentences for one side.

2. DESCRIPTIVE PATTERNS:
   - Claude:  gap_sideA mean = {model_stats[model_stats['model']=='claude']['gap_sideA_mean'].iloc[0]:.4f}  (negative = Side A under-represented vs original)
   - Chat:    gap_sideA mean = {model_stats[model_stats['model']=='chat']['gap_sideA_mean'].iloc[0]:.4f}
   - Grok:    gap_sideA mean = {model_stats[model_stats['model']=='groc']['gap_sideA_mean'].iloc[0]:.4f}

3. STATISTICAL TESTS (Step 5): {len(ttest_df[ttest_df['sig']=='YES'])} of {len(ttest_df)} one-sample
   t-tests were significant at p<0.05, suggesting that compression gaps differ
   systematically from zero for some model/metric combinations.

4. TOPIC EFFECTS (Step 6): {len(anova_df[anova_df['sig']=='YES'])} of {len(anova_df)} ANOVAs were
   significant, indicating that compression gap varies across topics for at least
   one model/metric combination.

5. BETWEEN-MODEL DIFFERENCES (Step 7): {len(paired_df[paired_df['sig']=='YES'])} of {len(paired_df)}
   paired comparisons reached significance, suggesting models differ in how much
   they compress each side.

6. DIRECTIONAL CONSISTENCY (Step 8): {all_agree_pct:.1f}% of articles saw all three
   models compress the same side; {two_agree_pct:.1f}% saw at least two models agree.
   {'High consistency points to structural features of articles (e.g., original imbalance) driving compression patterns rather than model-specific behavior.' if all_agree_pct > 55 else 'Moderate consistency suggests both structural and model-specific factors are at play.'}

7. METRIC CORRELATION (Step 9): Pearson r = {corr_df[corr_df['subset']=='all']['r'].iloc[0]:.4f} between
   word-count compression gap and BERTScore gap across all 180 rows.
   {'Strong correlation validates both metrics are capturing the same compression phenomenon.' if abs(corr_df[corr_df['subset']=='all']['r'].iloc[0]) > 0.5 else 'Moderate/weak correlation suggests the metrics are partly capturing different dimensions of compression.'}

8. BALANCED SUBSET (Step 10): The subset of articles rated as originally balanced
   ({len(bal)//3 if len(bal)>0 else 0} articles) provides the key robustness check. Effects that survive
   in this subset cannot be attributed purely to faithful compression of pre-existing
   asymmetry and instead implicate model-introduced bias.

9. OUTLIERS (Step 11): {len(outlier_df)} articles exceeded 2 SD on compression gap Side A
   and are candidates for qualitative review.
""")
