"""
Step 1 of the analysis pipeline: semantic similarity scoring.

Computes BERTScore F1 between each article's original per-side text and the
corresponding AI summary text, for all three models (Claude, ChatGPT, Grok).
Also flags complete omissions (a side that received zero words in the summary).

Note on interpretation: BERTScore is treated as a SECONDARY metric only. Because
summaries are ~5x shorter than the originals, a short on-topic summary scores
high regardless of whether one side's argument was gutted. The primary metric in
this study is word-count compression (see 2_statistical_tests.py). BERTScore is
reported as a robustness check / limitation, not as evidence of meaning loss.

If the bert-score library is unavailable, the script falls back to
sentence-transformers cosine similarity so the pipeline still runs.

Input:  test_export.csv   (per-side original + summary text, one row per article/model)
Output: bertscore_results.csv

Usage:
    python 1_compute_bertscore.py                 # uses ./data
    DATA_DIR=/path/to/data python 1_compute_bertscore.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Directory holding the input CSVs and receiving the output CSVs.
# Override with the DATA_DIR environment variable.
DATA_DIR = os.environ.get("DATA_DIR", "data")


def is_missing(text):
    """True if a cell is empty / NaN / a 'none' placeholder (i.e. a side was omitted)."""
    if text is None:
        return True
    if isinstance(text, float) and np.isnan(text):
        return True
    s = str(text).strip().lower()
    return s in ("none", "", "nan")


def main():
    tx = pd.read_csv(os.path.join(DATA_DIR, "test_export.csv"))
    tx.columns = tx.columns.str.strip()
    # 'groc' is the internal spelling of Grok in the raw data; keep a display column.
    tx["model_display"] = tx["model"].replace({"groc": "grok"})

    print(f"test_export.csv: {tx.shape[0]} rows, {tx.shape[1]} cols")
    print(f"Models: {sorted(tx['model'].unique())}")
    print(f"Topics: {sorted(tx['topic'].unique())}")

    similarity_method = "unknown"

    # ---- Try BERTScore first ------------------------------------------------
    try:
        from bert_score import score as bert_score_fn  # noqa: F401  (import side effect)

        def compute_bertscore(refs, hyps):
            _, _, f1 = bert_score_fn(
                hyps, refs, lang="en", verbose=False, rescale_with_baseline=False
            )
            return f1.tolist()

        sideA_scores, sideB_scores = [], []
        omit_A, omit_B = [], []
        valid_A_idx, valid_A_refs, valid_A_hyps = [], [], []
        valid_B_idx, valid_B_refs, valid_B_hyps = [], [], []

        for i, row in tx.iterrows():
            if is_missing(row["sideA_summary_text"]):
                omit_A.append(True)
                sideA_scores.append(np.nan)
            else:
                omit_A.append(False)
                valid_A_idx.append(i)
                valid_A_refs.append(str(row["sideA_original_text"]))
                valid_A_hyps.append(str(row["sideA_summary_text"]))
                sideA_scores.append(None)  # placeholder, filled below

            if is_missing(row["sideB_summary_text"]):
                omit_B.append(True)
                sideB_scores.append(np.nan)
            else:
                omit_B.append(False)
                valid_B_idx.append(i)
                valid_B_refs.append(str(row["sideB_original_text"]))
                valid_B_hyps.append(str(row["sideB_summary_text"]))
                sideB_scores.append(None)

        print(f"Computing BERTScore for {len(valid_A_idx)} Side A pairs ...")
        if valid_A_refs:
            fa = compute_bertscore(valid_A_refs, valid_A_hyps)
            for pos, idx in enumerate(valid_A_idx):
                sideA_scores[idx - tx.index[0]] = fa[pos]

        print(f"Computing BERTScore for {len(valid_B_idx)} Side B pairs ...")
        if valid_B_refs:
            fb = compute_bertscore(valid_B_refs, valid_B_hyps)
            for pos, idx in enumerate(valid_B_idx):
                sideB_scores[idx - tx.index[0]] = fb[pos]

        similarity_method = "BERTScore F1 (bert-score library, roberta-large)"
        print("BERTScore computation complete.")

    # ---- Fallback: sentence-transformers cosine similarity ------------------
    except Exception as e:
        print(f"BERTScore unavailable ({e}); falling back to cosine similarity.")
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model_st = SentenceTransformer("all-MiniLM-L6-v2")
        sideA_scores, sideB_scores = [], []
        omit_A, omit_B = [], []

        all_texts = []
        for _, row in tx.iterrows():
            for col in ["sideA_original_text", "sideB_original_text",
                        "sideA_summary_text", "sideB_summary_text"]:
                t = row[col]
                all_texts.append("" if is_missing(t) else str(t))

        embs = model_st.encode(all_texts, batch_size=64, show_progress_bar=True)

        for i, row in tx.iterrows():
            pos = i * 4
            miss_A = is_missing(row["sideA_summary_text"])
            miss_B = is_missing(row["sideB_summary_text"])
            omit_A.append(miss_A)
            omit_B.append(miss_B)
            if miss_A:
                sideA_scores.append(np.nan)
            else:
                e_orig = embs[pos].reshape(1, -1)
                e_sum = embs[pos + 2].reshape(1, -1)
                sideA_scores.append(float(cosine_similarity(e_orig, e_sum)[0][0]))
            if miss_B:
                sideB_scores.append(np.nan)
            else:
                e_orig = embs[pos + 1].reshape(1, -1)
                e_sum = embs[pos + 3].reshape(1, -1)
                sideB_scores.append(float(cosine_similarity(e_orig, e_sum)[0][0]))

        similarity_method = "sentence-transformers cosine similarity (all-MiniLM-L6-v2)"

    print(f"\nSimilarity method used: {similarity_method}")

    # ---- Assemble and save --------------------------------------------------
    bs_df = tx[["article_id", "topic", "model"]].copy()
    bs_df["model_display"] = bs_df["model"].replace({"groc": "grok"})
    bs_df["bertscore_sideA"] = sideA_scores
    bs_df["bertscore_sideB"] = sideB_scores
    bs_df["complete_omission_sideA"] = omit_A
    bs_df["complete_omission_sideB"] = omit_B
    bs_df["bertscore_gap"] = bs_df["bertscore_sideA"] - bs_df["bertscore_sideB"]

    out_path = os.path.join(DATA_DIR, "bertscore_results.csv")
    bs_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Total rows: {len(bs_df)}")
    print(f"Complete Side A omissions: {bs_df['complete_omission_sideA'].sum()}")
    print(f"Complete Side B omissions: {bs_df['complete_omission_sideB'].sum()}")


if __name__ == "__main__":
    main()
