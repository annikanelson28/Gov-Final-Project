"""
Valence beta regression — does summarizing shift a text's political valence?

VALENCE of any text = liberal words / (liberal + conservative words)
              = Side B words / (Side A + Side B words)        [Side B = liberal]
A perfectly balanced text scores 0.5; liberal-leaning -> 1; conservative -> 0.

For each model (Claude, ChatGPT, Grok) separately, we regress valence on a single
binary predictor:  is_summary  (0 = original article, 1 = the model's summary).
The coefficient on is_summary answers: does moving from article to summary change
valence, and in which direction?

Method: beta regression (valence is a proportion bounded on (0,1), so OLS is wrong).
Because complete omissions produce valence of exactly 0 or 1 — which beta regression
cannot accept — we apply the Smithson & Verkuilen (2006) "squeeze" transform:
        y' = (y * (n - 1) + 0.5) / n
which nudges every value just inside (0,1) and KEEPS the omission rows (those are
the most important data points, e.g. Claude dropping a side entirely).

Standard errors are robust (HC1).

Input:  input_dataset.csv   (one row per article, raw per-side word counts)
Output: valence_long.csv             (one row per text: original + each summary)
        valence_regression_results.csv  (per-model coefficient, SE, p-value)

Usage:
    python valence_beta_regression.py
    DATA_DIR=/path/to/folder python valence_beta_regression.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel

# Folder containing input_dataset.csv (and where outputs are written).
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/Users/carolinekrantz/Desktop/project_analysis/INPUT_CSV",
)
INPUT_CSV = os.path.join(DATA_DIR, "input_dataset.csv")


def squeeze(y):
    """Smithson-Verkuilen transform: pull proportions just inside (0, 1)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    return (y * (n - 1) + 0.5) / n


def valence(side_a, side_b):
    """Liberal share of partisan words = Side B / (Side A + Side B)."""
    total = side_a + side_b
    # If a text has zero partisan words on both sides, valence is undefined.
    return np.where(total > 0, side_b / total, np.nan)


def main():
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()  # column names have stray spaces

    # Raw per-side word-count columns (Side A = conservative, Side B = liberal).
    # Model summary columns use slightly different spellings in the sheet.
    orig_a, orig_b = "Article: Side A", "Article: Side B"
    model_cols = {
        "claude": ("CLAUDE Side A", "CLAUDE Side B"),
        "chat":   ("Chat Side A", "Chat Side B"),
        "groc":   ("Groc Side A:", "Groc Side B:"),
    }

    # Sanity-check that every expected column is present.
    needed = [orig_a, orig_b] + [c for pair in model_cols.values() for c in pair]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected columns: {missing}\nAvailable: {list(df.columns)}"
        )

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Build long table: one row per text ───────────────────────────────────
    # For each model we have: the original article (is_summary=0) and that
    # model's summary (is_summary=1), both with their own valence.
    rows = []
    for _, r in df.iterrows():
        aid, topic = r["Article ID"], r["Topic"]
        v_orig = valence(r[orig_a], r[orig_b])
        for model, (ca, cb) in model_cols.items():
            # original article (shared text, but labeled per model so each
            # model's regression has matched original/summary pairs)
            rows.append({"article_id": aid, "topic": topic, "model": model,
                         "is_summary": 0, "valence": float(v_orig)})
            # this model's summary
            v_sum = valence(r[ca], r[cb])
            rows.append({"article_id": aid, "topic": topic, "model": model,
                         "is_summary": 1, "valence": float(v_sum)})

    long = pd.DataFrame(rows)

    # Drop rows where valence is undefined (text had zero partisan words at all).
    n_before = len(long)
    long = long.dropna(subset=["valence"]).reset_index(drop=True)
    n_dropped = n_before - len(long)
    if n_dropped:
        print(f"Note: dropped {n_dropped} rows with zero partisan words "
              f"(valence undefined).")

    long.to_csv(os.path.join(DATA_DIR, "valence_long.csv"), index=False)
    print(f"Built valence_long.csv: {len(long)} rows "
          f"({long['model'].nunique()} models)\n")

    # ── Per-model beta regression: valence ~ is_summary ──────────────────────
    print("=" * 68)
    print("BETA REGRESSION  (valence ~ is_summary)   — per model")
    print("squeeze transform applied; robust (HC1) standard errors")
    print("=" * 68)

    results = []
    for model in ["claude", "chat", "groc"]:
        sub = long[long["model"] == model].copy()

        # Squeeze valence into the open interval (0,1) so omission cases
        # (valence exactly 0 or 1) are kept rather than dropped.
        sub["valence_sq"] = squeeze(sub["valence"].values)

        X = sm.add_constant(sub["is_summary"].astype(float))
        y = sub["valence_sq"].values

        fit = BetaModel(y, X).fit(disp=0)

        # Robust standard errors (HC1) for the mean (location) parameters.
        try:
            robust = fit.get_robustcov_results(cov_type="HC1")
            coef = np.asarray(robust.params)
            se = np.asarray(robust.bse)
            pvals = np.asarray(robust.pvalues)
        except Exception:
            # Fallback to model SEs if robust cov is unavailable for this fit.
            coef = np.asarray(fit.params)
            se = np.asarray(fit.bse)
            pvals = np.asarray(fit.pvalues)

        # Locate the 'is_summary' mean parameter by name, then index positionally.
        param_names = list(fit.params.index) if hasattr(fit.params, "index") \
            else [str(i) for i in range(len(coef))]
        idx = next(i for i, nm in enumerate(param_names)
                   if "is_summary" in str(nm))
        b = coef[idx]
        b_se = se[idx]
        b_p = pvals[idx]

        # Convenience: mean valence for originals vs summaries (raw, not squeezed).
        mean_orig = sub.loc[sub.is_summary == 0, "valence"].mean()
        mean_sum = sub.loc[sub.is_summary == 1, "valence"].mean()

        direction = ("more liberal" if b > 0 else
                     "more conservative" if b < 0 else "no change")
        sig = "YES" if b_p < 0.05 else "no"

        print(f"\n{model.upper()}")
        print(f"  mean valence  original={mean_orig:.4f}   summary={mean_sum:.4f}")
        print(f"  is_summary coef = {b:+.4f}   robust SE = {b_se:.4f}   "
              f"p = {b_p:.4f}   sig={sig}")
        print(f"  -> summaries shift valence {direction} (coef on logit scale)")

        results.append({
            "model": model,
            "n_texts": len(sub),
            "mean_valence_original": round(mean_orig, 4),
            "mean_valence_summary": round(mean_sum, 4),
            "is_summary_coef": round(b, 4),
            "robust_se": round(b_se, 4),
            "p_value": round(b_p, 4),
            "significant": sig,
            "direction": direction,
        })

    res_df = pd.DataFrame(results)
    out = os.path.join(DATA_DIR, "valence_regression_results.csv")
    res_df.to_csv(out, index=False)
    print("\n" + "=" * 68)
    print(res_df.to_string(index=False))
    print(f"\nSaved: {out}")
    print("\nNote: the coefficient is on the beta-regression (logit) scale, so its "
          "sign and significance are what matter; the mean-valence columns give "
          "the plain-language size of the shift.")


if __name__ == "__main__":
    main()
