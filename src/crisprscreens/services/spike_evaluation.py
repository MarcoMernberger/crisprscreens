"""
Evaluation metrics for spike-in quality control in CRISPR screens.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from scipy.stats import mannwhitneyu, spearmanr


def classify_spike_genes(gene_ids: pd.Series) -> pd.Series:
    """
    Classify genes into POS, NEG, NEUTRAL, or OTHER based on ID.

    Parameters
    ----------
    gene_ids : pd.Series
        Gene identifiers

    Returns
    -------
    pd.Series
        Classification labels
    """

    def classify(gene_id):
        if not isinstance(gene_id, str):
            return "OTHER"
        if gene_id.startswith("SPIKE_POS_"):
            return "POS"
        elif gene_id.startswith("SPIKE_NEG_"):
            return "NEG"
        elif gene_id.startswith("SPIKE_NEUTRAL_"):
            return "NEUTRAL"
        else:
            return "OTHER"

    return gene_ids.apply(classify)


def calculate_precision_recall(
    df: pd.DataFrame,
    fdr_col: str,
    lfc_col: str,
    gene_col: str = "id",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 for detecting positive and negative spike-ins.

    Parameters
    ----------
    df : pd.DataFrame
        MAGeCK gene summary output
    fdr_col : str
        Column name for FDR (e.g., 'neg|fdr' or 'pos|fdr')
    lfc_col : str
        Column name for log fold change (e.g., 'neg|lfc' or 'pos|lfc')
    gene_col : str
        Column name for gene IDs
    fdr_threshold : float
        FDR threshold for significance
    lfc_threshold : float
        Absolute log fold change threshold

    Returns
    -------
    Dict[str, float]
        Dictionary with precision, recall, F1, etc.
    """
    df = df.copy()
    df["spike_type"] = classify_spike_genes(df[gene_col])

    # Filter only spike-in genes
    spike_df = df[df["spike_type"].isin(["POS", "NEG", "NEUTRAL"])].copy()

    if len(spike_df) == 0:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "n_true_pos": 0,
            "n_false_pos": 0,
            "n_false_neg": 0,
            "n_true_neg": 0,
        }

    # Determine expected direction from column name
    is_negative_selection = "neg" in fdr_col.lower()

    # True labels
    if is_negative_selection:
        # For negative selection: NEG spike-ins should be significant
        y_true = (spike_df["spike_type"] == "NEG").astype(int)
    else:
        # For positive selection: POS spike-ins should be significant
        y_true = (spike_df["spike_type"] == "POS").astype(int)

    # Predicted labels: significant by FDR and LFC
    y_pred = (
        (spike_df[fdr_col] < fdr_threshold)
        & (spike_df[lfc_col].abs() > lfc_threshold)
    ).astype(int)

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix elements
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_true_pos": int(tp),
        "n_false_pos": int(fp),
        "n_false_neg": int(fn),
        "n_true_neg": int(tn),
        "n_total_spikes": len(spike_df),
        "n_expected_hits": int(y_true.sum()),
        "n_detected_hits": int(y_pred.sum()),
    }


def squash(x, scale: float = 1.0, method: str = "linear"):
    """
    Vectorized squash: accepts pd.Series or scalar.
    Methods:
      - "linear": x/(x+scale)
      - "arctan": (2/pi)*arctan(x/scale) (good for effect sizes like Cohen's d)
      - "tanh": tanh(x/scale)
    Returns same type as input (Series or scalar). Outputs clipped to [0.0, 1.0].
    """
    # Series input -> return Series
    if isinstance(x, pd.Series):
        x_s = x.astype(float).copy().clip(lower=0.0)
        if method == "arctan":
            out = (2.0 / np.pi) * np.arctan(x_s / float(scale))
        elif method == "tanh":
            out = np.tanh(x_s / float(scale))
        else:
            out = x_s / (x_s + float(scale))
        # clip numerical artefacts and keep same index
        out = out.clip(lower=0.0, upper=1.0)
        return out

    # Scalar fallback
    x = max(0.0, float(x))
    if method == "arctan":
        out = float((2.0 / np.pi) * np.arctan(x / float(scale)))
    elif method == "tanh":
        out = float(np.tanh(x / float(scale)))
    else:
        out = x / (x + scale)
    return float(min(max(out, 0.0), 1.0))


# ...existing code...


def calculate_separation_metrics(
    df: pd.DataFrame,
    lfc_col: str,
    fdr_col: str,
    gene_col: str = "id",
) -> Dict[str, float]:
    """
    Calculate separation between spike-in groups.

    Parameters
    ----------
    df : pd.DataFrame
        MAGeCK gene summary output
    lfc_col : str
        Column name for log fold change
    fdr_col : str
        Column name for FDR
    gene_col : str
        Column name for gene IDs

    Returns
    -------
    Dict[str, float]
        Separation metrics (effect sizes, p-values)
    """
    df = df.copy()
    df["spike_type"] = classify_spike_genes(df[gene_col])

    pos_lfc = df[df["spike_type"] == "POS"][lfc_col].dropna()
    neg_lfc = df[df["spike_type"] == "NEG"][lfc_col].dropna()
    neutral_lfc = df[df["spike_type"] == "NEUTRAL"][lfc_col].dropna()

    results = {}

    # Mean differences
    if len(pos_lfc) > 0 and len(neutral_lfc) > 0:
        results["pos_vs_neutral_mean_diff"] = (
            pos_lfc.mean() - neutral_lfc.mean()
        )
        results["pos_vs_neutral_median_diff"] = (
            pos_lfc.median() - neutral_lfc.median()
        )

        # Mann-Whitney U test
        stat, pval = mannwhitneyu(pos_lfc, neutral_lfc, alternative="two-sided")
        results["pos_vs_neutral_pvalue"] = pval

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((pos_lfc.std() ** 2 + neutral_lfc.std() ** 2) / 2)
        if pooled_std > 0:
            results["pos_vs_neutral_cohens_d"] = (
                pos_lfc.mean() - neutral_lfc.mean()
            ) / pooled_std
        else:
            results["pos_vs_neutral_cohens_d"] = np.nan

    if len(neg_lfc) > 0 and len(neutral_lfc) > 0:
        results["neg_vs_neutral_mean_diff"] = (
            neg_lfc.mean() - neutral_lfc.mean()
        )
        results["neg_vs_neutral_median_diff"] = (
            neg_lfc.median() - neutral_lfc.median()
        )

        # Mann-Whitney U test
        stat, pval = mannwhitneyu(neg_lfc, neutral_lfc, alternative="two-sided")
        results["neg_vs_neutral_pvalue"] = pval

        # Effect size
        pooled_std = np.sqrt((neg_lfc.std() ** 2 + neutral_lfc.std() ** 2) / 2)
        if pooled_std > 0:
            results["neg_vs_neutral_cohens_d"] = (
                neg_lfc.mean() - neutral_lfc.mean()
            ) / pooled_std
        else:
            results["neg_vs_neutral_cohens_d"] = np.nan

    if len(pos_lfc) > 0 and len(neg_lfc) > 0:
        results["pos_vs_neg_mean_diff"] = pos_lfc.mean() - neg_lfc.mean()
        results["pos_vs_neg_median_diff"] = pos_lfc.median() - neg_lfc.median()

        stat, pval = mannwhitneyu(pos_lfc, neg_lfc, alternative="two-sided")
        results["pos_vs_neg_pvalue"] = pval

        pooled_std = np.sqrt((pos_lfc.std() ** 2 + neg_lfc.std() ** 2) / 2)
        if pooled_std > 0:
            results["pos_vs_neg_cohens_d"] = (
                pos_lfc.mean() - neg_lfc.mean()
            ) / pooled_std
        else:
            results["pos_vs_neg_cohens_d"] = np.nan

    return results


def calculate_ranking_power(
    df: pd.DataFrame,
    rank_col: str,
    gene_col: str = "id",
) -> Dict[str, float]:
    """
    Calculate ranking power: how well are true hits ranked.

    Parameters
    ----------
    df : pd.DataFrame
        MAGeCK gene summary output
    rank_col : str
        Column name for rank (e.g., 'neg|rank' or 'pos|rank')
    gene_col : str
        Column name for gene IDs

    Returns
    -------
    Dict[str, float]
        Ranking metrics (median rank, top-k enrichment)
    """
    df = df.copy()
    df["spike_type"] = classify_spike_genes(df[gene_col])

    # Determine expected hits
    is_negative_selection = "neg" in rank_col.lower()
    if is_negative_selection:
        expected_type = "NEG"
    else:
        expected_type = "POS"

    spike_df = df[df["spike_type"].isin(["POS", "NEG", "NEUTRAL"])].copy()

    if len(spike_df) == 0:
        return {}

    expected_hits = spike_df[spike_df["spike_type"] == expected_type]
    n_expected = len(expected_hits)

    if n_expected == 0:
        return {}

    ranks = expected_hits[rank_col].dropna()

    results = {
        "median_rank": ranks.median(),
        "mean_rank": ranks.mean(),
        "min_rank": ranks.min(),
        "max_rank": ranks.max(),
        "n_expected_hits": n_expected,
    }

    # Top-k enrichment: fraction of expected hits in top k genes
    total_genes = len(df)
    for k in [10, 50, 100, 500]:
        if k <= total_genes:
            n_in_top_k = (ranks <= k).sum()
            results[f"frac_in_top_{k}"] = n_in_top_k / n_expected

    # Area under the cumulative curve (AUCC) - early enrichment metric
    sorted_ranks = np.sort(ranks.values)
    cumulative_fraction = np.arange(1, len(sorted_ranks) + 1) / len(
        sorted_ranks
    )
    normalized_ranks = sorted_ranks / total_genes
    aucc = np.trapz(cumulative_fraction, normalized_ranks)
    results["aucc"] = aucc  # Higher is better (ideal = 1.0)

    return results


def calculate_auc_metrics(
    df: pd.DataFrame,
    score_col: str,
    fdr_col: str,
    gene_col: str = "id",
) -> Dict[str, float]:
    """
    Calculate AUC-ROC and AUC-PR for spike-in detection.

    Parameters
    ----------
    df : pd.DataFrame
        MAGeCK gene summary output
    score_col : str
        Column name for score (e.g., 'neg|score')
    fdr_col : str
        Column name for FDR
    gene_col : str
        Column name for gene IDs

    Returns
    -------
    Dict[str, float]
        AUC-ROC and AUC-PR
    """
    df = df.copy()
    df["spike_type"] = classify_spike_genes(df[gene_col])

    spike_df = df[df["spike_type"].isin(["POS", "NEG", "NEUTRAL"])].copy()

    if len(spike_df) == 0:
        return {}

    # Determine expected hits
    is_negative_selection = "neg" in score_col.lower()
    if is_negative_selection:
        y_true = (spike_df["spike_type"] == "NEG").astype(int)
    else:
        y_true = (spike_df["spike_type"] == "POS").astype(int)

    # Use -log10(FDR) as score (higher = more significant)
    fdr_vals = spike_df[fdr_col].replace(0, 1e-300)  # avoid log(0)
    y_score = -np.log10(fdr_vals)

    results = {}

    # Only calculate if we have both positive and negative examples
    if y_true.sum() > 0 and (1 - y_true).sum() > 0:
        try:
            auc_roc = roc_auc_score(y_true, y_score)
            results["auc_roc"] = auc_roc
        except:
            results["auc_roc"] = np.nan

        try:
            auc_pr = average_precision_score(y_true, y_score)
            results["auc_pr"] = auc_pr
        except:
            results["auc_pr"] = np.nan

    return results


def calculate_spike_consistency(
    df: pd.DataFrame,
    lfc_col: str,
    gene_col: str = "id",
) -> Dict[str, float]:
    """
    Calculate consistency within spike-in groups (lower variance = better).

    Parameters
    ----------
    df : pd.DataFrame
        MAGeCK gene summary output
    lfc_col : str
        Column name for log fold change
    gene_col : str
        Column name for gene IDs

    Returns
    -------
    Dict[str, float]
        CV and IQR for each spike-in group
    """
    df = df.copy()
    df["spike_type"] = classify_spike_genes(df[gene_col])

    results = {}

    for spike_type in ["POS", "NEG", "NEUTRAL"]:
        values = df[df["spike_type"] == spike_type][lfc_col].dropna()

        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            cv = std_val / abs(mean_val) if mean_val != 0 else np.nan
            iqr = values.quantile(0.75) - values.quantile(0.25)

            results[f"{spike_type.lower()}_mean_lfc"] = mean_val
            results[f"{spike_type.lower()}_std_lfc"] = std_val
            results[f"{spike_type.lower()}_cv"] = cv
            results[f"{spike_type.lower()}_iqr"] = iqr
            results[f"{spike_type.lower()}_n"] = len(values)

    return results


def evaluate_mageck_result(
    gene_summary_file: Path,
    comparison_name: str,
    direction: str = "neg",  # "neg" or "pos"
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
    gene_col: str = "id",
) -> pd.DataFrame:
    """
    Comprehensive evaluation of a single MAGeCK result file.

    Parameters
    ----------
    gene_summary_file : Path
        Path to MAGeCK gene_summary.tsv file
    comparison_name : str
        Name for this comparison (e.g., "RRA_paired_median")
    direction : str
        "neg" for negative selection, "pos" for positive selection
    fdr_threshold : float
        FDR threshold for significance
    lfc_threshold : float
        Absolute LFC threshold
    gene_col : str
        Column name for gene IDs

    Returns
    -------
    pd.DataFrame
        Single row with all metrics
    """
    df = pd.read_csv(gene_summary_file, sep="\t")

    # Column names
    fdr_col = f"{direction}|fdr"
    lfc_col = f"{direction}|lfc"
    rank_col = f"{direction}|rank"
    score_col = f"{direction}|score"

    # Calculate all metrics
    metrics = {"comparison": comparison_name, "direction": direction}

    # Precision/Recall
    pr_metrics = calculate_precision_recall(
        df, fdr_col, lfc_col, gene_col, fdr_threshold, lfc_threshold
    )
    metrics.update(pr_metrics)

    # Separation
    sep_metrics = calculate_separation_metrics(df, lfc_col, fdr_col, gene_col)
    metrics.update(sep_metrics)

    # Ranking power
    rank_metrics = calculate_ranking_power(df, rank_col, gene_col)
    metrics.update(rank_metrics)

    # AUC metrics
    auc_metrics = calculate_auc_metrics(df, score_col, fdr_col, gene_col)
    metrics.update(auc_metrics)

    # Consistency
    cons_metrics = calculate_spike_consistency(df, lfc_col, gene_col)
    metrics.update(cons_metrics)

    return pd.DataFrame([metrics])


def evaluate_multiple_mageck_results(
    results_dict: Dict[str, Path],
    # direction: str = "neg",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
    gene_col: str = "id",
    combine_directions: bool = True,
) -> pd.DataFrame:
    """
    Evaluate multiple MAGeCK results and compare them.

    Parameters
    ----------
    results_dict : Dict[str, Path]
        Dictionary mapping comparison names to gene_summary.tsv files
    direction : str
        "neg" for negative selection, "pos" for positive selection
    fdr_threshold : float
        FDR threshold for significance
    lfc_threshold : float
        Absolute LFC threshold
    gene_col : str
        Column name for gene IDs
    combine_directions : bool
        If True, add aggregated "both" rows combining pos and neg per comparison

    Returns
    -------
    pd.DataFrame
        Combined metrics for all comparisons
    """
    all_results = []

    for name, filepath in results_dict.items():
        if not Path(filepath).exists():
            print(f"Warning: {filepath} does not exist, skipping {name}")
            continue

        result_neg = evaluate_mageck_result(
            filepath, name, "neg", fdr_threshold, lfc_threshold, gene_col
        )
        all_results.append(result_neg)

        result_pos = evaluate_mageck_result(
            filepath, name, "pos", fdr_threshold, lfc_threshold, gene_col
        )
        all_results.append(result_pos)

    if len(all_results) == 0:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Optionally combine pos/neg into a single 'both' summary per comparison
    if combine_directions and len(combined) > 0:
        combined_both = _combine_pos_neg_by_comparison(
            combined, weight_col="n_expected_hits"
        )
        if len(combined_both) > 0:
            combined = pd.concat([combined, combined_both], ignore_index=True)

    return combined


# Helper functions for combining pos/neg rows
def _safe_weighted_mean(vals, weights):
    """Compute weighted mean ignoring NaNs. If weights all zero or missing, fall back to simple mean."""
    vals = np.asarray(vals, dtype=float)
    if weights is None:
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            return np.nan
        return float(np.nanmean(vals))
    w = np.asarray(weights, dtype=float)
    # Filter out NaNs
    mask = ~np.isnan(vals) & (~np.isnan(w))
    if mask.sum() == 0:
        return np.nan
    vals_f = vals[mask]
    w_f = w[mask]
    if np.nansum(w_f) == 0:
        return float(np.mean(vals_f))
    return float(np.sum(vals_f * w_f) / np.sum(w_f))


def _safe_weighted_harmonic_mean(vals, weights=None):
    """Weighted harmonic mean; treats zeros as valid (resulting in 0) and ignores NaNs.
    If all non-NaN vals are > 0 and weights sum to > 0, returns weighted harmonic mean.
    """
    vals = np.asarray(vals, dtype=float)
    if weights is None:
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan
        if np.any(vals == 0):
            return 0.0
        return float(len(vals) / np.sum(1.0 / vals))

    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(vals) & ~np.isnan(w)
    if mask.sum() == 0:
        return np.nan
    vals_f = vals[mask]
    w_f = w[mask]
    if w_f.sum() == 0:
        # fall back to unweighted harmonic
        return _safe_weighted_harmonic_mean(vals_f, None)
    if np.any(vals_f == 0):
        # weighted harmonic mean zero if any value zero with positive weight
        if np.any((vals_f == 0) & (w_f > 0)):
            return 0.0
        # else ignore zero-weight zeros
        nonzero_mask = vals_f > 0
        if nonzero_mask.sum() == 0:
            return 0.0
        vals_f = vals_f[nonzero_mask]
        w_f = w_f[nonzero_mask]
    return float(np.sum(w_f) / np.sum(w_f / vals_f))


def _combine_pos_neg_by_comparison(
    eval_df: pd.DataFrame, weight_col: str = "n_expected_hits"
) -> pd.DataFrame:
    """Combine pos/neg rows per comparison into a single 'both' row.

    Rules:
    - columns starting with 'n_' are summed
    - precision/recall are combined with weighted harmonic mean (weights from weight_col)
    - f1 is recalculated from combined precision/recall
    - other numeric columns are weighted averages (weights from weight_col)
    - non-numeric columns copied from pos if available else neg
    """
    rows = []
    numeric_cols = eval_df.select_dtypes(include=[np.number]).columns

    for name in eval_df["comparison"].unique():
        sub = eval_df[eval_df["comparison"] == name]
        pos = sub[sub["direction"] == "pos"]
        neg = sub[sub["direction"] == "neg"]

        if pos.empty and neg.empty:
            continue
        if pos.empty:
            r = neg.iloc[0].copy()
            r["direction"] = "both"
            rows.append(r)
            continue
        if neg.empty:
            r = pos.iloc[0].copy()
            r["direction"] = "both"
            rows.append(r)
            continue

        p = pos.iloc[0]
        n = neg.iloc[0]

        combined = p.copy()
        combined["direction"] = "both"

        # Weights
        w_p = (
            float(p.get(weight_col, np.nan))
            if not pd.isna(p.get(weight_col, np.nan))
            else np.nan
        )
        w_n = (
            float(n.get(weight_col, np.nan))
            if not pd.isna(n.get(weight_col, np.nan))
            else np.nan
        )

        # Normalize weights: if both NaN or both zero, fallback to 1/1
        if (pd.isna(w_p) or w_p == 0) and (pd.isna(w_n) or w_n == 0):
            w_p = w_n = 1.0
        else:
            if pd.isna(w_p) or w_p < 0:
                w_p = 0.0
            if pd.isna(w_n) or w_n < 0:
                w_n = 0.0

        for col in numeric_cols:
            vp = p.get(col, np.nan)
            vn = n.get(col, np.nan)

            if pd.isna(vp) and pd.isna(vn):
                combined[col] = np.nan
                continue

            if str(col).startswith("n_"):
                combined[col] = (0 if pd.isna(vp) else vp) + (
                    0 if pd.isna(vn) else vn
                )
                continue

            if col in ("precision", "recall"):
                combined[col] = _safe_weighted_harmonic_mean(
                    [vp, vn], weights=[w_p, w_n]
                )
                continue

            # otherwise weighted mean
            combined[col] = _safe_weighted_mean([vp, vn], weights=[w_p, w_n])

        # Recompute f1 from combined precision & recall if possible
        p_comb = combined.get("precision", np.nan)
        r_comb = combined.get("recall", np.nan)
        if (
            not pd.isna(p_comb)
            and not pd.isna(r_comb)
            and (p_comb + r_comb) > 0
        ):
            combined["f1"] = 2 * p_comb * r_comb / (p_comb + r_comb)
        else:
            # fall back to weighted mean of f1s
            combined["f1"] = _safe_weighted_mean(
                [p.get("f1", np.nan), n.get("f1", np.nan)], weights=[w_p, w_n]
            )

        # Non-numeric: keep from pos if available else neg
        for col in set(eval_df.columns) - set(numeric_cols):
            if col in ("comparison", "direction"):
                continue
            combined[col] = p.get(col) if pd.notna(p.get(col)) else n.get(col)

        rows.append(combined)

    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def rank_mageck_methods(
    eval_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Rank MAGeCK methods by composite score.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output from evaluate_multiple_mageck_results
    weights : Optional[Dict[str, float]]
        Weights for different metrics (default: equal weights)

    Returns
    -------
    pd.DataFrame
        Ranked methods with composite scores
    """
    if len(eval_df) == 0:
        return pd.DataFrame()

    # Default weights (higher is better)
    if weights is None:
        weights = {
            "f1": 3.0,  # F1 score (most important)
            "auc_roc": 2.0,  # AUC-ROC
            "auc_pr": 2.0,  # AUC-PR
            "aucc": 2.0,  # Ranking power
            "pos_vs_neg_cohens_d": 1.5,  # Separation
            "neg_vs_neutral_cohens_d": 1.5,  # Separation
            "neg_cv": -1.0,  # Consistency (lower is better)
        }

    # Normalize metrics to 0-1 scale
    df = eval_df.copy()
    composite_score = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    df["detection_score"] = (
        0.5 * df["recall"]
        + 0.3 * df["frac_in_top_50"]
        + 0.2 * df["frac_in_top_100"]
    )
    df["quality_score"] = (
        0.4 * squash(df["pos_vs_neg_cohens_d"])
        + 0.3 * squash(df["neg_vs_neutral_cohens_d"])
        - 0.3 * df["neg_cv"]
    )
    df["final_score"] = 0.7 * df["detection_score"] + 0.3 * df["quality_score"]
    for metric, weight in weights.items():
        if metric in df.columns:
            vals = df[metric].copy()

            # Handle NaN
            if vals.isna().all():
                continue

            # Normalize to 0-1
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                if weight > 0:
                    # Higher is better
                    normalized = (vals - vmin) / (vmax - vmin)
                else:
                    # Lower is better (e.g., CV)
                    normalized = 1 - (vals - vmin) / (vmax - vmin)
                    weight = abs(weight)

                composite_score += normalized * weight
                total_weight += weight

    if total_weight > 0:
        df["composite_score"] = composite_score / total_weight
    else:
        df["composite_score"] = np.nan

    # Rank
    df = df.sort_values("final_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    return df
