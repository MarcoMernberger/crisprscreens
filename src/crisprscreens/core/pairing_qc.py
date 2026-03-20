"""
Quality control for paired vs. unpaired RRA analysis decision.

This module provides QC metrics to help decide whether paired MAGeCK RRA
analysis is beneficial compared to unpaired analysis.
"""

import numpy as np
import pandas as pd  # type: ignore
from pathlib import Path
from typing import Dict, List, Optional, Union
from pandas import DataFrame  # type: ignore
from scipy.stats import spearmanr

from .qc import read_counts, parse_metadata_from_columns
from .mageck import mageck_test


def replicate_gene_ranking_consistency(
    count_df: DataFrame,
    treatment_cols: List[str],
    control_cols: List[str],
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    top_n: int = 100,
) -> Dict:
    """
    Compute per-replicate gene rankings and measure consistency.

    For each treatment replicate, compute a simple gene-level score
    (e.g., median log2FC of sgRNAs) and rank genes. Then measure
    correlation and overlap between replicate rankings.

    Parameters
    ----------
    count_df : DataFrame
        MAGeCK count table
    treatment_cols : list
        Treatment sample column names
    control_cols : list
        Control sample column names
    sgrna_col : str
        sgRNA column name
    gene_col : str
        Gene column name
    top_n : int
        Number of top genes to compare for overlap

    Returns
    -------
    dict
        Results with:
        - replicate_rankings: dict mapping replicate -> gene ranking Series
        - rank_correlations: DataFrame with pairwise Spearman correlations
        - top_n_overlaps: DataFrame with Jaccard indices for top-N genes
        - consistency_score: float (0-1), mean of rank correlations
    """
    from scipy.stats import rankdata

    # Compute CPM
    sample_cols = treatment_cols + control_cols
    cpm_df = count_df.copy()
    for col in sample_cols:
        total = count_df[col].sum()
        cpm_df[col] = (count_df[col] / total) * 1e6

    # Mean control CPM
    mean_control = cpm_df[control_cols].mean(axis=1)

    # For each treatment replicate, compute gene-level log2FC
    replicate_rankings = {}

    for treatment_col in treatment_cols:
        # Compute log2FC per sgRNA
        log2fc = np.log2(cpm_df[treatment_col] + 1) - np.log2(mean_control + 1)

        # Aggregate to gene level (median log2FC)
        gene_scores = (
            pd.DataFrame({gene_col: count_df[gene_col], "log2fc": log2fc})
            .groupby(gene_col)["log2fc"]
            .median()
        )

        # Rank genes (lower log2FC = better rank for depletion screen)
        gene_ranks = pd.Series(
            rankdata(gene_scores, method="average"), index=gene_scores.index
        )

        replicate_rankings[treatment_col] = gene_ranks

    # Compute pairwise rank correlations
    n_reps = len(treatment_cols)
    corr_matrix = pd.DataFrame(
        index=treatment_cols, columns=treatment_cols, dtype=float
    )

    overlap_matrix = pd.DataFrame(
        index=treatment_cols, columns=treatment_cols, dtype=float
    )

    for i, rep1 in enumerate(treatment_cols):
        for j, rep2 in enumerate(treatment_cols):
            ranks1 = replicate_rankings[rep1]
            ranks2 = replicate_rankings[rep2]

            # Align on common genes
            common_genes = ranks1.index.intersection(ranks2.index)
            if len(common_genes) > 0:
                r1 = ranks1[common_genes]
                r2 = ranks2[common_genes]

                # Spearman correlation
                corr, _ = spearmanr(r1, r2)
                corr_matrix.loc[rep1, rep2] = corr

                # Top-N overlap (Jaccard index)
                top1 = set(r1.nsmallest(top_n).index)
                top2 = set(r2.nsmallest(top_n).index)
                jaccard = len(top1 & top2) / len(top1 | top2)
                overlap_matrix.loc[rep1, rep2] = jaccard
            else:
                corr_matrix.loc[rep1, rep2] = np.nan
                overlap_matrix.loc[rep1, rep2] = np.nan

    # Extract off-diagonal values for summary
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    off_diag_corrs = corr_matrix.where(mask).stack().values
    off_diag_overlaps = overlap_matrix.where(mask).stack().values

    consistency_score = (
        np.nanmean(off_diag_corrs) if len(off_diag_corrs) > 0 else np.nan
    )
    mean_overlap = (
        np.nanmean(off_diag_overlaps) if len(off_diag_overlaps) > 0 else np.nan
    )

    return {
        "replicate_rankings": replicate_rankings,
        "rank_correlations": corr_matrix,
        "top_n_overlaps": overlap_matrix,
        "consistency_score": consistency_score,
        "mean_top_n_overlap": mean_overlap,
        "top_n": top_n,
    }


def run_paired_unpaired_comparison(
    count_df: DataFrame,
    treatment_cols: List[str],
    control_cols: List[str],
    out_dir: Union[Path, str],
    control_sgrnas: Optional[Union[Path, str]] = None,
    norm_method: str = "median",
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
) -> Dict:
    """
    Run both paired and unpaired MAGeCK RRA and compare results.

    Parameters
    ----------
    count_df : DataFrame
        MAGeCK count table
    treatment_cols : list
        Treatment sample columns
    control_cols : list
        Control sample columns
    out_dir : Path or str
        Output directory for temporary MAGeCK results
    control_sgrnas : Path or str, optional
        Path to control sgRNA file
    norm_method : str
        Normalization method for MAGeCK
    sgrna_col : str
        sgRNA column name
    gene_col : str
        Gene column name

    Returns
    -------
    dict
        Comparison results with:
        - paired_results: DataFrame with paired analysis gene summary
        - unpaired_results: DataFrame with unpaired analysis gene summary
        - rank_correlation: Spearman correlation of gene ranks
        - top_100_overlap: Jaccard index of top 100 genes
        - direction_agreement: fraction of genes with consistent direction
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Save count table to temp file
    temp_count_file = out_dir / "temp_counts.tsv"
    count_df.to_csv(temp_count_file, sep="\t", index=False)

    # Run unpaired analysis
    print("Running unpaired RRA analysis...")
    mageck_test(
        count_table=temp_count_file,
        treatment_ids=treatment_cols,
        control_ids=control_cols,
        out_dir=out_dir,
        prefix="unpaired",
        control_sgrnas=control_sgrnas,
        norm_method=norm_method,
        paired=False,
        pdf_report=False,
    )

    # Run paired analysis
    print("Running paired RRA analysis...")
    mageck_test(
        count_table=temp_count_file,
        treatment_ids=treatment_cols,
        control_ids=control_cols,
        out_dir=out_dir,
        prefix="paired",
        control_sgrnas=control_sgrnas,
        norm_method=norm_method,
        paired=True,
        pdf_report=False,
    )

    # Load results
    unpaired_summary = pd.read_csv(
        out_dir / "unpaired.gene_summary.tsv", sep="\t"
    )
    paired_summary = pd.read_csv(out_dir / "paired.gene_summary.tsv", sep="\t")

    # Detect column names (MAGeCK may use different names)
    gene_col_actual = "Gene" if "Gene" in paired_summary.columns else "id"
    rank_col_unpaired = (
        "pos|rank" if "pos|rank" in unpaired_summary.columns else "neg|rank"
    )
    rank_col_paired = (
        "pos|rank" if "pos|rank" in paired_summary.columns else "neg|rank"
    )

    # Merge on gene
    merged = unpaired_summary[[gene_col_actual, rank_col_unpaired]].merge(
        paired_summary[[gene_col_actual, rank_col_paired]],
        on=gene_col_actual,
        how="inner",
        suffixes=("_unpaired", "_paired"),
    )

    # Rank correlation
    rank_corr, _ = spearmanr(
        merged[f"{rank_col_unpaired}_unpaired"],
        merged[f"{rank_col_paired}_paired"],
    )

    # Top-100 overlap
    top_100_unpaired = set(
        unpaired_summary.nsmallest(100, rank_col_unpaired)[gene_col_actual]
    )
    top_100_paired = set(
        paired_summary.nsmallest(100, rank_col_paired)[gene_col_actual]
    )
    jaccard = len(top_100_unpaired & top_100_paired) / len(
        top_100_unpaired | top_100_paired
    )

    # Direction agreement (both depletion or both enrichment)
    # Use neg|rank vs pos|rank to infer direction
    if (
        "neg|rank" in unpaired_summary.columns
        and "pos|rank" in unpaired_summary.columns
    ):
        merged_full = unpaired_summary[
            [gene_col_actual, "neg|rank", "pos|rank"]
        ].merge(
            paired_summary[[gene_col_actual, "neg|rank", "pos|rank"]],
            on=gene_col_actual,
            how="inner",
            suffixes=("_unpaired", "_paired"),
        )

        # Direction: depleted if neg|rank < pos|rank
        merged_full["dir_unpaired"] = (
            merged_full["neg|rank_unpaired"] < merged_full["pos|rank_unpaired"]
        ).astype(
            int
        ) * 2 - 1  # -1 for enriched, 1 for depleted

        merged_full["dir_paired"] = (
            merged_full["neg|rank_paired"] < merged_full["pos|rank_paired"]
        ).astype(int) * 2 - 1

        direction_agreement = (
            merged_full["dir_unpaired"] == merged_full["dir_paired"]
        ).mean()
    else:
        direction_agreement = np.nan

    return {
        "paired_results": paired_summary,
        "unpaired_results": unpaired_summary,
        "rank_correlation": rank_corr,
        "top_100_overlap": jaccard,
        "direction_agreement": direction_agreement,
    }


def downsampling_stability_qc(
    count_df: DataFrame,
    treatment_cols: List[str],
    control_cols: List[str],
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    fractions: List[float] = [0.5, 0.75, 1.0],
    n_bootstrap: int = 10,
    top_n: int = 100,
) -> Dict:
    """
    Test ranking stability under downsampling (bootstrap).

    Downsample reads to fractions of original depth, compute gene rankings,
    and measure stability (Jaccard index of top-N genes).

    Parameters
    ----------
    count_df : DataFrame
        MAGeCK count table
    treatment_cols : list
        Treatment sample columns
    control_cols : list
        Control sample columns
    sgrna_col : str
        sgRNA column name
    gene_col : str
        Gene column name
    fractions : list of float
        Fractions of reads to downsample to (e.g., [0.5, 0.75, 1.0])
    n_bootstrap : int
        Number of bootstrap iterations per fraction
    top_n : int
        Number of top genes to compare

    Returns
    -------
    dict
        Stability results with:
        - stability_by_fraction: DataFrame with mean/std Jaccard per fraction
        - mean_stability: float, overall mean Jaccard
    """
    from scipy.stats import rankdata

    sample_cols = treatment_cols + control_cols

    # Compute CPM once
    cpm_df = count_df.copy()
    for col in sample_cols:
        total = count_df[col].sum()
        cpm_df[col] = (count_df[col] / total) * 1e6

    # Full-depth ranking (reference)
    mean_control = cpm_df[control_cols].mean(axis=1)
    mean_treatment = cpm_df[treatment_cols].mean(axis=1)
    log2fc_full = np.log2(mean_treatment + 1) - np.log2(mean_control + 1)

    gene_scores_full = (
        pd.DataFrame({gene_col: count_df[gene_col], "log2fc": log2fc_full})
        .groupby(gene_col)["log2fc"]
        .median()
    )

    ranks_full = pd.Series(
        rankdata(gene_scores_full, method="average"),
        index=gene_scores_full.index,
    )
    top_genes_full = set(ranks_full.nsmallest(top_n).index)

    # Downsampling
    stability_records = []

    for frac in fractions:
        if frac == 1.0:
            # No downsampling, compare to itself
            stability_records.append(
                {
                    "fraction": frac,
                    "bootstrap": 0,
                    "jaccard": 1.0,
                }
            )
            continue

        for i in range(n_bootstrap):
            # Downsample counts
            downsampled_df = count_df.copy()
            for col in sample_cols:
                counts = count_df[col].values
                total_counts = counts.sum()
                target_counts = int(total_counts * frac)

                # Multinomial resampling
                probs = counts / total_counts
                downsampled = np.random.multinomial(target_counts, probs)
                downsampled_df[col] = downsampled

            # Compute CPM on downsampled
            cpm_down = downsampled_df.copy()
            for col in sample_cols:
                total = downsampled_df[col].sum()
                cpm_down[col] = (downsampled_df[col] / total) * 1e6

            # Gene-level log2FC
            mean_control_down = cpm_down[control_cols].mean(axis=1)
            mean_treatment_down = cpm_down[treatment_cols].mean(axis=1)
            log2fc_down = np.log2(mean_treatment_down + 1) - np.log2(
                mean_control_down + 1
            )

            gene_scores_down = (
                pd.DataFrame(
                    {gene_col: downsampled_df[gene_col], "log2fc": log2fc_down}
                )
                .groupby(gene_col)["log2fc"]
                .median()
            )

            ranks_down = pd.Series(
                rankdata(gene_scores_down, method="average"),
                index=gene_scores_down.index,
            )
            top_genes_down = set(ranks_down.nsmallest(top_n).index)

            # Jaccard
            jaccard = len(top_genes_full & top_genes_down) / len(
                top_genes_full | top_genes_down
            )

            stability_records.append(
                {
                    "fraction": frac,
                    "bootstrap": i,
                    "jaccard": jaccard,
                }
            )

    stability_df = pd.DataFrame(stability_records)

    # Summary per fraction
    summary = (
        stability_df.groupby("fraction")["jaccard"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    mean_stability = stability_df["jaccard"].mean()

    return {
        "stability_by_fraction": summary,
        "all_bootstraps": stability_df,
        "mean_stability": mean_stability,
        "top_n": top_n,
    }


def positive_control_enrichment(
    count_df: DataFrame,
    treatment_cols: List[str],
    control_cols: List[str],
    positive_control_genes: Union[Path, str, List[str]],
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    top_n: int = 100,
) -> Dict:
    """
    Check if known positive controls are enriched in top hits.

    Parameters
    ----------
    count_df : DataFrame
        MAGeCK count table
    treatment_cols : list
        Treatment sample columns
    control_cols : list
        Control sample columns
    positive_control_genes : Path, str, or list
        Path to file with positive control gene names (one per line)
        or list of gene names
    sgrna_col : str
        sgRNA column name
    gene_col : str
        Gene column name
    top_n : int
        Number of top genes to check for enrichment

    Returns
    -------
    dict
        Results with:
        - n_positive_controls: total number of positive controls
        - n_in_top_n: number of positive controls in top-N
        - fraction_in_top_n: fraction of positive controls in top-N
        - enrichment_pvalue: hypergeometric test p-value
    """
    from scipy.stats import hypergeom

    # Load positive controls
    if isinstance(positive_control_genes, (str, Path)):
        with open(positive_control_genes) as f:
            positive_controls = {line.strip() for line in f if line.strip()}
    else:
        positive_controls = set(positive_control_genes)

    # Compute gene rankings (same as in replicate_gene_ranking_consistency)
    sample_cols = treatment_cols + control_cols
    cpm_df = count_df.copy()
    for col in sample_cols:
        total = count_df[col].sum()
        cpm_df[col] = (count_df[col] / total) * 1e6

    mean_control = cpm_df[control_cols].mean(axis=1)
    mean_treatment = cpm_df[treatment_cols].mean(axis=1)
    log2fc = np.log2(mean_treatment + 1) - np.log2(mean_control + 1)

    gene_scores = (
        pd.DataFrame({gene_col: count_df[gene_col], "log2fc": log2fc})
        .groupby(gene_col)["log2fc"]
        .median()
    )

    from scipy.stats import rankdata

    gene_ranks = pd.Series(
        rankdata(gene_scores, method="average"), index=gene_scores.index
    )

    # Top-N genes
    top_genes = set(gene_ranks.nsmallest(top_n).index)

    # Check overlap with positive controls
    n_positive_controls = len(positive_controls)
    positive_in_top = top_genes & positive_controls
    n_in_top_n = len(positive_in_top)
    fraction_in_top_n = (
        n_in_top_n / n_positive_controls if n_positive_controls > 0 else 0
    )

    # Hypergeometric test for enrichment
    # M = total number of genes
    # n = number of positive controls
    # N = top_n
    # k = number of positive controls in top_n
    M = len(gene_ranks)
    n = n_positive_controls
    N = top_n
    k = n_in_top_n

    # P(X >= k)
    pvalue = hypergeom.sf(k - 1, M, n, N)

    return {
        "n_positive_controls": n_positive_controls,
        "n_in_top_n": n_in_top_n,
        "fraction_in_top_n": fraction_in_top_n,
        "enrichment_pvalue": pvalue,
        "positive_controls_in_top": list(positive_in_top),
        "top_n": top_n,
    }


def comprehensive_pairing_qc(
    count_table: Union[Path, str],
    treatment_ids: List[str],
    control_ids: List[str],
    output_dir: Union[Path, str],
    control_sgrnas: Optional[Union[Path, str]] = None,
    positive_control_genes: Optional[Union[Path, str, List[str]]] = None,
    norm_method: str = "median",
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    delimiter: str = "_",
) -> Dict:
    """
    Comprehensive QC to decide between paired and unpaired RRA analysis.

    Runs all QC checks and generates a recommendation.

    Parameters
    ----------
    count_table : Path or str
        Path to MAGeCK count table
    treatment_ids : list
        Treatment sample column names
    control_ids : list
        Control sample column names
    output_dir : Path or str
        Output directory for results and plots
    control_sgrnas : Path or str, optional
        Path to control sgRNA file for MAGeCK normalization
    positive_control_genes : Path, str, or list, optional
        Positive control genes for enrichment analysis
    norm_method : str
        MAGeCK normalization method
    sgrna_col : str
        sgRNA column name
    gene_col : str
        Gene column name
    delimiter : str
        Delimiter for parsing condition_replicate

    Returns
    -------
    dict
        Comprehensive results with:
        - replicate_consistency: dict from replicate_gene_ranking_consistency
        - paired_vs_unpaired: dict from run_paired_unpaired_comparison
        - downsampling_stability: dict from downsampling_stability_qc
        - positive_control_enrichment: dict (if positive controls provided)
        - recommendation: str ("paired" or "unpaired")
        - recommendation_reasons: list of reasons
        - recommendation_score: float (0-1, confidence in recommendation)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("Paired vs. Unpaired RRA Analysis QC")
    print("=" * 60)

    # Load count table
    print("\n[1/5] Loading count table...")
    count_df, sample_cols = read_counts(count_table, sgrna_col, gene_col)
    print(f"  Loaded {len(count_df)} sgRNAs, {len(sample_cols)} samples")

    # Check if we have replicates
    conditions, metadata_df = parse_metadata_from_columns(
        sample_cols, delimiter
    )
    n_treatment_reps = len(treatment_ids)
    n_control_reps = len(control_ids)

    if n_treatment_reps < 2:
        print("\n⚠ WARNING: Fewer than 2 treatment replicates found.")
        print("  Paired analysis requires replicates. Defaulting to unpaired.")
        return {
            "recommendation": "unpaired",
            "recommendation_reasons": ["Fewer than 2 treatment replicates"],
            "recommendation_score": 1.0,
        }

    # 1. Replicate consistency
    print("\n[2/5] Analyzing replicate consistency...")
    rep_consistency = replicate_gene_ranking_consistency(
        count_df,
        treatment_ids,
        control_ids,
        sgrna_col,
        gene_col,
        top_n=100,
    )
    print(f"  Consistency score: {rep_consistency['consistency_score']:.3f}")
    print(
        f"  Mean top-100 overlap: {rep_consistency['mean_top_n_overlap']:.3f}"
    )

    # Save
    rep_consistency["rank_correlations"].to_csv(
        output_dir / "replicate_rank_correlations.tsv", sep="\t"
    )
    rep_consistency["top_n_overlaps"].to_csv(
        output_dir / "replicate_top100_overlaps.tsv", sep="\t"
    )

    # 2. Paired vs. unpaired comparison
    print("\n[3/5] Comparing paired vs. unpaired RRA...")
    paired_vs_unpaired = run_paired_unpaired_comparison(
        count_df,
        treatment_ids,
        control_ids,
        output_dir / "mageck_temp",
        control_sgrnas,
        norm_method,
        sgrna_col,
        gene_col,
    )
    print(f"  Rank correlation: {paired_vs_unpaired['rank_correlation']:.3f}")
    print(f"  Top-100 overlap: {paired_vs_unpaired['top_100_overlap']:.3f}")
    print(
        f"  Direction agreement: {paired_vs_unpaired['direction_agreement']:.3f}"
    )

    # Save
    paired_vs_unpaired["paired_results"].to_csv(
        output_dir / "paired_gene_summary.tsv", sep="\t", index=False
    )
    paired_vs_unpaired["unpaired_results"].to_csv(
        output_dir / "unpaired_gene_summary.tsv", sep="\t", index=False
    )

    # 3. Downsampling stability
    print("\n[4/5] Testing downsampling stability...")
    downsampling = downsampling_stability_qc(
        count_df,
        treatment_ids,
        control_ids,
        sgrna_col,
        gene_col,
        fractions=[0.5, 0.75, 1.0],
        n_bootstrap=10,
        top_n=100,
    )
    print(f"  Mean stability: {downsampling['mean_stability']:.3f}")

    # Save
    downsampling["stability_by_fraction"].to_csv(
        output_dir / "downsampling_stability.tsv", sep="\t", index=False
    )

    # 4. Positive control enrichment (optional)
    pos_control_result = None
    if positive_control_genes is not None:
        print("\n[5/5] Testing positive control enrichment...")
        pos_control_result = positive_control_enrichment(
            count_df,
            treatment_ids,
            control_ids,
            positive_control_genes,
            sgrna_col,
            gene_col,
            top_n=100,
        )
        print(
            f"  {pos_control_result['n_in_top_n']}/{pos_control_result['n_positive_controls']} positive controls in top-100"
        )
        print(
            f"  Enrichment p-value: {pos_control_result['enrichment_pvalue']:.3e}"
        )

        # Save
        with open(output_dir / "positive_control_enrichment.json", "w") as f:
            import json

            json.dump(pos_control_result, f, indent=2, default=str)
    else:
        print(
            "\n[5/5] No positive controls provided, skipping enrichment test."
        )

    # Decision logic
    print("\n" + "=" * 60)
    print("Recommendation Analysis")
    print("=" * 60)

    reasons = []
    score_components = []

    # 1. Replicate consistency (weight: 40%)
    consistency = rep_consistency["consistency_score"]
    if consistency >= 0.7:
        reasons.append(
            f"✓ High replicate consistency ({consistency:.3f}) → paired analysis beneficial"
        )
        score_components.append(("consistency", 0.8, 0.4))
    elif consistency >= 0.5:
        reasons.append(
            f"○ Moderate replicate consistency ({consistency:.3f}) → paired may help"
        )
        score_components.append(("consistency", 0.5, 0.4))
    else:
        reasons.append(
            f"✗ Low replicate consistency ({consistency:.3f}) → paired unlikely to help"
        )
        score_components.append(("consistency", 0.2, 0.4))

    # 2. Paired vs unpaired difference (weight: 30%)
    rank_diff = 1 - paired_vs_unpaired["rank_correlation"]
    overlap_diff = 1 - paired_vs_unpaired["top_100_overlap"]

    if overlap_diff > 0.3 or rank_diff > 0.3:
        reasons.append(
            f"✓ Substantial difference between paired/unpaired (overlap diff: {overlap_diff:.3f}) → paired gives different results"
        )
        score_components.append(("difference", 0.8, 0.3))
    elif overlap_diff > 0.15 or rank_diff > 0.15:
        reasons.append(
            f"○ Moderate difference between paired/unpaired → paired somewhat different"
        )
        score_components.append(("difference", 0.5, 0.3))
    else:
        reasons.append(
            f"✗ Little difference between paired/unpaired → both methods similar"
        )
        score_components.append(("difference", 0.2, 0.3))

    # 3. Downsampling stability (weight: 20%)
    stability = downsampling["mean_stability"]
    if stability >= 0.7:
        reasons.append(
            f"✓ High stability under downsampling ({stability:.3f}) → robust results"
        )
        score_components.append(("stability", 0.8, 0.2))
    elif stability >= 0.5:
        reasons.append(
            f"○ Moderate stability under downsampling ({stability:.3f})"
        )
        score_components.append(("stability", 0.5, 0.2))
    else:
        reasons.append(
            f"✗ Low stability under downsampling ({stability:.3f}) → results unstable"
        )
        score_components.append(("stability", 0.2, 0.2))

    # 4. Positive controls (weight: 10% if available)
    if pos_control_result is not None:
        frac_in_top = pos_control_result["fraction_in_top_n"]
        pval = pos_control_result["enrichment_pvalue"]

        if pval < 0.01 and frac_in_top > 0.3:
            reasons.append(
                f"✓ Positive controls significantly enriched in top hits (p={pval:.3e})"
            )
            score_components.append(("pos_controls", 0.8, 0.1))
        elif pval < 0.05:
            reasons.append(
                f"○ Positive controls moderately enriched (p={pval:.3e})"
            )
            score_components.append(("pos_controls", 0.5, 0.1))
        else:
            reasons.append(f"✗ Positive controls not significantly enriched")
            score_components.append(("pos_controls", 0.2, 0.1))

    # Calculate weighted score
    total_weight = sum(w for _, _, w in score_components)
    weighted_score = sum(s * w for _, s, w in score_components) / total_weight

    # Recommendation
    if weighted_score >= 0.6:
        recommendation = "paired"
        reasons.append(
            f"\n→ RECOMMENDATION: Use PAIRED analysis (confidence: {weighted_score:.2f})"
        )
    elif weighted_score >= 0.4:
        recommendation = "both"
        reasons.append(
            f"\n→ RECOMMENDATION: Try BOTH methods and compare (confidence: mixed)"
        )
    else:
        recommendation = "unpaired"
        reasons.append(
            f"\n→ RECOMMENDATION: Use UNPAIRED analysis (confidence: {1-weighted_score:.2f})"
        )

    print("\n".join(reasons))
    print("=" * 60)

    # Save recommendation
    recommendation_dict = {
        "recommendation": recommendation,
        "recommendation_score": weighted_score,
        "recommendation_reasons": reasons,
        "score_components": [
            {"component": name, "score": s, "weight": w}
            for name, s, w in score_components
        ],
    }

    import json

    with open(output_dir / "pairing_recommendation.json", "w") as f:
        json.dump(recommendation_dict, f, indent=2)

    # Generate markdown report
    _generate_pairing_qc_report(
        output_dir,
        rep_consistency,
        paired_vs_unpaired,
        downsampling,
        pos_control_result,
        recommendation_dict,
    )

    return {
        "replicate_consistency": rep_consistency,
        "paired_vs_unpaired": paired_vs_unpaired,
        "downsampling_stability": downsampling,
        "positive_control_enrichment": pos_control_result,
        **recommendation_dict,
    }


def _generate_pairing_qc_report(
    output_dir: Path,
    rep_consistency: Dict,
    paired_vs_unpaired: Dict,
    downsampling: Dict,
    pos_control: Optional[Dict],
    recommendation: Dict,
) -> None:
    """Generate markdown report summarizing pairing QC results."""

    lines = []
    lines.append("# Paired vs. Unpaired RRA Analysis QC Report\n")

    lines.append("## 1. Replicate Consistency\n")
    lines.append(
        f"- **Consistency score**: {rep_consistency['consistency_score']:.3f}"
    )
    lines.append(
        f"- **Mean top-100 overlap**: {rep_consistency['mean_top_n_overlap']:.3f}"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- High consistency (>0.7) suggests paired analysis will leverage replicate structure"
    )
    lines.append(
        "- Low consistency (<0.5) suggests replicates are noisy, unpaired may be safer"
    )
    lines.append("")

    lines.append("## 2. Paired vs. Unpaired Comparison\n")
    lines.append(
        f"- **Rank correlation**: {paired_vs_unpaired['rank_correlation']:.3f}"
    )
    lines.append(
        f"- **Top-100 overlap**: {paired_vs_unpaired['top_100_overlap']:.3f}"
    )
    lines.append(
        f"- **Direction agreement**: {paired_vs_unpaired.get('direction_agreement', 'N/A')}"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- High correlation (>0.9) means both methods give similar results"
    )
    lines.append(
        "- Low overlap (<0.7) means paired analysis finds different hits"
    )
    lines.append("")

    lines.append("## 3. Downsampling Stability\n")
    lines.append(f"- **Mean stability**: {downsampling['mean_stability']:.3f}")
    lines.append("")
    lines.append("Per-fraction stability:")
    for _, row in downsampling["stability_by_fraction"].iterrows():
        lines.append(
            f"- {row['fraction']:.0%} depth: {row['mean']:.3f} ± {row['std']:.3f}"
        )
    lines.append("")

    if pos_control is not None:
        lines.append("## 4. Positive Control Enrichment\n")
        lines.append(
            f"- **Total positive controls**: {pos_control['n_positive_controls']}"
        )
        lines.append(
            f"- **In top-100**: {pos_control['n_in_top_n']} ({pos_control['fraction_in_top_n']:.1%})"
        )
        lines.append(
            f"- **Enrichment p-value**: {pos_control['enrichment_pvalue']:.3e}"
        )
        lines.append("")

    lines.append("## 5. Recommendation\n")
    lines.append(
        f"**{recommendation['recommendation'].upper()}** (confidence: {recommendation['recommendation_score']:.2f})\n"
    )
    lines.append("Reasons:")
    for reason in recommendation["recommendation_reasons"]:
        if reason.startswith("→"):
            continue
        lines.append(f"- {reason}")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by crisprscreens pairing QC pipeline*")

    report_file = output_dir / "pairing_qc_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to: {report_file}")
