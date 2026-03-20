"""
Quality control functions for control sgRNA analysis in CRISPR screens.

This module provides comprehensive QC checks to validate that control sgRNAs
are stable and suitable for normalization across multiple experimental
conditions.
"""

import json
import warnings
import numpy as np
import pandas as pd  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from pandas import DataFrame, Series  # noqa: F401
from .plots import (
    plot_ma_grid,
    plot_library_pca,
    plot_sample_correlations,
)
from scipy.stats import kendalltau, spearmanr


def calculate_size_factors_for_method(
    method: str,
    count_df: DataFrame,
    sample_cols: List[str],
    control_sgrna_txt: Optional[Union[Path, str]] = None,
    sgrna_col: str = "sgRNA",
) -> DataFrame:
    if method == "total":
        sf = compute_size_factors_total(count_df, sample_cols)
    elif method == "median":
        sf = compute_size_factors_median_ratio(count_df, sample_cols)
    elif method == "stable_set":
        # First compute logCPM with simple CPM
        simple_cpm = calculate_cpm(count_df, sample_cols)
        logcpm_simple = np.log2(simple_cpm[sample_cols] + 1)
        stable_guides = select_stable_guides(logcpm_simple, sample_cols)
        print(
            f"  Selected {len(stable_guides)} stable guides ({len(stable_guides)/len(count_df)*100:.1f}%)"  # noqa: E501
        )
        sf = compute_size_factors_stable_set(
            count_df, sample_cols, stable_guides
        )
    elif method == "control":
        if control_sgrna_txt is None:
            raise ValueError(f"Method '{method}' - no controls provided")
        sf = compute_size_factors_control(
            count_df, sample_cols, control_ids, sgrna_col
        )
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return sf


def get_samples_to_baselines(
    conditions_dict: Dict[str, List[str]],
    baseline_condition: str,
    delimiter: str,
    paired_replicates: bool = False,
) -> Dict[str, List[str]]:
    """
    get_samples_to_baselines generates mapping of samples.

    Parameters
    ----------
    conditions_dict : Dict[str, List[str]]
        Conditions dictionary mapping condition names to sample lists.
    baseline_condition : str
        Name of the baseline condition.
    delimiter : str
        Delimiter for parsing condition and replicate from sample name.
    paired_replicates : bool
        Whether replicates are paired.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of sample names to their corresponding baseline replicate columns.
    """
    if paired_replicates:
        return get_paired_sample_to_baselines(
            conditions_dict, baseline_condition, baseline_cols, delimiter
        )
    else:
        return get_samples_to_baselines_unpaired(
            conditions_dict, baseline_condition
        )


def get_samples_to_baselines_unpaired(
    conditions_dict: Dict[str, List[str]],
    baseline_condition: str,
) -> Dict[str, List[str]]:
    """
    get_samples_to_baselines generates mapping of samples to their baseline replicates.

    Parameters
    ----------
    conditions_dict : Dict[str, List[str]]
        Conditions dictionary mapping condition names to sample lists.
    baseline_condition: str
        Name of the baseline condition.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of sample names to their corresponding baseline replicate columns.
    """
    samples_to_baselines = {}
    baseline_cols = conditions_dict[baseline_condition]
    for condition, samples in conditions_dict.items():
        if baseline_condition == condition:
            continue
        for sample in samples:
            samples_to_baselines[sample] = baseline_cols
    return samples_to_baselines


def get_paired_sample_to_baselines(
    conditions_dict: Dict[str, List[str]],
    baseline_condition: str,
    delimiter: str,
) -> Dict[str, List[str]]:
    """
    get_paired_sample_to_baselines generates mapping of samples to their paired baseline replicates.

    Parameters
    ----------
    conditions_dict : Dict[str, List[str]]
        Conditions dictionary mapping condition names to sample lists.
    baseline_condition : str
        Name of the baseline condition.
    delimiter : str
        Delimiter for parsing condition and replicate from sample name.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of sample names to their corresponding baseline replicate columns.
    """
    samples_to_baselines = {}
    baseline_cols = conditions_dict[baseline_condition]
    for condition, samples in conditions_dict.items():
        if baseline_condition == condition:
            continue
        for sample in samples:
            replicate = parse_condition_replicate(sample, delimiter)[1]
            # Find matching baseline replicate
            baseline_rep_cols = [
                col
                for col in baseline_cols
                if parse_condition_replicate(col, delimiter)[1] == replicate
            ]
            if len(baseline_rep_cols) > 0:
                samples_to_baselines[sample] = baseline_rep_cols
    return samples_to_baselines


def calculate_size_factors(
    count_df: DataFrame,
    sample_cols: List[str],
    norm_methods: List[str],
    control_sgrna_txt: Optional[Union[Path, str]] = None,
    sgrna_col: str = "sgRNA",
) -> DataFrame:
    size_factors_dict = {}

    for method in norm_methods:
        if control_sgrna_txt is None:
            warnings.warn(f"Skipping '{method}' - no controls provided")
            continue
        elif method not in ["total", "median", "stable_set", "control"]:
            warnings.warn(f"Unknown normalization method: {method}")
            continue
        sf = calculate_size_factors_for_method(
            method,
            count_df,
            sample_cols,
            control_sgrna_txt,
            sgrna_col,
        )
        size_factors_dict[method] = sf
        print(
            f"  {method}: median SF = {sf.median():.3f}, range = [{sf.min():.3f}, {sf.max():.3f}]"  # noqa: E501
        )
    return pd.DataFrame(size_factors_dict)


def calculate_logCPM_for_all_method(
    method: str,
    count_df: DataFrame,
    size_factors_df: DataFrame,
    sample_cols: List[str],
    pseudocount: float = 1.0,
) -> Dict[str, DataFrame]:
    """
    calculate_logCPM calculates logCPM values for all normalization methods.

    Parameters
    ----------
    method : str
        The normalization method to use.
    count_df : DataFrame
        Count table with sgRNA and count columns.
    size_factors_df : DataFrame
        DataFrame of size factors for all methods.
    sample_cols : List[str]
        The sample columns to calculate logCPM for.
    pseudocount : float, optional
        Pseudocount to add before log transformation, by default 1.0.

    Returns
    -------
    Dict[str, DataFrame]
        Dictionary mapping method name to logCPM DataFrame.
    """
    logcpm_dict = {}
    for method in size_factors_df.columns:
        logcpm, norm_cpm = calculate_logCPM(
            method, count_df, size_factors_df[method], sample_cols, pseudocount
        )
        logcpm_dict[method] = (logcpm, norm_cpm)
    return logcpm_dict


def calculate_logCPM(
    method: str,
    count_df: DataFrame,
    size_factors: Series,
    sample_cols: List[str],
    pseudocount: float = 1.0,
) -> DataFrame:
    """
    Calculate logCPM for a given normalization method.

    Parameters
    ----------
    method : str
        Normalization method name.
    count_df : DataFrame
        Count table with sgRNA and count columns.
    size_factors : pd.Series
        Size factors for samples.
    sample_cols : list
        Sample column names.
    baseline_cols : list
        Baseline/reference column names.
    pseudocount : float
        Pseudocount to add before log transformation.

    Returns
    -------
    DataFrame
        DataFrame with logCPM values for sample columns.
    """
    print(f"  Analyzing '{method}' normalization...")
    # Apply normalization
    norm_counts = apply_size_factors(count_df, sample_cols, size_factors)
    # Compute CPM on normalized counts
    norm_cpm = calculate_cpm(norm_counts, sample_cols)
    logcpm = np.log2(norm_cpm[sample_cols] + pseudocount)
    return logcpm, norm_cpm


def calculate_paired_logfcs(
    logcpm: pd.DataFrame,
    samples_to_baselines: Dict[str, str],
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """
    Calculates paired logFCs between logCPM columns of sort samples and their baselines.

    Parameters
    ----------
    logcpm : pd.DataFrame
        DataFrame containing log2CPM values.
        Rows = sgRNAs (or genes), columns = samples.
    samples_to_baselines : Dict[str, str]
        Mapping {sample: baseline_sample}, e.g.
        {
            "SortedA1": "Total1",
            "SortedA2": "Total2",
            "SortedB1": "Total1",
        }
    pseudocount : float, optional
        Only kept for API clarity/documentation.
        Not used here because logCPM is assumed to be already computed.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per sample:
        logFC_<sample> = logCPM(sample) - logCPM(baseline)
    """
    if not isinstance(logcpm, pd.DataFrame):
        raise TypeError("logcpm must be a pandas DataFrame")

    print(samples_to_baselines)
    all_samples = list(samples_to_baselines.keys()) + list(
        [s for sam in samples_to_baselines.values() for s in sam]
    )
    print(all_samples)
    missing = {s for s in set(all_samples) if s not in logcpm.columns}
    if missing:
        raise ValueError(f"Missing columns in logcpm: {sorted(missing)}")

    logfcs = pd.DataFrame(index=logcpm.index)

    for sample, baseline in samples_to_baselines.items():
        print(sample, baseline)
        print(logcpm[baseline].mean(axis=1))
        logfcs[f"logFC_{sample}"] = logcpm[sample] - logcpm[baseline].mean(
            axis=1
        )
    print(logfcs.head())
    return logfcs


def calculate_norm_cpms_and_ma(
    count_df: DataFrame,
    sample_cols: List[str],
    method: str,
    size_factors: Series,
    pseudocount: float = 1.0,
    paired_replicates: bool = False,
    conditions_dict: Dict[str, List[str]] = {},
    baseline_condition: str = "total",
    delimiter: str = "_",
    sgrna_col: str = "sgRNA",
) -> DataFrame:

    count_df = count_df.copy()
    count_df = count_df.set_index(sgrna_col)
    logcpm, norm_cpm = calculate_logCPM(
        method, count_df, size_factors, sample_cols, pseudocount
    )
    # Compute MA values
    if paired_replicates:
        samples_to_baselines = get_paired_sample_to_baselines(
            conditions_dict, baseline_condition, delimiter
        )
    else:
        samples_to_baselines = get_samples_to_baselines_unpaired(
            conditions_dict, baseline_condition
        )
    ma_df = calculate_ma(samples_to_baselines, logcpm)
    df_expanded = count_df.copy()
    df_expanded = df_expanded.join(logcpm, how="left", rsuffix="_logcpm")
    df_expanded = df_expanded.join(ma_df, how="left")
    if paired_replicates:
        logfcs = calculate_paired_logfcs(
            logcpm, samples_to_baselines, pseudocount
        )
        df_expanded = df_expanded.join(logfcs, how="left")
    df_expanded = df_expanded.reset_index()
    return df_expanded


def calculate_ma(
    samples_to_baselines: Dict[str, List[str]],
    logcpm_df: DataFrame,
) -> DataFrame:
    """
    calculate_ma calculates M and A values for all non-baseline samples.

    Parameters
    ----------
    conditions_dict : Dict[str, List[str]]
        _description_
    baseline_cols : List[str]
        _description_
    logcpm_df : DataFrame
        _description_

    Returns
    -------
    DataFrame
        _description_
    """
    to_df = {}
    for sample_col in samples_to_baselines.keys():
        baseline_cols = samples_to_baselines[sample_col]
        baseline_mean = logcpm_df[baseline_cols].mean(axis=1)
        M_val = logcpm_df[sample_col] - baseline_mean
        A_val = 0.5 * (logcpm_df[sample_col] + baseline_mean)
        to_df[f"{sample_col}-M"] = M_val
        to_df[f"{sample_col}-A"] = A_val
    ma_df = pd.DataFrame(to_df, index=logcpm_df.index)
    return ma_df


def generate_standard_qc_report(
    count_tsv: Union[Path, str],
    output_dir: Union[Path, str],
    metadata_tsv: Optional[Union[Path, str]] = None,
    control_sgrna_txt: Optional[Union[Path, str]] = None,
    baseline_condition: str = "total",
    samples_to_select: Optional[List[str]] = None,
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    delimiter: str = "_",
    prefix: str = "",
    norm_methods: Optional[List[str]] = None,
    pseudocount: float = 1.0,
    paired_replicates: bool = False,
    save_formats: Optional[List[str]] = None,
) -> Dict:
    """
    Generate comprehensive standard QC report for CRISPR screens.

    Analyzes full library with multiple normalization methods, compares them,
    and recommends best normalization and analysis method (RRA vs MLE).

    Parameters
    ----------
    count_tsv : Path or str
        Path to MAGeCK count table TSV.
    output_dir : Path or str
        Output directory for all QC files.
    metadata_tsv : Path or str, optional
        Path to metadata file with sample/condition/replicate columns.
        If None, will parse from column names using delimiter.
    control_sgrna_txt : Path or str, optional
        Path to control sgRNA file. If provided, will run control neutrality QC
        and optionally include control-based normalization.
    baseline_condition : str
        Name of baseline condition (e.g., "total", "T0").
    sgrna_col : str
        sgRNA column name.
    gene_col : str
        Gene column name.
    delimiter : str
        Delimiter for parsing condition_replicate from column names.
    norm_methods : list of str, optional
        Normalization methods to test. Default: ["median", "total",
        "stable_set"].
        If controls provided and good: adds "control".
    pseudocount : float
        Pseudocount for log transformation.
    paired_replicates : bool
        Whether replicates are paired.
    save_formats : list of str, optional
        Plot formats to save. Default: ["png"].
    samples_to_select : list of str, optional
        If provided, only analyze these samples from count table.

    Returns
    -------
    dict
        Comprehensive QC results with keys:
        - library_stats: DataFrame
        - qc_by_norm: dict mapping norm_method -> QC results
        - size_factors: dict mapping norm_method -> Series
        - best_normalization: dict with recommendation
        - analysis_recommendation: dict with RRA vs MLE recommendation
        - control_qc: dict (if controls provided)
        - files: dict mapping output_name -> file path
    """

    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    outfiles = {
        "library_stats": output_dir / f"{prefix}_library_stats.tsv",
        "size_factors": output_dir / f"{prefix}_size_factors.tsv",
        "size_factors_comparison": output_dir
        / f"{prefix}_size_factors_comparison.tsv",
        "normalization_recommendation": output_dir
        / f"{prefix}_normalization_recommendation.json",
        "analysis_recommendation": output_dir
        / f"{prefix}_analysis_recommendation.json",
        "qc_summary": output_dir / f"{prefix}_qc_summary.md",
    }

    # Add control QC file if controls provided
    if control_sgrna_txt is not None:
        outfiles["control_neutrality_qc"] = (
            output_dir / f"{prefix}_control_neutrality_qc.json"
        )

    # Add per-normalization method output directories
    for method in norm_methods:
        method_dir = output_dir / f"{prefix}_norm_{method}"
        method_dir.mkdir(exist_ok=True)
        outfiles[f"logfc_distribution_{method}"] = (
            method_dir / f"{prefix}_logfc_distribution.tsv"
        )
        outfiles[f"replicate_consistency_{method}"] = (
            method_dir / f"{prefix}_replicate_consistency.tsv"
        )
        for fmt in save_formats:
            outfiles[f"{method}_ma_plots.{fmt}"] = (
                method_dir / f"{prefix}_ma_plots.{fmt}"
            )

    # Add summary plots
    for fmt in save_formats:
        outfiles[f"pca_full_library.{fmt}"] = (
            output_dir / f"{prefix}_pca_full_library.{fmt}"
        )
        outfiles[f"sample_correlations.{fmt}"] = (
            output_dir / f"{prefix}_sample_correlations.{fmt}"
        )

    if save_formats is None:
        save_formats = ["png"]

    if norm_methods is None:
        norm_methods = ["median", "total", "stable_set"]

    print("=" * 60)
    print("Standard CRISPR Screen QC Pipeline")
    print("=" * 60)

    saved_files = {}

    # Step 1: Load data
    print("\n[1/9] Loading count data...")
    count_df, sample_cols = read_counts(count_tsv, sgrna_col, gene_col)
    if samples_to_select is not None:
        sample_cols = [s for s in sample_cols if s in samples_to_select]
        count_df = count_df[[sgrna_col, gene_col] + sample_cols]
    print(f"  Loaded {len(count_df)} sgRNAs, {len(sample_cols)} samples")

    # Step 2: Parse/load metadata
    print("\n[2/9] Parsing metadata...")
    if metadata_tsv is not None:
        metadata_df = read_metadata(metadata_tsv)
        # Build conditions dict from metadata
        conditions_dict = {}
        for _, row in metadata_df.iterrows():
            condition = row["condition"]
            sample = row["sample"]
            if condition not in conditions_dict:
                conditions_dict[condition] = []
            conditions_dict[condition].append(sample)
    else:
        conditions_dict, metadata_df = parse_metadata_from_columns(
            sample_cols, delimiter
        )

    print(
        f"  Found {len(conditions_dict)} conditions: {list(conditions_dict.keys())}"  # noqa: E501
    )

    # Validate baseline
    if baseline_condition not in conditions_dict:
        raise ValueError(
            f"Baseline condition '{baseline_condition}' not found. "
            f"Available: {list(conditions_dict.keys())}"
        )
    baseline_cols = conditions_dict[baseline_condition]

    # Step 3: Library stats
    print("\n[3/9] Computing library statistics...")
    lib_stats = compute_library_stats(count_df, sample_cols)
    lib_stats_file = outfiles["library_stats"]
    lib_stats.to_csv(lib_stats_file, sep="\t", index=False)
    saved_files["library_stats"] = lib_stats_file
    print(f"  Saved to {lib_stats_file}")

    # Step 4: Load controls if provided
    controls_data = None
    if control_sgrna_txt is not None:
        print("\n[4/9] Loading control sgRNAs and testing neutrality...")
        control_ids = load_control_sgrnas(control_sgrna_txt)
        control_ids = [
            cid for cid in control_ids if cid in set(count_df[sgrna_col])
        ]
        print(f"  Loaded {len(control_ids)} control sgRNAs")

        # Test control neutrality
        controls_data = qc_controls_neutrality(
            count_df,
            sample_cols,
            control_ids,
            conditions_dict,
            baseline_cols,
            sgrna_col,
            pseudocount,
            baseline_condition=baseline_condition,
        )

        if controls_data["controls_good"]:
            print("  ✓ Controls are neutral - suitable for normalization")
            if "control" not in norm_methods:
                norm_methods.append("control")
        else:
            print("  ✗ Controls show systematic biases:")
            for reason in controls_data["reasons"]:
                print(f"    - {reason}")
            print("  → Control-based normalization NOT recommended")

        # Save control QC
        control_qc_file = outfiles["control_neutrality_qc"]
        with open(control_qc_file, "w") as f:
            # Convert to JSON-serializable format
            json_data = {
                "controls_good": controls_data["controls_good"],
                "metrics": controls_data["metrics"],
                "overall_median_corr": (
                    float(controls_data["overall_median_corr"])
                    if not np.isnan(controls_data["overall_median_corr"])
                    else None
                ),
                "reasons": controls_data["reasons"],
            }
            json.dump(json_data, f, indent=2)
        saved_files["control_neutrality"] = control_qc_file
    else:
        print("\n[4/9] No controls provided - skipping control QC")

    # Step 5: Compute size factors for all methods
    print(
        f"\n[5/9] Computing size factors for normalization methods: {norm_methods}"  # noqa: E501
    )
    # size_factors_dict = {}

    # for method in norm_methods:
    #     if method == "total":
    #         sf = compute_size_factors_total(count_df, sample_cols)
    #     elif method == "median":
    #         sf = compute_size_factors_median_ratio(count_df, sample_cols)
    #     elif method == "stable_set":
    #         # First compute logCPM with simple CPM
    #         simple_cpm = calculate_cpm(count_df, sample_cols)
    #         logcpm_simple = np.log2(simple_cpm[sample_cols] + 1)
    #         stable_guides = select_stable_guides(logcpm_simple, sample_cols)
    #         print(
    #             f"  Selected {len(stable_guides)} stable guides ({len(stable_guides)/len(count_df)*100:.1f}%)"  # noqa: E501
    #         )
    #         sf = compute_size_factors_stable_set(
    #             count_df, sample_cols, stable_guides
    #         )
    #     elif method == "control":
    #         if control_sgrna_txt is None:
    #             warnings.warn(f"Skipping '{method}' - no controls provided")
    #             continue
    #         sf = compute_size_factors_control(
    #             count_df, sample_cols, control_ids, sgrna_col
    #         )
    #     else:
    #         warnings.warn(f"Unknown normalization method: {method}")
    #         continue

    #     size_factors_dict[method] = sf
    #     print(
    #         f"  {method}: median SF = {sf.median():.3f}, range = [{sf.min():.3f}, {sf.max():.3f}]"  # noqa: E501
    #     )

    # Save size factors
    # sf_df = pd.DataFrame(size_factors_dict)
    size_factors_df = calculate_size_factors(
        count_df,
        sample_cols,
        norm_methods,
        control_sgrna_txt,
        sgrna_col,
    )
    sf_file = outfiles["size_factors"]
    size_factors_df.to_csv(sf_file, sep="\t")
    saved_files["size_factors"] = sf_file

    # Compare size factors
    sf_comparison = compare_size_factors(size_factors_df)
    sf_comp_file = outfiles["size_factors_comparison"]
    sf_comparison.to_csv(sf_comp_file, sep="\t", index=False)
    saved_files["size_factors_comparison"] = sf_comp_file

    # Step 6: Run QC for each normalization method
    print("\n[6/9] Running QC analysis for each normalization method...")
    qc_by_norm = {}

    logcpm_dict = calculate_logCPM_for_all_method(
        method, count_df, size_factors_df, sample_cols, pseudocount
    )
    for method in size_factors_df.keys():
        print(f"  Analyzing '{method}' normalization...")

        logcpm, norm_cpm = logcpm_dict[method]

        # Compute delta (non-baseline vs baseline)
        non_baseline_cols = [c for c in sample_cols if c not in baseline_cols]
        delta_df = calculate_delta_logfc(
            norm_cpm, non_baseline_cols, baseline_cols, pseudocount
        )

        # QC metrics
        logfc_dist = qc_logfc_distribution(
            delta_df, conditions_dict, baseline_condition=baseline_condition
        )
        rep_consistency = qc_replicate_consistency(
            delta_df, conditions_dict, baseline_condition=baseline_condition
        )

        # compute MA paired
        samples_to_baselines = get_samples_to_baselines_unpaired(
            conditions_dict, baseline_condition
        )
        ma_df = calculate_ma(samples_to_baselines, logcpm)
        ma_df_paired = None
        if paired_replicates:
            samples_to_baselines = get_paired_sample_to_baselines(
                conditions_dict, baseline_condition, delimiter
            )
            ma_df_paired = calculate_ma(samples_to_baselines, logcpm)

        qc_by_norm[method] = {
            "logfc_dist": logfc_dist,
            "replicate_consistency": rep_consistency,
            "logcpm": logcpm,
            "delta": delta_df,
            "ma": ma_df,
            "ma_paired": ma_df_paired,
        }
        print(logfc_dist.head())
        # Save per-method metrics
        method_dir = output_dir / f"norm_{method}"
        method_dir.mkdir(exist_ok=True)

        logfc_dist.to_csv(
            outfiles[f"logfc_distribution_{method}"], sep="\t", index=False
        )
        rep_consistency["summary"].to_csv(
            outfiles[f"replicate_consistency_{method}"], sep="\t", index=False
        )

        # Generate MA plots for this normalization
        print("    Generating MA plots...")

        fig_ma, ma_metrics = plot_ma_grid(ma_df)
        for fmt in save_formats:
            ma_file = outfiles[f"{method}_ma_plots.{fmt}"]
            fig_ma.savefig(ma_file, dpi=300, bbox_inches="tight")
        plt.close(fig_ma)

        if paired_replicates and ma_df_paired is not None:
            print("    Generating paired MA plots...")

            fig_ma_paired, ma_metrics = plot_ma_grid(ma_df_paired)
            for fmt in save_formats:
                ma_file_paired = outfiles[
                    f"{method}_ma_plots.{fmt}"
                ].with_suffix(f".paired.{fmt}")
                fig_ma_paired.savefig(
                    ma_file_paired, dpi=300, bbox_inches="tight"
                )

            plt.close(fig_ma_paired)

    # End of per-normalization method loop
    # Step 7: Choose best normalization
    print("\n[7/9] Selecting best normalization method...")
    norm_recommendation = choose_best_normalization(qc_by_norm, size_factors_df)
    best_method = norm_recommendation["best_method"]
    print(f"  ✓ Recommended: {best_method}")
    print("  Reasons:")
    for reason in norm_recommendation["reasons"]:
        print(f"    - {reason}")

    # Save recommendation
    norm_rec_file = outfiles["normalization_recommendation"]
    with open(norm_rec_file, "w") as f:
        json.dump(
            {
                "best_method": norm_recommendation["best_method"],
                "ranking": [
                    {"method": m, "score": float(s)}
                    for m, s in norm_recommendation["ranking"]
                ],
                "reasons": norm_recommendation["reasons"],
            },
            f,
            indent=2,
        )
    saved_files["normalization_recommendation"] = norm_rec_file

    # Step 8: Recommend analysis method (RRA vs MLE)
    print("\n[8/9] Recommending analysis method (RRA vs MLE)...")
    analysis_rec = recommend_analysis_method(
        metadata_df, qc_by_norm[best_method], baseline_condition
    )
    print(f"  ✓ Recommended: {analysis_rec['preferred_method']}")
    print("  Reasons:")
    for reason in analysis_rec["reasons"]:
        print(f"    - {reason}")

    # Save recommendation
    analysis_rec_file = outfiles["analysis_recommendation"]
    with open(analysis_rec_file, "w") as f:
        json.dump(
            {
                "preferred_method": analysis_rec["preferred_method"],
                "reasons": analysis_rec["reasons"],
                "replicate_quality_score": float(
                    analysis_rec["replicate_quality_score"]
                ),
                "normalization_quality_score": float(
                    analysis_rec["normalization_quality_score"]
                ),
                "replicate_correlation": (
                    float(analysis_rec["replicate_correlation"])
                    if not np.isnan(analysis_rec["replicate_correlation"])
                    else None
                ),
                "experimental_design": analysis_rec["experimental_design"],
            },
            f,
            indent=2,
        )
    saved_files["analysis_recommendation"] = analysis_rec_file

    # Step 9: Generate summary plots for best normalization
    print(
        f"\n[9/9] Generating summary plots (using '{best_method}' normalization)..."  # noqa: E501
    )

    best_qc = qc_by_norm[best_method]

    # PCA
    print("  - PCA analysis...")
    fig_pca, _ = plot_library_pca(
        best_qc["logcpm"],
        sample_cols,
        conditions_dict,
        color_by="condition",
        title="Full Library PCA",
    )
    for fmt in save_formats:
        pca_file = outfiles[f"pca_full_library.{fmt}"]
        fig_pca.savefig(pca_file, dpi=300, bbox_inches="tight")
        saved_files[f"pca_{fmt}"] = pca_file
    plt.close(fig_pca)

    # Sample correlations
    print("  - Sample correlation heatmap...")
    fig_corr = plot_sample_correlations(
        best_qc["logcpm"], sample_cols, conditions_dict
    )
    for fmt in save_formats:
        corr_file = outfiles[f"sample_correlations.{fmt}"]
        fig_corr.savefig(corr_file, dpi=300, bbox_inches="tight")
        saved_files[f"correlations_{fmt}"] = corr_file
    plt.close(fig_corr)

    # Generate summary markdown report
    print("\n  Writing summary report...")
    summary_md = _generate_summary_markdown(
        lib_stats,
        norm_recommendation,
        analysis_rec,
        controls_data,
        qc_by_norm[best_method],
        best_method,
    )

    summary_file = outfiles["qc_summary"]
    with open(summary_file, "w") as f:
        f.write(summary_md)
    saved_files["summary_md"] = summary_file

    print("\n" + "=" * 60)
    print("QC Pipeline Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return {
        "library_stats": lib_stats,
        "metadata": metadata_df,
        "conditions": conditions_dict,
        "size_factors": size_factors_df,
        "qc_by_norm": qc_by_norm,
        "best_normalization": norm_recommendation,
        "analysis_recommendation": analysis_rec,
        "control_qc": controls_data,
        "files": saved_files,
    }


def load_control_sgrnas(control_file: Union[Path, str]) -> set:
    """
    Load control sgRNA IDs from a text file.

    Parameters
    ----------
    control_file : Path or str
        Path to file containing one control sgRNA ID per line.

    Returns
    -------
    set
        Set of control sgRNA IDs.
    """
    control_file = Path(control_file)
    with open(control_file) as f:
        controls = {line.strip() for line in f if line.strip()}
    return controls


def read_counts(
    count_tsv: Union[Path, str],
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
) -> Tuple[DataFrame, List[str]]:
    """
    Load MAGeCK count table and identify sample columns.

    Parameters
    ----------
    count_tsv : Path or str
        Path to MAGeCK count TSV file.
    sgrna_col : str
        Name of sgRNA ID column.
    gene_col : str
        Name of gene column.

    Returns
    -------
    tuple
        (count_df, sample_cols) where:
        - count_df: DataFrame with all columns
        - sample_cols: List of sample column names (excluding sgRNA/Gene)
    """
    count_tsv = Path(count_tsv)
    if not count_tsv.exists():
        raise FileNotFoundError(f"Count file not found: {count_tsv}")

    count_df = pd.read_csv(count_tsv, sep="\t")
    # Validate required columns
    if sgrna_col not in count_df.columns:
        raise ValueError(f"sgRNA column '{sgrna_col}' not found in count table")
    if gene_col not in count_df.columns:
        raise ValueError(f"Gene column '{gene_col}' not found in count table")

    # Identify sample columns
    sample_cols = [
        col for col in count_df.columns if col not in [sgrna_col, gene_col]
    ]

    if len(sample_cols) == 0:
        raise ValueError("No sample columns found in count table")

    # Validate counts are numeric and non-negative
    for col in sample_cols:
        if not pd.api.types.is_numeric_dtype(count_df[col]):
            raise ValueError(
                f"Sample column '{col}' contains non-numeric values"
            )
        if (count_df[col] < 0).any():
            raise ValueError(f"Sample column '{col}' contains negative counts")

    return count_df, sample_cols


def parse_condition_replicate(
    column_name: str, delimiter: str = "_"
) -> Tuple[str, str]:
    """
    Parse condition and replicate from column name.

    Parameters
    ----------
    column_name : str
        Column name like "Total_Rep1" or "Sort1_Rep2".
    delimiter : str
        Delimiter between condition and replicate.

    Returns
    -------
    tuple
        (condition, replicate) e.g. ("Total", "Rep1")
    """
    parts = column_name.rsplit(delimiter, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return column_name, "Rep1"


def parse_metadata_from_columns(
    sample_cols: List[str], delimiter: str = "_"
) -> Tuple[Dict[str, List[str]], DataFrame]:
    """
    Parse metadata from sample column names.

    Uses parse_condition_replicate to extract condition and replicate info.

    Parameters
    ----------
    sample_cols : list
        List of sample column names.
    delimiter : str
        Delimiter between condition and replicate.

    Returns
    -------
    tuple
        (conditions_dict, metadata_df) where:
        - conditions_dict: Dict mapping condition -> list of column names
        - metadata_df: DataFrame with columns [sample, condition, replicate]
    """
    metadata_records = []
    conditions = {}

    for col in sample_cols:
        condition, replicate = parse_condition_replicate(col, delimiter)
        metadata_records.append(
            {
                "sample": col,
                "condition": condition,
                "replicate": replicate,
            }
        )

        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(col)

    metadata_df = pd.DataFrame(metadata_records)
    return conditions, metadata_df


def read_metadata(metadata_tsv: Union[Path, str]) -> DataFrame:
    """
    Load and validate metadata TSV file.

    Parameters
    ----------
    metadata_tsv : Path or str
        Path to metadata file with columns: sample, condition, replicate.

    Returns
    -------
    DataFrame
        Validated metadata DataFrame.
    """
    metadata_tsv = Path(metadata_tsv)
    if not metadata_tsv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_tsv}")

    # Auto-detect separator
    if metadata_tsv.suffix == ".csv":
        metadata_df = pd.read_csv(metadata_tsv)
    else:
        metadata_df = pd.read_csv(metadata_tsv, sep="\t")

    # Validate required columns
    required_cols = ["sample", "condition", "replicate"]
    missing_cols = [
        col for col in required_cols if col not in metadata_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Metadata file missing required columns: {missing_cols}. "
            f"Found: {list(metadata_df.columns)}"
        )

    return metadata_df


def calculate_cpm(count_df: DataFrame, sample_cols: List[str]) -> DataFrame:
    """
    Calculate counts per million (CPM) from raw counts.

    Parameters
    ----------
    count_df : DataFrame
        Count table with sgRNA and count columns.
    sample_cols : list
        List of sample column names to normalize.

    Returns
    -------
    DataFrame
        DataFrame with CPM values for sample columns.
    """
    cpm_df = count_df.copy()
    for col in sample_cols:
        total_counts = count_df[col].sum()
        cpm_df[col] = (count_df[col] / total_counts) * 1e6
    return cpm_df


def calculate_delta_logfc(
    cpm_df: DataFrame,
    sample_cols: List[str],
    baseline_cols: List[str],
    pseudocount: float = 1.0,
) -> DataFrame:
    """
    Calculate Δc = log2(CPM_condition + pc) - log2(CPM_baseline + pc).

    Parameters
    ----------
    cpm_df : DataFrame
        CPM values for all samples.
    sample_cols : list
        Sample columns to compare to baseline.
    baseline_cols : list
        Baseline/reference columns.
    pseudocount : float
        Pseudocount to add before log transformation.

    Returns
    -------
    DataFrame
        DataFrame with delta log2 fold-change values.
    """
    delta_df = pd.DataFrame(index=cpm_df.index)

    # Average baseline if multiple replicates
    baseline_mean = cpm_df[baseline_cols].mean(axis=1)

    for col in sample_cols:
        delta_df[col] = np.log2(cpm_df[col] + pseudocount) - np.log2(
            baseline_mean + pseudocount
        )

    return delta_df


def compute_library_stats(
    count_df: DataFrame, sample_cols: List[str]
) -> DataFrame:
    """
    Compute per-sample library statistics.

    Parameters
    ----------
    count_df : DataFrame
        Count table with sgRNA counts.
    sample_cols : list
        List of sample column names.

    Returns
    -------
    DataFrame
        Statistics table with columns:
        - library_size: Total counts per sample
        - n_sgrnas: Total number of sgRNAs
        - n_zeros: Number of zero-count sgRNAs
        - zero_fraction: Fraction of zero-count sgRNAs
        - top1pct_fraction: Fraction of counts in top 1% sgRNAs
    """
    stats_records = []

    for col in sample_cols:
        counts = count_df[col]
        n_sgrnas = len(counts)
        library_size = counts.sum()
        n_zeros = (counts == 0).sum()
        zero_fraction = n_zeros / n_sgrnas

        # Top 1% concentration
        n_top = max(1, int(n_sgrnas * 0.01))
        top_counts = counts.nlargest(n_top).sum()
        top1pct_fraction = top_counts / library_size if library_size > 0 else 0

        stats_records.append(
            {
                "sample": col,
                "library_size": library_size,
                "n_sgrnas": n_sgrnas,
                "n_zeros": n_zeros,
                "zero_fraction": zero_fraction,
                "top1pct_fraction": top1pct_fraction,
            }
        )

    return pd.DataFrame(stats_records)


def compute_size_factors_total(
    count_df: DataFrame, sample_cols: List[str]
) -> pd.Series:
    """
    Compute size factors using total count normalization.

    sf = library_size / median(library_size)

    Parameters
    ----------
    count_df : DataFrame
        Count table.
    sample_cols : list
        Sample column names.

    Returns
    -------
    pd.Series
        Size factors indexed by sample name.
    """
    lib_sizes = count_df[sample_cols].sum(axis=0)
    median_lib_size = lib_sizes.median()
    size_factors = lib_sizes / median_lib_size
    return size_factors


def compute_size_factors_median_ratio(
    count_df: DataFrame, sample_cols: List[str], use_nonzero: bool = True
) -> pd.Series:
    """
    Compute size factors using median-of-ratios (DESeq2-style).

    For each sgRNA: compute geometric mean across samples.
    For each sample: compute ratios to geometric mean, take median.

    Parameters
    ----------
    count_df : DataFrame
        Count table.
    sample_cols : list
        Sample column names.
    use_nonzero : bool
        If True, compute geometric mean only over non-zero counts.

    Returns
    -------
    pd.Series
        Size factors indexed by sample name.
    """
    counts = count_df[sample_cols].values

    # Compute geometric mean per sgRNA
    if use_nonzero:
        # Replace zeros with NaN for geometric mean calculation
        counts_for_gm = np.where(counts > 0, counts, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_counts = np.log(counts_for_gm)
            geometric_means = np.exp(np.nanmean(log_counts, axis=1))
    else:
        # Standard geometric mean (will be 0 if any sample is 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            geometric_means = np.exp(np.mean(np.log(counts + 1e-10), axis=1))

    # Compute ratios and take median per sample
    size_factors = []
    for i, col in enumerate(sample_cols):
        sample_counts = counts[:, i]
        ratios = sample_counts / geometric_means
        # Filter out invalid ratios
        valid_ratios = ratios[(geometric_means > 0) & (sample_counts > 0)]
        if len(valid_ratios) > 0:
            sf = np.median(valid_ratios)
        else:
            sf = 1.0
        size_factors.append(sf)

    size_factors = pd.Series(size_factors, index=sample_cols)

    # Normalize to median = 1
    size_factors = size_factors / size_factors.median()

    return size_factors


def select_stable_guides(
    logcpm_df: DataFrame, sample_cols: List[str], quantile: float = 0.3
) -> pd.Index:
    """
    Select most stable sgRNAs based on low variance across samples.

    Parameters
    ----------
    logcpm_df : DataFrame
        Log2(CPM+1) values.
    sample_cols : list
        Sample column names.
    quantile : float
        Quantile of lowest-variance guides to select (default 0.3 = bottom 30%).

    Returns
    -------
    pd.Index
        Index of stable sgRNAs.
    """
    # Compute variance per sgRNA across samples
    variances = logcpm_df[sample_cols].var(axis=1)

    # Select lowest-variance quantile
    threshold = variances.quantile(quantile)
    stable_mask = variances <= threshold

    return logcpm_df[stable_mask].index


def compute_size_factors_stable_set(
    count_df: DataFrame, sample_cols: List[str], stable_guides: pd.Index
) -> pd.Series:
    """
    Compute size factors using stable guide set (median-of-ratios on subset).

    Parameters
    ----------
    count_df : DataFrame
        Full count table.
    sample_cols : list
        Sample column names.
    stable_guides : pd.Index
        Index of stable sgRNAs to use.

    Returns
    -------
    pd.Series
        Size factors indexed by sample name.
    """
    # Subset to stable guides
    stable_counts = count_df.loc[stable_guides, sample_cols]

    # Apply median-of-ratios on subset
    return compute_size_factors_median_ratio(stable_counts, sample_cols)


def compute_size_factors_control(
    count_df: DataFrame,
    sample_cols: List[str],
    control_ids: Union[set, pd.Index],
    sgrna_col: str = "sgRNA",
) -> pd.Series:
    """
    Compute size factors using control sgRNAs only.

    Parameters
    ----------
    count_df : DataFrame
        Full count table.
    sample_cols : list
        Sample column names.
    control_ids : set or Index
        Control sgRNA IDs.
    sgrna_col : str
        sgRNA column name.

    Returns
    -------
    pd.Series
        Size factors indexed by sample name.
    """
    # Subset to control sgRNAs
    control_mask = count_df[sgrna_col].isin(control_ids)
    control_counts = count_df[control_mask]

    if len(control_counts) == 0:
        raise ValueError("No control sgRNAs found in count table")

    # Apply median-of-ratios on controls
    return compute_size_factors_median_ratio(control_counts, sample_cols)


def apply_size_factors(
    count_df: DataFrame, sample_cols: List[str], size_factors: pd.Series
) -> DataFrame:
    """
    Apply size factors to normalize counts.

    norm_counts = counts / size_factor

    Parameters
    ----------
    count_df : DataFrame
        Raw count table.
    sample_cols : list
        Sample column names.
    size_factors : pd.Series
        Size factors per sample.

    Returns
    -------
    DataFrame
        Normalized count table.
    """
    norm_df = count_df.copy()
    for col in sample_cols:
        norm_df[col] = count_df[col] / size_factors.loc[col]
    return norm_df


def qc_logfc_distribution(
    delta_df: DataFrame,
    conditions_dict: Dict[str, List[str]],
    thresholds: Optional[Dict[str, float]] = None,
    baseline_condition: Optional[str] = None,
) -> DataFrame:
    """
    QC check for log-fold-change distributions per condition.

    Parameters
    ----------
    delta_df : DataFrame
        Delta log2 fold-change values (condition vs baseline).
    conditions_dict : dict
        Mapping condition -> list of sample columns.
    thresholds : dict, optional
        Thresholds for QC checks:
        - median_shift_warn: |median| threshold for global shift warning
          (default: 0.3)
        - median_shift_strong: |median| threshold for strong warning
          (default: 0.5)
        - tail_rate_warn: tail_rate threshold for heavy tails (default: 0.1)
    baseline_condition : str, optional
        Name of the baseline condition to skip if present in `conditions_dict`.

    Returns
    -------
    DataFrame
        QC metrics per condition with columns:
        - median, mean, iqr, std, skewness
        - tail_rate_0.5, tail_rate_1.0
        - shift_warning: bool flag for global shift
        - heavy_tails_warning: bool flag for heavy tails
    """
    if thresholds is None:
        thresholds = {
            "median_shift_warn": 0.3,
            "median_shift_strong": 0.5,
            "tail_rate_warn": 0.1,
        }

    qc_records = []

    for condition, sample_cols in conditions_dict.items():
        # Skip baseline condition explicitly if provided
        if baseline_condition is not None and condition == baseline_condition:
            continue

        # Only use sample columns that exist in delta_df (non-baseline samples)
        cols_in_delta = [c for c in sample_cols if c in delta_df.columns]
        if len(cols_in_delta) == 0:
            # No delta values for this condition (likely baseline or absent)
            continue

        # Get delta values for this condition
        cond_deltas = delta_df[cols_in_delta].values.flatten()
        cond_deltas = cond_deltas[~np.isnan(cond_deltas)]

        if len(cond_deltas) == 0:
            continue

        # Compute metrics
        median_val = np.median(cond_deltas)
        mean_val = np.mean(cond_deltas)
        iqr_val = stats.iqr(cond_deltas)
        std_val = np.std(cond_deltas)
        skewness_val = stats.skew(cond_deltas)

        tail_rate_05 = np.mean(np.abs(cond_deltas) > 0.5)
        tail_rate_10 = np.mean(np.abs(cond_deltas) > 1.0)

        # Flags
        abs_median = np.abs(median_val)
        shift_warning = abs_median >= thresholds["median_shift_warn"]
        shift_strong = abs_median >= thresholds["median_shift_strong"]
        heavy_tails = tail_rate_10 >= thresholds["tail_rate_warn"]

        qc_records.append(
            {
                "condition": condition,
                "median": median_val,
                "mean": mean_val,
                "iqr": iqr_val,
                "std": std_val,
                "skewness": skewness_val,
                "tail_rate_0.5": tail_rate_05,
                "tail_rate_1.0": tail_rate_10,
                "shift_warning": shift_warning,
                "shift_strong": shift_strong,
                "heavy_tails_warning": heavy_tails,
                "n_values": len(cond_deltas),
            }
        )

    return pd.DataFrame(qc_records)


def qc_replicate_consistency(
    delta_df: DataFrame,
    conditions_dict: Dict[str, List[str]],
    method: str = "spearman",
    baseline_condition: Optional[str] = None,
) -> Dict:
    """
    QC check for replicate consistency via correlation of delta values.

    Parameters
    ----------
    delta_df : DataFrame
        Delta log2 fold-change values (non-baseline samples as columns).
    conditions_dict : dict
        Mapping condition -> list of sample columns (may include baseline).
    method : str
        Correlation method: 'spearman' or 'pearson'.
    baseline_condition : str, optional
        Name of the baseline condition to explicitly skip when computing
        replicate consistency.

    Returns
    -------
    dict
        Results with:
        - correlations: dict of correlation matrices per condition
        - summary: DataFrame with min/median/mean correlation per condition
    """
    correlations = {}
    summary_records = []

    for condition, sample_cols in conditions_dict.items():
        # If a baseline condition is provided, skip it explicitly
        if baseline_condition is not None and condition == baseline_condition:
            continue

        # Use only columns that exist in delta_df (non-baseline columns)
        cols_in_delta = [c for c in sample_cols if c in delta_df.columns]

        # If there are fewer than two delta columns, we cannot compute replicate correlations  # noqa: E501
        if len(cols_in_delta) < 2:
            summary_records.append(
                {
                    "condition": condition,
                    "n_replicates": len(cols_in_delta),
                    "min_corr": np.nan,
                    "median_corr": np.nan,
                    "mean_corr": np.nan,
                    "quality": "no_replicates",
                }
            )
            continue

        cond_df = delta_df[cols_in_delta]
        if method == "spearman":
            corr_matrix = cond_df.corr(method="spearman")
        else:
            corr_matrix = cond_df.corr(method="pearson")

        correlations[condition] = corr_matrix

        # Extract off-diagonal correlations
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        off_diag = corr_matrix.where(mask).stack().values

        if len(off_diag) > 0:
            min_corr = off_diag.min()
            median_corr = np.median(off_diag)
            mean_corr = off_diag.mean()

            # Quality assessment
            if median_corr >= 0.5:
                quality = "good"
            elif median_corr >= 0.3:
                quality = "ok"
            else:
                quality = "bad"
        else:
            min_corr = median_corr = mean_corr = np.nan
            quality = "unknown"

        summary_records.append(
            {
                "condition": condition,
                "n_replicates": len(cols_in_delta),
                "min_corr": min_corr,
                "median_corr": median_corr,
                "mean_corr": mean_corr,
                "quality": quality,
            }
        )

    summary_df = pd.DataFrame(summary_records)

    return {
        "correlations": correlations,
        "summary": summary_df,
    }


def qc_controls_neutrality(
    count_df: DataFrame,
    sample_cols: List[str],
    control_ids: Union[set, pd.Index],
    conditions_dict: Dict[str, List[str]],
    baseline_cols: List[str],
    sgrna_col: str = "sgRNA",
    pseudocount: float = 1.0,
    thresholds: Optional[Dict[str, float]] = None,
    baseline_condition: Optional[str] = None,
) -> Dict:
    """
    QC check to validate if control sgRNAs are suitable for normalization.

    Controls should be neutral: no systematic shifts, low tail rates, good
    replicate correlation.

    Parameters
    ----------
    count_df : DataFrame
        Full count table.
    sample_cols : list
        Sample column names.
    control_ids : set or Index
        Control sgRNA IDs.
    conditions_dict : dict
        Mapping condition -> list of sample columns.
    baseline_cols : list
        Baseline columns for delta calculation.
    sgrna_col : str
        sgRNA column name.
    pseudocount : float
        Pseudocount for log transformation.
    thresholds : dict, optional
        Thresholds: median_shift_max (0.2), tail_rate_max (0.05), corr_min (0.3)

    Returns
    -------
    dict
        Results with:
        - controls_good: bool, overall assessment
        - metrics: per-condition metrics (median shift, tail rate)
        - replicate_consistency: correlation summary
        - reasons: list of failure reasons if controls_good=False
    """
    if thresholds is None:
        thresholds = {
            "median_shift_max": 0.2,
            "tail_rate_max": 0.05,
            "corr_min": 0.3,
        }

    # Filter to controls
    control_mask = count_df[sgrna_col].isin(control_ids)
    control_df = count_df[control_mask].copy()

    if len(control_df) == 0:
        return {
            "controls_good": False,
            "metrics": {},
            "reasons": ["No control sgRNAs found"],
        }

    # Compute CPM on full library
    full_cpm = calculate_cpm(count_df, sample_cols)
    control_cpm = full_cpm[control_mask].copy()

    # Compute delta for non-baseline conditions
    non_baseline_cols = [col for col in sample_cols if col not in baseline_cols]
    if len(non_baseline_cols) == 0:
        return {
            "controls_good": False,
            "metrics": {},
            "reasons": ["No non-baseline samples"],
        }

    delta_df = calculate_delta_logfc(
        control_cpm, non_baseline_cols, baseline_cols, pseudocount
    )
    # Compute metrics per condition
    metrics = {}
    reasons = []
    for condition, cond_cols in conditions_dict.items():
        # If a baseline condition is provided, skip it explicitly
        if baseline_condition is not None and condition == baseline_condition:
            continue

        # Fallback: if baseline_condition not provided butcolumns match baseline
        if baseline_condition is None and set(cond_cols) == set(baseline_cols):
            continue

        cond_delta = delta_df[
            [c for c in cond_cols if c in delta_df.columns]
        ].values.flatten()
        cond_delta = cond_delta[~np.isnan(cond_delta)]

        if len(cond_delta) == 0:
            continue

        median_shift = np.median(cond_delta)
        tail_rate = np.mean(np.abs(cond_delta) > 1.0)

        metrics[condition] = {
            "median_shift": median_shift,
            "tail_rate_1.0": tail_rate,
            "n_controls": len(control_df),
        }

        # Check thresholds
        if np.abs(median_shift) >= thresholds["median_shift_max"]:
            reasons.append(
                f"{condition}: large median shift ({median_shift:.3f})"
            )

        if tail_rate >= thresholds["tail_rate_max"]:
            reasons.append(f"{condition}: high tail rate ({tail_rate:.3f})")

    # Check replicate consistency
    rep_consistency = qc_replicate_consistency(
        delta_df, conditions_dict, baseline_condition=baseline_condition
    )
    median_corrs = rep_consistency["summary"]["median_corr"].dropna()

    if len(median_corrs) > 0:
        overall_median_corr = median_corrs.median()
        if overall_median_corr < thresholds["corr_min"]:
            reasons.append(
                f"Low replicate correlation ({overall_median_corr:.3f})"
            )
    else:
        overall_median_corr = np.nan

    # Overall decision
    controls_good = len(reasons) == 0

    return {
        "controls_good": controls_good,
        "metrics": metrics,
        "replicate_consistency": rep_consistency,
        "overall_median_corr": overall_median_corr,
        "reasons": reasons,
    }


def compare_size_factors(size_factors: DataFrame) -> DataFrame:
    """
    Compare size factors from different normalization methods.

    Parameters
    ----------
    size_factors : DataFrame
        DataFrame with size factors per method (columns) and sample (index).

    Returns
    -------
    DataFrame
        Comparison metrics: correlation and max fold-difference between methods.
    """
    methods = list(size_factors.columns)
    n_methods = size_factors.shape[1]

    comparison_records = []

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            method1 = methods[i]
            method2 = methods[j]

            sf1 = size_factors[method1]
            sf2 = size_factors[method2]

            # Correlation
            corr = sf1.corr(sf2)

            # Max fold-difference
            fold_diffs = sf1 / sf2
            max_fold_diff = max(fold_diffs.max(), 1 / fold_diffs.min())

            comparison_records.append(
                {
                    "method1": method1,
                    "method2": method2,
                    "correlation": corr,
                    "max_fold_diff": max_fold_diff,
                }
            )

    return pd.DataFrame(comparison_records)


def choose_best_normalization(
    qc_results_by_norm: Dict[str, Dict],
    size_factors_df: DataFrame,
) -> Dict:
    """
    Choose best normalization method based on QC metrics.

    Ranks methods by:
    1. Replicate consistency (highest weight)
    2. Global shift magnitude (lower is better)
    3. Heavy tails frequency (lower is better)
    4. Size factor stability (correlation with median method)

    Parameters
    ----------
    qc_results_by_norm : dict
        Dict mapping norm_method -> qc_results dict with:
        - logfc_dist: DataFrame from qc_logfc_distribution
        - replicate_consistency: dict from qc_replicate_consistency
    size_factors_df : DataFrame
        DataFrame with size factors per method (columns) and sample (index).

    Returns
    -------
    dict
        Results with:
        - best_method: str
        - ranking: list of (method, score) tuples
        - reasons: list of reasons for recommendation
        - scores_detail: dict with score breakdown per method
    """
    scores = {}
    details = {}

    for method, qc_res in qc_results_by_norm.items():
        score = 0.0
        detail = {}

        # 1. Replicate consistency (0-40 points)
        rep_summary = qc_res["replicate_consistency"]["summary"]
        median_corrs = rep_summary["median_corr"].dropna()
        if len(median_corrs) > 0:
            median_corr = median_corrs.median()
            rep_score = median_corr * 40  # 0-40 points
        else:
            median_corr = 0.0
            rep_score = 0.0

        score += rep_score
        detail["replicate_consistency"] = median_corr
        detail["replicate_score"] = rep_score

        # 2. Global shift (0-30 points, inverse)
        logfc_dist = qc_res["logfc_dist"]
        avg_abs_median = logfc_dist["median"].abs().mean()
        shift_penalty = min(avg_abs_median / 0.5, 1.0) * 30  # 0-30 penalty
        shift_score = 30 - shift_penalty

        score += shift_score
        detail["avg_abs_median_shift"] = avg_abs_median
        detail["shift_score"] = shift_score

        # 3. Heavy tails (0-20 points, inverse)
        n_heavy_tails = (logfc_dist["heavy_tails_warning"]).sum()
        n_conditions = len(logfc_dist)
        tail_fraction = n_heavy_tails / n_conditions if n_conditions > 0 else 0
        tail_penalty = tail_fraction * 20
        tail_score = 20 - tail_penalty

        score += tail_score
        detail["heavy_tails_fraction"] = tail_fraction
        detail["tail_score"] = tail_score

        # 4. Size factor stability (0-10 points)
        # Compare to median method if available
        if "median" in size_factors_df.columns and method != "median":
            sf_corr = size_factors_df[method].corr(size_factors_df["median"])
            sf_score = sf_corr * 10 if not np.isnan(sf_corr) else 5
        else:
            sf_score = 10  # Full points if this IS median or no comparison

        score += sf_score
        detail["size_factor_stability"] = sf_score

        scores[method] = score
        details[method] = detail

    # Rank methods
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_method = ranking[0][0]

    # Generate reasons
    reasons = []
    best_detail = details[best_method]

    if best_detail["replicate_consistency"] >= 0.5:
        reasons.append(
            f"Good replicate consistency ({best_detail['replicate_consistency']:.2f})"  # noqa: E501
        )
    elif best_detail["replicate_consistency"] >= 0.3:
        reasons.append(
            f"Acceptable replicate consistency ({best_detail['replicate_consistency']:.2f})"  # noqa: E501
        )
    else:
        reasons.append(
            f"WARNING: Low replicate consistency ({best_detail['replicate_consistency']:.2f})"  # noqa: E501
        )

    if best_detail["avg_abs_median_shift"] < 0.3:
        reasons.append(
            f"Low global shift ({best_detail['avg_abs_median_shift']:.2f})"
        )
    else:
        reasons.append(
            f"Moderate global shift ({best_detail['avg_abs_median_shift']:.2f})"
        )

    if best_detail["heavy_tails_fraction"] < 0.3:
        reasons.append("Few heavy-tailed distributions")
    else:
        reasons.append(
            f"WARNING: Many heavy-tailed distributions ({best_detail['heavy_tails_fraction']:.1%})"  # noqa: E501
        )

    return {
        "best_method": best_method,
        "ranking": ranking,
        "reasons": reasons,
        "scores_detail": details,
    }


def recommend_analysis_method(
    metadata_df: DataFrame,
    qc_core_results: Dict,
    baseline_condition: str,
) -> Dict:
    """
    Recommend analysis method (RRA vs MLE) based on experimental design and QC.

    Decision logic:
    - If replicate consistency < 0.3 OR PCA shows batch effects: RRA
    - If consistency >= 0.5 AND low global shift AND clean MA plots: MLE
    - If sort-like design with strong global shift: RRA primary
    - Otherwise: RRA+MLE (both methods recommended)

    Parameters
    ----------
    metadata_df : DataFrame
        Metadata with sample/condition/replicate.
    qc_core_results : dict
        QC results with replicate_consistency, logfc_dist, etc.
    baseline_condition : str
        Name of baseline condition.

    Returns
    -------
    dict
        Recommendation with:
        - preferred_method: "RRA", "MLE", or "RRA+MLE"
        - reasons: list of reasoning strings
        - replicate_quality_score: float (0-1)
        - normalization_quality_score: float (0-1)
        - experimental_design: dict with design characteristics
    """
    reasons = []

    # Analyze experimental design
    conditions = metadata_df["condition"].unique()
    n_conditions = len(conditions)
    is_sort_like = any(
        "sort" in c.lower() or "high" in c.lower() or "low" in c.lower()
        for c in conditions
    )

    design_info = {
        "n_conditions": n_conditions,
        "conditions": list(conditions),
        "baseline": baseline_condition,
        "is_sort_like": is_sort_like,
    }

    # Check replicate consistency
    rep_summary = qc_core_results["replicate_consistency"]["summary"]
    median_corrs = rep_summary["median_corr"].dropna()

    if len(median_corrs) > 0:
        overall_rep_corr = median_corrs.median()
    else:
        overall_rep_corr = np.nan
        reasons.append("WARNING: No replicate information available")

    # Replicate quality score
    if not np.isnan(overall_rep_corr):
        if overall_rep_corr >= 0.5:
            replicate_quality_score = 1.0
            rep_quality_text = "excellent"
        elif overall_rep_corr >= 0.3:
            replicate_quality_score = 0.7
            rep_quality_text = "acceptable"
        else:
            replicate_quality_score = 0.3
            rep_quality_text = "poor"
    else:
        replicate_quality_score = 0.5
        rep_quality_text = "unknown"

    # Check global shift
    logfc_dist = qc_core_results["logfc_dist"]
    avg_abs_median = logfc_dist["median"].abs().mean()
    n_shift_warnings = (logfc_dist["shift_warning"]).sum()
    shift_fraction = (
        n_shift_warnings / len(logfc_dist) if len(logfc_dist) > 0 else 0
    )

    # Normalization quality score
    if avg_abs_median < 0.3:
        norm_quality_score = 1.0
        norm_quality_text = "excellent"
    elif avg_abs_median < 0.5:
        norm_quality_score = 0.7
        norm_quality_text = "acceptable"
    else:
        norm_quality_score = 0.3
        norm_quality_text = "poor"

    # Decision logic
    preferred_method = None

    # Case 1: Poor replicate consistency -> RRA
    if replicate_quality_score < 0.5:
        preferred_method = "RRA"
        reasons.append(
            f"Low replicate consistency ({overall_rep_corr:.2f}) favors rank-based RRA"  # noqa: E501
        )

    # Case 2: Sort-like design with strong shift -> RRA
    elif is_sort_like and shift_fraction > 0.5:
        preferred_method = "RRA"
        reasons.append(
            "Sort-like screen with global compositional shifts favors RRA"
        )

    # Case 3: Good replicates + low shift -> MLE
    elif replicate_quality_score >= 0.7 and norm_quality_score >= 0.7:
        preferred_method = "MLE"
        reasons.append(
            f"Good replicate consistency ({overall_rep_corr:.2f}) and low global shifts favor MLE"  # noqa: E501
        )
        reasons.append("MLE provides more power for multi-condition designs")

    # Case 4: Mixed signals -> both methods
    else:
        preferred_method = "RRA+MLE"
        reasons.append(
            f"Replicate consistency: {rep_quality_text} ({overall_rep_corr:.2f})"  # noqa: E501
        )
        reasons.append(
            f"Normalization quality: {norm_quality_text} (avg shift: {avg_abs_median:.2f})"  # noqa: E501
        )
        reasons.append(
            "Recommend running both RRA (robust) and MLE (descriptive) for comparison"  # noqa: E501
        )

    # Additional context
    if n_conditions > 2:
        reasons.append(
            f"Multi-condition design ({n_conditions} conditions) benefits from MLE's joint modeling"  # noqa: E501
        )

    return {
        "preferred_method": preferred_method,
        "reasons": reasons,
        "replicate_quality_score": replicate_quality_score,
        "normalization_quality_score": norm_quality_score,
        "experimental_design": design_info,
        "replicate_correlation": overall_rep_corr,
    }


def control_sgrna_qc(
    count_table: Union[Path, str, DataFrame],
    control_sgrnas: Union[Path, str, set],
    baseline_condition: str,
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    delimiter: str = "_",
    pseudocount: float = 1.0,
) -> Dict:
    """
    Comprehensive QC analysis for control sgRNAs across conditions.

    Computes CPM from raw counts, calculates Δc (log2FC vs baseline) for
    each condition, and returns metrics including median, IQR, tail-rate,
    pairwise comparisons, and replicate correlations.

    Parameters
    ----------
    count_table : Path, str, or DataFrame
        Path to MAGeCK count table or DataFrame.
    control_sgrnas : Path, str, or set
        Path to control sgRNA file or set of IDs.
    baseline_condition : str
        Name of baseline condition (e.g., "Total", "T0").
    sgrna_col : str
        Name of sgRNA ID column.
    gene_col : str
        Name of gene column.
    delimiter : str
        Delimiter for parsing condition_replicate.
    pseudocount : float
        Pseudocount for log transformation.

    Returns
    -------
    dict
        QC metrics including:
        - raw_counts: DataFrame with control sgRNA raw counts
        - cpm: DataFrame with control sgRNA CPM values
        - delta: DataFrame with control sgRNA Δc values
        - conditions: dict mapping condition -> list of column names
        - baseline_cols: list of baseline columns
        - metrics: dict with per-condition metrics (median, IQR, tail_rate)
        - pairwise_median: DataFrame with pairwise median Δc
        - replicate_correlations: dict of correlation matrices per condition
        - wilcoxon_tests: dict of Wilcoxon test results per condition
    """
    # Load data
    if isinstance(count_table, (str, Path)):
        count_df, sample_cols = read_counts(count_table, sgrna_col, gene_col)
    else:
        count_df = count_table.copy()
        sample_cols = [
            col for col in count_df.columns if col not in [sgrna_col, gene_col]
        ]

    if isinstance(control_sgrnas, (str, Path)):
        controls = load_control_sgrnas(control_sgrnas)
    else:
        controls = control_sgrnas

    # Filter for control sgRNAs
    control_mask = count_df[sgrna_col].isin(controls)
    control_df = count_df[control_mask].copy()

    if len(control_df) == 0:
        raise ValueError("No control sgRNAs found in count table!")

    # Parse conditions and replicates using refactored function
    conditions, metadata_df = parse_metadata_from_columns(
        sample_cols, delimiter
    )

    # Identify baseline columns
    if baseline_condition not in conditions:
        raise ValueError(
            f"Baseline condition '{baseline_condition}' not found. "
            f"Available: {list(conditions.keys())}"
        )
    baseline_cols = conditions[baseline_condition]

    # Calculate CPM on FULL library (not just controls)
    # This is critical: CPM must be normalized by total library size
    full_cpm_df = calculate_cpm(count_df, sample_cols)

    # Extract CPM values for controls only
    cpm_df = full_cpm_df[control_mask].copy()

    # Calculate delta for non-baseline conditions
    non_baseline_conditions = {
        k: v for k, v in conditions.items() if k != baseline_condition
    }

    all_delta_cols = []
    for cond, cols in non_baseline_conditions.items():
        all_delta_cols.extend(cols)

    delta_df = calculate_delta_logfc(
        cpm_df, all_delta_cols, baseline_cols, pseudocount
    )

    # Calculate metrics per condition
    metrics = {}
    for cond, cols in non_baseline_conditions.items():
        print("CONDITION", cond)
        cond_delta = delta_df[cols].values.flatten()
        cond_delta = cond_delta[~np.isnan(cond_delta)]

        metrics[cond] = {
            "median": np.median(cond_delta),
            "mean": np.mean(cond_delta),
            "iqr": stats.iqr(cond_delta),
            "std": np.std(cond_delta),
            "tail_rate_1.0": np.mean(np.abs(cond_delta) > 1.0),
            "tail_rate_0.5": np.mean(np.abs(cond_delta) > 0.5),
            "n_sgrnas": len(control_df),
            "n_values": len(cond_delta),
        }
        print(metrics[cond])

        # Wilcoxon signed-rank test: H0: median = 0
        if len(cond_delta) > 0:
            try:
                statistic, pval = stats.wilcoxon(
                    cond_delta, alternative="two-sided"
                )
                metrics[cond]["wilcoxon_statistic"] = statistic
                metrics[cond]["wilcoxon_pvalue"] = pval
            except ValueError:
                metrics[cond]["wilcoxon_statistic"] = np.nan
                metrics[cond]["wilcoxon_pvalue"] = np.nan

    # Pairwise median comparisons
    pairwise_median = pd.DataFrame(
        index=list(non_baseline_conditions.keys()),
        columns=list(non_baseline_conditions.keys()),
    )

    for cond1 in non_baseline_conditions.keys():
        for cond2 in non_baseline_conditions.keys():
            cols1 = conditions[cond1]
            cols2 = conditions[cond2]

            # Mean CPM across replicates for each condition
            cpm1 = cpm_df[cols1].mean(axis=1)
            cpm2 = cpm_df[cols2].mean(axis=1)

            delta_pairwise = np.log2(cpm1 + pseudocount) - np.log2(
                cpm2 + pseudocount
            )
            pairwise_median.loc[cond1, cond2] = np.median(delta_pairwise)

    pairwise_median = pairwise_median.astype(float)

    # Replicate correlations per condition
    replicate_correlations = {}
    for cond, cols in conditions.items():
        if len(cols) > 1:
            # Use log2(CPM+1) for correlation
            log_cpm = np.log2(cpm_df[cols] + 1)
            corr_matrix = log_cpm.corr(method="pearson")
            replicate_correlations[cond] = corr_matrix
    # Check neutrality thresholds
    reasons = []
    for cond, metric in metrics.items():
        if abs(metric["median"]) >= 0.2:
            reasons.append(
                f"{cond}: large median shift ({metric['median']:.3f})"
            )
        if metric["tail_rate_1.0"] >= 0.05:
            reasons.append(
                f"{cond}: high tail rate ({metric['tail_rate_1.0']:.3f})"
            )

    # Check replicate consistency
    rep_corrs = [
        corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        for corr_matrix in replicate_correlations.values()
    ]
    overall_median_corr = np.median(rep_corrs) if rep_corrs else np.nan

    if overall_median_corr < 0.3:
        reasons.append(f"Low replicate correlation ({overall_median_corr:.3f})")

    controls_good = len(reasons) == 0

    return {
        "raw_counts": control_df,
        "cpm": cpm_df,
        "delta": delta_df,
        "conditions": conditions,
        "baseline_cols": baseline_cols,
        "baseline_condition": baseline_condition,
        "metrics": metrics,
        "pairwise_median": pairwise_median,
        "replicate_correlations": replicate_correlations,
        "sample_cols": sample_cols,
        # Neutrality check
        "controls_good": controls_good,
        "overall_median_corr": overall_median_corr,
        "reasons": reasons,
    }


def export_control_counts_and_cpm(
    count_table: Union[Path, str, DataFrame],
    control_sgrnas: Union[Path, str, set],
    output_dir: Union[Path, str],
    prefix: str = "control_data",
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
) -> Dict[str, Path]:
    """
    Export raw counts and CPM values for control sgRNAs only.

    CPM is calculated on the full library (correct normalization),
    then filtered to show only control sgRNAs.

    Parameters
    ----------
    count_table : Path, str, or DataFrame
        MAGeCK count table.
    control_sgrnas : Path, str, or set
        Control sgRNA IDs.
    output_dir : Path or str
        Output directory.
    prefix : str
        Filename prefix.
    sgrna_col : str
        sgRNA column name.
    gene_col : str
        Gene column name.

    Returns
    -------
    dict
        Dictionary with paths to exported files:
        - "raw_counts": Path to raw counts TSV
        - "cpm": Path to CPM TSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    if isinstance(count_table, (str, Path)):
        count_df = pd.read_csv(count_table, sep="\t")
    else:
        count_df = count_table.copy()

    if isinstance(control_sgrnas, (str, Path)):
        controls = load_control_sgrnas(control_sgrnas)
    else:
        controls = control_sgrnas

    # Filter for control sgRNAs
    control_mask = count_df[sgrna_col].isin(controls)
    control_df = count_df[control_mask].copy()

    if len(control_df) == 0:
        raise ValueError("No control sgRNAs found in count table!")

    # Identify sample columns
    sample_cols = [
        col for col in count_df.columns if col not in [sgrna_col, gene_col]
    ]

    # Calculate CPM on FULL library (correct normalization)
    full_cpm_df = calculate_cpm(count_df, sample_cols)

    # Extract CPM for controls only
    cpm_df = full_cpm_df[control_mask].copy()

    # Save raw counts for controls
    raw_counts_file = output_dir / f"{prefix}_raw_counts.tsv"
    control_df.to_csv(raw_counts_file, sep="\t", index=False)
    print(f"Saved control raw counts to {raw_counts_file}")

    # Save CPM for controls
    cpm_file = output_dir / f"{prefix}_cpm.tsv"
    cpm_df.to_csv(cpm_file, sep="\t", index=False)
    print(f"Saved control CPM to {cpm_file}")

    return {
        "raw_counts": raw_counts_file,
        "cpm": cpm_file,
    }


def _generate_summary_markdown(
    lib_stats: DataFrame,
    norm_rec: Dict,
    analysis_rec: Dict,
    control_qc: Optional[Dict],
    best_qc: Dict,
    best_method: str,
) -> str:
    """Generate markdown summary report."""

    md = []
    md.append("# CRISPR Screen QC Summary Report\n")
    md.append("## 1. Library Statistics\n")
    md.append(f"- Total sgRNAs: {lib_stats['n_sgrnas'].iloc[0]}")
    md.append(f"- Samples: {len(lib_stats)}")
    md.append(
        f"- Median library size: {lib_stats['library_size'].median():.0f}"
    )
    md.append(
        f"- Median zero fraction: {lib_stats['zero_fraction'].median():.2%}"
    )
    md.append(
        f"- Median top 1% concentration: {lib_stats['top1pct_fraction'].median():.2%}\n"  # noqa: E501
    )

    md.append("## 2. Normalization Recommendation\n")
    md.append(f"**Recommended method: {norm_rec['best_method']}**\n")
    md.append("Reasons:")
    for reason in norm_rec["reasons"]:
        md.append(f"- {reason}")
    md.append("")

    md.append("### Normalization ranking:")
    for method, score in norm_rec["ranking"]:
        md.append(f"- {method}: {score:.1f} points")
    md.append("")

    md.append("## 3. Analysis Method Recommendation\n")
    md.append(f"**Recommended: {analysis_rec['preferred_method']}**\n")
    md.append("Reasons:")
    for reason in analysis_rec["reasons"]:
        md.append(f"- {reason}")
    md.append("")

    md.append("### Quality Scores:")
    md.append(
        f"- Replicate consistency: {analysis_rec['replicate_quality_score']:.2f}"  # noqa: E501
    )
    md.append(
        f"- Normalization quality: {analysis_rec['normalization_quality_score']:.2f}"  # noqa: E501
    )
    md.append("")

    if control_qc is not None:
        md.append("## 4. Control sgRNA QC\n")
        if control_qc["controls_good"]:
            md.append(
                "✓ **Controls are neutral** - suitable for normalization\n"
            )
        else:
            md.append(
                "✗ **Controls show biases** - not recommended for normalization\n"  # noqa: E501
            )
            md.append("Issues:")
            for reason in control_qc["reasons"]:
                md.append(f"- {reason}")
            md.append("")

    md.append(
        f"## 5. Replicate Consistency (using {best_method} normalization)\n"
    )
    rep_summary = best_qc["replicate_consistency"]["summary"]
    for _, row in rep_summary.iterrows():
        if not pd.isna(row["median_corr"]):
            md.append(
                f"- {row['condition']}: median corr = {row['median_corr']:.3f} ({row['quality']})"  # noqa: E501
            )
    md.append("")

    md.append(
        f"## 6. Log-Fold-Change Distribution (using {best_method} normalization)\n"  # noqa: E501
    )
    logfc_dist = best_qc["logfc_dist"]
    for _, row in logfc_dist.iterrows():
        flags = []
        if row["shift_warning"]:
            flags.append("⚠ global shift")
        if row["heavy_tails_warning"]:
            flags.append("⚠ heavy tails")
        flag_str = f" {', '.join(flags)}" if flags else ""
        md.append(
            f"- {row['condition']}: median Δc = {row['median']:.3f}, "
            f"tail rate = {row['tail_rate_1.0']:.2%}{flag_str}"
        )
    md.append("")

    md.append("---\n")
    md.append("*Generated by crisprscreens standard QC pipeline*\n")

    return "\n".join(md)


# def generate_control_qc_report(
#     count_table: Union[Path, str, DataFrame],
#     control_sgrnas: Union[Path, str, set],
#     baseline_condition: str,
#     output_dir: Union[Path, str],
#     prefix: str = "control_qc",
#     sgrna_col: str = "sgRNA",
#     gene_col: str = "Gene",
#     delimiter: str = "_",
#     save_formats: List[str] = ["png", "pdf"],
# ) -> Dict:
#     """
#     Generate comprehensive control sgRNA QC report.

#     Creates all QC plots and saves metrics to file.

#     Parameters
#     ----------
#     count_table : Path, str, or DataFrame
#         MAGeCK count table.
#     control_sgrnas : Path, str, or set
#         Control sgRNA IDs.
#     baseline_condition : str
#         Baseline condition name.
#     output_dir : Path or str
#         Output directory.
#     prefix : str
#         Filename prefix.
#     sgrna_col : str
#         sgRNA column name.
#     gene_col : str
#         Gene column name.
#     delimiter : str
#         Condition/replicate delimiter.
#     save_formats : list
#         List of formats to save ("png", "pdf", "svg").

#     Returns
#     -------
#     dict
#         QC results with file paths.
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True, parents=True)

#     # Run QC analysis
#     print("Running control sgRNA QC analysis...")
#     qc_results = control_sgrna_qc(
#         count_table=count_table,
#         control_sgrnas=control_sgrnas,
#         baseline_condition=baseline_condition,
#         sgrna_col=sgrna_col,
#         gene_col=gene_col,
#         delimiter=delimiter,
#     )

#     # Save metrics
#     metrics_file = output_dir / f"{prefix}_metrics.tsv"
#     metrics_df = pd.DataFrame(qc_results["metrics"]).T
#     metrics_df.to_csv(metrics_file, sep="\t")
#     print(f"Saved metrics to {metrics_file}")

#     # Save pairwise median shifts
#     pairwise_file = output_dir / f"{prefix}_pairwise_shifts.tsv"
#     qc_results["pairwise_median"].to_csv(pairwise_file, sep="\t")
#     print(f"Saved pairwise shifts to {pairwise_file}")
#     cpm_files = export_control_counts_and_cpm(
#         count_table=qc_results["raw_counts"],
#         control_sgrnas=control_sgrnas,
#         output_dir=output_dir,
#         prefix=f"{prefix}",
#         sgrna_col=sgrna_col,
#         gene_col=gene_col,
#     )
#     saved_files = {
#         "metrics": metrics_file,
#         "pairwise_shifts": pairwise_file,
#     }
#     saved_files.update(cpm_files)

#     # Generate and save plots
#     plots_to_generate = [
#         ("distribution", plot_control_distribution_per_condition),
#         ("pairwise_heatmap", plot_pairwise_control_shifts),
#         ("replicate_correlation", plot_control_replicate_correlation),
#         (
#             "pca_condition",
#             lambda qc: plot_control_pca(qc, color_by="condition"),
#         ),
#         (
#             "pca_replicate",
#             lambda qc: plot_control_pca(qc, color_by="replicate"),
#         ),
#     ]

#     for plot_name, plot_func in plots_to_generate:
#         print(f"Generating {plot_name} plot...")
#         try:
#             fig_result = plot_func(qc_results)
#             if isinstance(fig_result, tuple):
#                 fig = fig_result[0]
#             else:
#                 fig = fig_result

#             for fmt in save_formats:
#                 outfile = output_dir / f"{prefix}_{plot_name}.{fmt}"
#                 fig.savefig(outfile, dpi=300, bbox_inches="tight")
#                 saved_files[f"{plot_name}_{fmt}"] = outfile

#             plt.close(fig)
#             print(f"  Saved {plot_name}")
#         except Exception as e:
#             print(f"  Warning: Failed to generate {plot_name}: {e}")

#     print(f"\nControl QC report complete. Files saved to {output_dir}")

#     return {
#         "qc_results": qc_results,
#         "files": saved_files,
#     }


def _dcg_from_ranks(ref_ranks: np.ndarray, test_ranks: np.ndarray) -> float:
    """
    Simple DCG-style similarity:
    - relevance = 1 / ref_rank
    - order by test_rank
    - normalized by ideal DCG
    """
    if len(ref_ranks) == 0:
        return np.nan

    rel = 1.0 / ref_ranks
    order_test = np.argsort(test_ranks)
    order_ideal = np.argsort(ref_ranks)

    def dcg(order):
        rel_o = rel[order]
        discounts = 1.0 / np.log2(np.arange(2, len(rel_o) + 2))
        return np.sum(rel_o * discounts)

    ideal = dcg(order_ideal)
    if ideal == 0:
        return np.nan

    return dcg(order_test) / ideal


def calculate_ranking_metrics(
    gene_ranking_files_dict: Dict[str, Union[Path, str]],
    gene_id_columns: Dict[str, str],
    ranking_columns: Dict[str, str],
    ascending: Dict[str, bool],
) -> pd.DataFrame:
    """
    Calculate pairwise ranking similarity metrics between multiple gene rankings.

    Returns a long-format DataFrame with:
    ranking_a, ranking_b, n_genes, kendall_tau, spearman_r, dcg
    """
    # load all rankings
    rankings = {}
    for name, path in gene_ranking_files_dict.items():
        df = pd.read_csv(path, sep=None, engine="python")
        df = df[[gene_id_columns[name], ranking_columns[name]]].copy()
        df = df.sort_values(by=ranking_columns[name], ascending=ascending[name])
        df.columns = ["gene_id", "sort_value"]
        df["rank"] = np.arange(1, len(df) + 1)
        df["gene_id"] = df["gene_id"].astype(str)
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna().drop_duplicates("gene_id")
        rankings[name] = df
    results = []

    for a, b in itertools.combinations(rankings.keys(), 2):
        df = rankings[a].merge(
            rankings[b],
            on="gene_id",
            suffixes=("_a", "_b"),
            how="inner",
        )

        ra = df["rank_a"].to_numpy()
        rb = df["rank_b"].to_numpy()

        results.append(
            {
                "ranking_a": a,
                "ranking_b": b,
                "n_genes": len(df),
                "kendall_tau": kendalltau(ra, rb).correlation,
                "spearman_r": spearmanr(ra, rb).correlation,
                "dcg": np.nanmean(
                    [
                        _dcg_from_ranks(ra, rb),
                        _dcg_from_ranks(rb, ra),
                    ]
                ),
            }
        )

    return pd.DataFrame(results)
