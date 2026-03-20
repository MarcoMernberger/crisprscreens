"""
Core functions for comparing MAGeCK RRA and MLE methods.

This module provides comprehensive analysis tools to compare different
MAGeCK analysis methods (RRA vs MLE) through:
1. Leave-one-replicate-out analysis
2. sgRNA coherence analysis
3. Control sgRNA false-positive checks
4. Permutation tests

These analyses help determine which method is more robust for a given dataset.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Sequence, Tuple
from scipy.stats import spearmanr, kendalltau
from itertools import combinations
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt


###############################################################################
# 1. Leave-One-Replicate-Out Analysis
###############################################################################


def compute_overlap(set1: set, set2: set) -> float:
    """Compute Jaccard index between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    union = set1.union(set2)
    if len(union) == 0:
        return 0.0
    intersection = set1.intersection(set2)
    return len(intersection) / len(union)


def get_top_n_genes(
    gene_summary: Union[Path, str, pd.DataFrame],
    n: int = 100,
    score_col: Optional[str] = None,
    fdr_col: Optional[str] = None,
    gene_col: str = "id",
    direction: str = "both",
) -> set:
    """
    Extract top N genes from MAGeCK gene summary.

    Parameters
    ----------
    gene_summary : Path, str, or DataFrame
        Path to gene summary file or DataFrame
    n : int
        Number of top genes to extract
    score_col : str, optional
        Column name for score (e.g., 'neg|score' for RRA, 'beta' for MLE)
    fdr_col : str, optional
        Column name for FDR
    gene_col : str
        Column name for gene identifiers
    direction : str
        'neg', 'pos', or 'both'

    Returns
    -------
    set
        Set of top N gene names
    """
    if isinstance(gene_summary, (str, Path)):
        df = pd.read_csv(gene_summary, sep="\t")
    else:
        df = gene_summary.copy()

    top_genes = set()

    # Auto-detect columns if not provided
    if score_col is None and fdr_col is None:
        # Try to detect method type
        if "neg|score" in df.columns:  # RRA
            score_col = "neg|score"
            fdr_col = "neg|fdr"
        elif "pos|score" in df.columns:
            score_col = "pos|score"
            fdr_col = "pos|fdr"
        elif any("beta" in col for col in df.columns):  # MLE
            beta_cols = [col for col in df.columns if "beta" in col.lower()]
            if beta_cols:
                score_col = beta_cols[0]
                fdr_cols = [col for col in df.columns if "fdr" in col.lower()]
                if fdr_cols:
                    fdr_col = fdr_cols[0]

    if direction in ["neg", "both"]:
        df_sorted = df.sort_values(by=fdr_col if fdr_col else score_col)
        top_genes.update(df_sorted.head(n)[gene_col].tolist())

    if direction in ["pos", "both"]:
        if "pos|score" in df.columns:
            df_sorted = df.sort_values(by="pos|fdr")
            top_genes.update(df_sorted.head(n)[gene_col].tolist())

    return top_genes


def compute_rank_correlation(
    gene_summary1: Union[Path, str, pd.DataFrame],
    gene_summary2: Union[Path, str, pd.DataFrame],
    score_col: Optional[str] = None,
    gene_col: str = "id",
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation between two gene summaries.

    Returns
    -------
    Tuple[float, float]
        Spearman correlation coefficient and p-value
    """
    if isinstance(gene_summary1, (str, Path)):
        df1 = pd.read_csv(gene_summary1, sep="\t")
    else:
        df1 = gene_summary1.copy()

    if isinstance(gene_summary2, (str, Path)):
        df2 = pd.read_csv(gene_summary2, sep="\t")
    else:
        df2 = gene_summary2.copy()

    # Auto-detect score column if not provided
    if score_col is None:
        if "neg|score" in df1.columns:
            score_col = "neg|score"
        elif any("beta" in col for col in df1.columns):
            score_col = [col for col in df1.columns if "beta" in col.lower()][0]

    # Merge on gene
    merged = df1[[gene_col, score_col]].merge(
        df2[[gene_col, score_col]], on=gene_col, suffixes=("_1", "_2")
    )

    if len(merged) < 3:
        return np.nan, np.nan

    corr, pval = spearmanr(
        merged[f"{score_col}_1"], merged[f"{score_col}_2"], nan_policy="omit"
    )

    return corr, pval


def run_mageck_function(
    mageck_func,
    run_dir: Union[Path, str],
    count_table: Union[Path, str],
    ctrl_subset: List[str],
    treat_subset: List[str],
    design_matrix: Optional[Union[Path, str, None]],
    run_name: str,
    **method_params,
):
    if mageck_func.__name__ == "mageck_test":
        mageck_func(
            count_table=count_table,
            treatment_ids=treat_subset,
            control_ids=ctrl_subset,
            out_dir=run_dir,
            prefix=run_name,
            **method_params,
        )
    elif mageck_func.__name__ == "mageck_mle":
        mageck_func(
            count_table=count_table,
            design_matrix=design_matrix,
            out_dir=run_dir,
            prefix=run_name,
            **method_params,
        )
    else:
        raise ValueError(f"Unsupported MAGeCK function: {mageck_func.__name__}")


def leave_one_replicate_out_analysis(
    count_table: Union[Path, str],
    output_dir: Union[Path, str],
    control_ids: List[str],
    treatment_ids: List[str],
    full_design_matrix: Optional[Union[Path, str, None]],
    prefix: str,
    run_mageck_func: Callable,
    method_params: Dict = {},
    top_n_list: List[int] = [50, 100, 200],
    gene_col: str = "id",
) -> pd.DataFrame:
    """
    Perform leave-one-replicate-out consistency analysis.

    Parameters
    ----------
    count_table : Path or str
        Path to MAGeCK count table
    control_ids : List[str]
        List of control sample IDs
    treatment_ids : List[str]
        List of treatment sample IDs
    output_dir : Path or str
        Directory to save results
    prefix : str
        Prefix for output files
    run_mageck_func : Callable
        Function to run MAGeCK analysis
        Should accept: count_table, control_ids, treatment_ids, out_dir, prefix, **method_params
    method_params : Dict
        Additional parameters for MAGeCK run
    top_n_list : List[int]
        List of top-N values to test
    gene_col : str
        Gene identifier column name

    Returns
    -------
    pd.DataFrame
        Summary statistics for each leave-one-out run
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_replicates = min(len(control_ids), len(treatment_ids))
    if n_replicates < 2:
        raise ValueError(
            "Need at least 2 replicates for leave-one-out analysis"
        )

    results = []

    # Generate all leave-one-out combinations
    for i in range(n_replicates):
        # Leave out replicate i
        ctrl_subset = [c for j, c in enumerate(control_ids) if j != i]
        treat_subset = [t for j, t in enumerate(treatment_ids) if j != i]

        design_matrix = None
        with NamedTemporaryFile(mode="w+", delete=False) as tmp_design:
            if full_design_matrix is not None:
                # dump a temporal design_matrix
                design_df = pd.read_csv(full_design_matrix, sep="\t")
                # Assuming design matrix has a 'Sample' column matching control/treatment IDs
                subset_samples = ctrl_subset + treat_subset
                subset_design = design_df[
                    design_df["Samples"].isin(subset_samples)
                ]
                subset_design = subset_design.loc[:, (subset_design != 0).any()]
                subset_design.to_csv(tmp_design.name, sep="\t", index=False)
                design_matrix = tmp_design.name

        run_name = f"{prefix}_leave_out_rep{i+1}"
        run_dir = output_dir / f"leave_out_rep{i+1}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Run MAGeCK
        run_mageck_function(
            run_mageck_func,
            run_dir=run_dir,
            count_table=count_table,
            ctrl_subset=ctrl_subset,
            treat_subset=treat_subset,
            design_matrix=design_matrix,
            run_name=run_name,
            **method_params,
        )
        if design_matrix is not None:
            Path(design_matrix).unlink()  # Clean up temp file
        gene_summary = run_dir / f"{run_name}.gene_summary.tsv"
        if not gene_summary.exists():
            # Try alternative naming
            gene_summary = run_dir / f"{run_name}.gene_summary.txt"

        if gene_summary.exists():
            results.append(
                {
                    "run_name": run_name,
                    "left_out_replicate": i + 1,
                    "n_controls": len(ctrl_subset),
                    "n_treatments": len(treat_subset),
                    "gene_summary_path": str(gene_summary),
                }
            )
        else:
            print(f"Warning: Gene summary not found for {run_name}")

    if len(results) < 2:
        raise ValueError("Not enough successful runs for comparison")

    # Compute pairwise metrics
    comparison_results = []
    for (i, r1), (j, r2) in combinations(enumerate(results), 2):
        gs1 = r1["gene_summary_path"]
        gs2 = r2["gene_summary_path"]

        # Rank correlation
        corr, pval = compute_rank_correlation(gs1, gs2, gene_col=gene_col)

        comp = {
            "run1": r1["run_name"],
            "run2": r2["run_name"],
            "spearman_correlation": corr,
            "spearman_pvalue": pval,
        }

        # Top-N overlap for each N
        for n in top_n_list:
            top1 = get_top_n_genes(gs1, n=n, gene_col=gene_col)
            top2 = get_top_n_genes(gs2, n=n, gene_col=gene_col)

            overlap = len(top1.intersection(top2))
            jaccard = compute_overlap(top1, top2)

            comp[f"top_{n}_overlap"] = overlap
            comp[f"top_{n}_jaccard"] = jaccard

        comparison_results.append(comp)

    comparison_df = pd.DataFrame(comparison_results)

    # Save results
    comparison_df.to_csv(
        output_dir / f"{prefix}_leave_one_out_comparison.tsv",
        sep="\t",
        index=False,
    )

    # Compute summary statistics
    summary = {
        "method": prefix,
        "n_replicates": n_replicates,
        "mean_spearman": comparison_df["spearman_correlation"].mean(),
        "std_spearman": comparison_df["spearman_correlation"].std(),
        "median_spearman": comparison_df["spearman_correlation"].median(),
    }

    for n in top_n_list:
        summary[f"mean_jaccard_top_{n}"] = comparison_df[
            f"top_{n}_jaccard"
        ].mean()
        summary[f"std_jaccard_top_{n}"] = comparison_df[
            f"top_{n}_jaccard"
        ].std()
        summary[f"mean_overlap_top_{n}"] = comparison_df[
            f"top_{n}_overlap"
        ].mean()

    return pd.DataFrame([summary])


###############################################################################
# 2. sgRNA Coherence Analysis
###############################################################################


def analyze_sgrna_coherence(
    gene_summary: Union[Path, str, pd.DataFrame],
    sgrna_summary: Union[Path, str, pd.DataFrame],
    top_n: int = 100,
    gene_col: str = "Gene",
    sgrna_gene_col: str = "Gene",
    sgrna_col: str = "sgrna",
    lfc_col: Optional[str] = None,
    fdr_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Analyze sgRNA coherence for top-ranked genes.

    Parameters
    ----------
    gene_summary : Path, str, or DataFrame
        MAGeCK gene summary
    sgrna_summary : Path, str, or DataFrame
        MAGeCK sgRNA summary
    top_n : int
        Number of top genes to analyze
    gene_col : str
        Gene column name in gene summary
    sgrna_gene_col : str
        Gene column name in sgRNA summary
    sgrna_col : str
        sgRNA identifier column
    lfc_col : str, optional
        Log-fold-change column in sgRNA summary
    fdr_col : str, optional
        FDR column in sgRNA summary

    Returns
    -------
    pd.DataFrame
        Coherence metrics for each top gene
    """
    if isinstance(gene_summary, (str, Path)):
        gene_df = pd.read_csv(gene_summary, sep="\t")
    else:
        gene_df = gene_summary.copy()

    if isinstance(sgrna_summary, (str, Path)):
        sgrna_df = pd.read_csv(sgrna_summary, sep="\t")
    else:
        sgrna_df = sgrna_summary.copy()

    # Auto-detect columns if not provided
    if lfc_col is None:
        lfc_candidates = [
            col
            for col in sgrna_df.columns
            if "lfc" in col.lower() or "beta" in col.lower()
        ]
        if lfc_candidates:
            lfc_col = lfc_candidates[0]

    if fdr_col is None:
        fdr_candidates = [
            col for col in sgrna_df.columns if "fdr" in col.lower()
        ]
        if fdr_candidates:
            fdr_col = fdr_candidates[0]

    # Get top genes
    top_genes = get_top_n_genes(
        gene_df, n=top_n, gene_col=gene_col, direction="both"
    )

    # Analyze each gene
    coherence_results = []

    for gene in top_genes:
        gene_sgrnas = sgrna_df[sgrna_df[sgrna_gene_col] == gene]

        if len(gene_sgrnas) == 0:
            continue

        n_sgrnas = len(gene_sgrnas)

        # LFC consistency
        if lfc_col and lfc_col in gene_sgrnas.columns:
            lfcs = gene_sgrnas[lfc_col].dropna()

            if len(lfcs) > 0:
                # Direction consistency: what fraction of sgRNAs agree in sign?
                positive = (lfcs > 0).sum()
                negative = (lfcs < 0).sum()
                direction_consistency = max(positive, negative) / len(lfcs)

                # Magnitude consistency
                lfc_mean = lfcs.mean()
                lfc_std = lfcs.std()
                lfc_cv = lfc_std / abs(lfc_mean) if lfc_mean != 0 else np.nan

                # Significant sgRNAs
                if fdr_col and fdr_col in gene_sgrnas.columns:
                    significant_sgrnas = (gene_sgrnas[fdr_col] < 0.05).sum()
                else:
                    significant_sgrnas = np.nan

                coherence_results.append(
                    {
                        "gene": gene,
                        "n_sgrnas": n_sgrnas,
                        "mean_lfc": lfc_mean,
                        "std_lfc": lfc_std,
                        "cv_lfc": lfc_cv,
                        "direction_consistency": direction_consistency,
                        "n_positive": positive,
                        "n_negative": negative,
                        "n_significant": significant_sgrnas,
                        "fraction_significant": (
                            significant_sgrnas / n_sgrnas
                            if not np.isnan(significant_sgrnas)
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(coherence_results)


###############################################################################
# 3. Control sgRNA False-Positive Analysis
###############################################################################


def analyze_control_false_positives(
    gene_summary: Union[Path, str, pd.DataFrame],
    control_sgrnas: Union[Path, str, List[str]],
    top_n_list: List[int] = [50, 100, 200, 500],
    gene_col: str = "id",
    control_prefix: str = "Non-Targeting",
) -> pd.DataFrame:
    """
    Check how many control sgRNAs appear in top-N hits.

    Parameters
    ----------
    gene_summary : Path, str, or DataFrame
        MAGeCK gene summary
    control_sgrnas : Path, str, or List[str]
        Control sgRNA identifiers or file
    top_n_list : List[int]
        List of top-N values to check
    gene_col : str
        Gene identifier column name
    control_prefix : str
        Prefix used for control genes

    Returns
    -------
    pd.DataFrame
        Count of controls in top-N for each N
    """
    if isinstance(gene_summary, (str, Path)):
        df = pd.read_csv(gene_summary, sep="\t")
    else:
        df = gene_summary.copy()

    if isinstance(control_sgrnas, (str, Path)):
        with open(control_sgrnas, "r") as f:
            control_genes = set([line.strip() for line in f])
    elif isinstance(control_sgrnas, list):
        control_genes = set(control_sgrnas)
    else:
        control_genes = set()

    # Also try to detect controls by prefix in gene names
    if gene_col in df.columns:
        potential_controls = df[
            df[gene_col].str.contains(control_prefix, case=False, na=False)
        ][gene_col].tolist()
        control_genes.update(potential_controls)

    results = []

    for n in top_n_list:
        top_genes = get_top_n_genes(
            df, n=n, gene_col=gene_col, direction="both"
        )

        control_in_top = top_genes.intersection(control_genes)
        n_controls = len(control_in_top)
        fraction = n_controls / n if n > 0 else 0

        results.append(
            {
                "top_n": n,
                "n_controls": n_controls,
                "fraction_controls": fraction,
                "control_genes_in_top": (
                    ",".join(sorted(control_in_top)) if control_in_top else ""
                ),
            }
        )

    return pd.DataFrame(results)


###############################################################################
# 4. Permutation Test
###############################################################################


def create_permuted_count_table(
    count_table: Union[Path, str],
    output_file: Union[Path, str],
    sample_columns: List[str],
    permutation_type: str = "within_sample",
    gene_col: str = "Gene",
    sgrna_col: str = "sgRNA",
) -> None:
    """
    Create permuted version of count table for negative control.

    Parameters
    ----------
    count_table : Path or str
        Original count table
    output_file : Path or str
        Output path for permuted table
    sample_columns : List[str]
        List of sample column names to permute
    permutation_type : str
        'within_sample': permute counts within each sample independently
        'swap_labels': swap treatment/control labels
    gene_col : str
        Gene identifier column
    sgrna_col : str
        sgRNA identifier column
    """
    df = pd.read_csv(count_table, sep="\t")

    if permutation_type == "within_sample":
        # Permute counts within each sample column independently
        for col in sample_columns:
            if col in df.columns:
                df[col] = np.random.permutation(df[col].values)

    elif permutation_type == "swap_labels":
        # This is typically done at the analysis level, not here
        # But we can shuffle gene labels
        df[gene_col] = np.random.permutation(df[gene_col].values)

    df.to_csv(output_file, sep="\t", index=False)


def permutation_test_analysis(
    count_table: Union[Path, str],
    control_ids: List[str],
    treatment_ids: List[str],
    full_design_matrix: Optional[Union[Path, str, None]],
    output_dir: Union[Path, str],
    prefix: str,
    run_mageck_func: Callable,
    method_params: Dict = {},
    n_permutations: int = 5,
    permutation_types: List[str] = ["within_sample"],
) -> pd.DataFrame:
    """
    Run permutation tests to assess false positive rates.

    Parameters
    ----------
    count_table : Path or str
        Original count table
    control_ids : List[str]
        Control sample IDs
    treatment_ids : List[str]
        Treatment sample IDs
    output_dir : Path or str
        Output directory
    prefix : str
        Prefix for output files
    run_mageck_func : Callable
        Function to run MAGeCK
    method_params : Dict
        MAGeCK parameters
    n_permutations : int
        Number of permutations to run
    permutation_types : List[str]
        Types of permutations: 'within_sample', 'swap_labels'

    Returns
    -------
    pd.DataFrame
        Summary of permutation test results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for perm_type in permutation_types:
        for i in range(n_permutations):
            perm_name = f"{prefix}_perm_{perm_type}_{i+1}"
            perm_dir = output_dir / f"permutation_{perm_type}_{i+1}"
            perm_dir.mkdir(parents=True, exist_ok=True)

            # Create permuted count table
            perm_count_table = perm_dir / "permuted_counts.tsv"
            all_samples = control_ids + treatment_ids

            create_permuted_count_table(
                count_table=count_table,
                output_file=perm_count_table,
                sample_columns=all_samples,
                permutation_type=perm_type,
            )

            # Run MAGeCK on permuted data
            run_mageck_function(
                run_mageck_func,
                run_dir=perm_dir,
                count_table=perm_count_table,
                ctrl_subset=control_ids,
                treat_subset=treatment_ids,
                design_matrix=full_design_matrix,
                run_name=perm_name,
                **method_params,
            )

            gene_summary = perm_dir / f"{perm_name}.gene_summary.tsv"
            if not gene_summary.exists():
                gene_summary = perm_dir / f"{perm_name}.gene_summary.txt"

            if gene_summary.exists():
                # Count significant hits
                df = pd.read_csv(gene_summary, sep="\t")

                # Try to find FDR columns
                fdr_cols = [col for col in df.columns if "fdr" in col.lower()]

                n_sig_genes = 0
                for fdr_col in fdr_cols:
                    n_sig_genes += (df[fdr_col] < 0.05).sum()

                results.append(
                    {
                        "permutation_type": perm_type,
                        "permutation_number": i + 1,
                        "n_significant_genes": n_sig_genes,
                        "n_fdr_columns": len(fdr_cols),
                    }
                )

    return pd.DataFrame(results)


###############################################################################
# 5. Comprehensive Comparison Wrapper
###############################################################################


def compare_mageck_methods(
    count_table: Union[Path, str],
    output_dir: Union[Path, str],
    control_ids: List[str],
    treatment_ids: List[str],
    full_design_matrix: Optional[Union[Path, str]] = None,
    control_sgrnas: Optional[Union[Path, str]] = None,
    methods: Dict[str, Dict] = None,
    top_n_list: List[int] = [50, 100, 200],
    run_leave_one_out: bool = True,
    run_coherence: bool = True,
    run_control_fp: bool = True,
    run_permutation: bool = True,
    n_permutations: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive comparison of MAGeCK methods.

    Parameters
    ----------
    count_table : Path or str
        MAGeCK count table
    control_ids : List[str]
        Control sample IDs
    treatment_ids : List[str]
        Treatment sample IDs
    output_dir : Path or str
        Output directory for all results
    control_sgrnas : Path or str, optional
        Path to control sgRNAs file
    methods : Dict[str, Dict]
        Dictionary of method configurations:
        {
            'RRA_paired': {
                'run_func': mageck_rra_function,
                'params': {'paired': True, 'norm_method': 'median'},
                'gene_col': 'id',
                'sgrna_summary_suffix': '.sgrna_summary.txt'
            },
            'MLE': {
                'run_func': mageck_mle_function,
                'params': {'design_matrix': 'design.tsv'},
                'gene_col': 'Gene',
                'sgrna_summary_suffix': '.sgrna_summary.txt'
            }
        }
    top_n_list : List[int]
        List of top-N values
    run_leave_one_out : bool
        Whether to run leave-one-out analysis
    run_coherence : bool
        Whether to run sgRNA coherence analysis
    run_control_fp : bool
        Whether to run control false-positive check
    run_permutation : bool
        Whether to run permutation tests
    n_permutations : int
        Number of permutations

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing all analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Run analyses for each method
    for method_name, method_config in methods.items():
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Analyzing method: {method_name}")
        print(f"{'='*80}\n")

        run_func = method_config["run_func"]
        params = method_config.get("params", {})
        gene_col = method_config.get("gene_col", "id")

        # 1. Leave-one-out analysis
        if run_leave_one_out:
            print(f"Running leave-one-replicate-out analysis...")
            try:
                loo_results = leave_one_replicate_out_analysis(
                    count_table=count_table,
                    output_dir=method_dir / "leave_one_out",
                    control_ids=control_ids,
                    treatment_ids=treatment_ids,
                    full_design_matrix=full_design_matrix,
                    prefix=method_name,
                    run_mageck_func=run_func,
                    method_params=params,
                    top_n_list=top_n_list,
                    gene_col=gene_col,
                )
                results[f"{method_name}_leave_one_out"] = loo_results
            except Exception as e:
                print(f"Error in leave-one-out analysis: {e}")

        # 2. sgRNA Coherence (requires full run first)
        if run_coherence:
            print(f"Running sgRNA coherence analysis...")
            try:
                # Run full analysis if not already done
                full_run_dir = method_dir / "full_run"
                full_run_dir.mkdir(parents=True, exist_ok=True)

                run_mageck_function(
                    run_func,
                    run_dir=full_run_dir,
                    count_table=count_table,
                    ctrl_subset=control_ids,
                    treat_subset=treatment_ids,
                    design_matrix=full_design_matrix,
                    run_name=f"{method_name}_full",
                    **params,
                )

                gene_summary = (
                    full_run_dir / f"{method_name}_full.gene_summary.tsv"
                )
                sgrna_summary = (
                    full_run_dir / f"{method_name}_full.sgrna_summary.tsv"
                )

                if not gene_summary.exists():
                    gene_summary = (
                        full_run_dir / f"{method_name}_full.gene_summary.txt"
                    )
                if not sgrna_summary.exists():
                    sgrna_summary = (
                        full_run_dir / f"{method_name}_full.sgrna_summary.txt"
                    )

                if gene_summary.exists() and sgrna_summary.exists():
                    coherence_results = analyze_sgrna_coherence(
                        gene_summary=gene_summary,
                        sgrna_summary=sgrna_summary,
                        top_n=200,
                        gene_col=gene_col,
                    )
                    results[f"{method_name}_coherence"] = coherence_results

                    coherence_results.to_csv(
                        method_dir / f"{method_name}_sgrna_coherence.tsv",
                        sep="\t",
                        index=False,
                    )
            except Exception as e:
                print(f"Error in coherence analysis: {e}")

        # 3. Control False-Positives
        if run_control_fp and control_sgrnas:
            print(f"Running control false-positive analysis...")
            try:
                full_run_dir = method_dir / "full_run"
                gene_summary = (
                    full_run_dir / f"{method_name}_full.gene_summary.tsv"
                )

                if not gene_summary.exists():
                    gene_summary = (
                        full_run_dir / f"{method_name}_full.gene_summary.txt"
                    )

                if gene_summary.exists():
                    fp_results = analyze_control_false_positives(
                        gene_summary=gene_summary,
                        control_sgrnas=control_sgrnas,
                        top_n_list=top_n_list,
                        gene_col=gene_col,
                    )
                    results[f"{method_name}_control_fp"] = fp_results

                    fp_results.to_csv(
                        method_dir
                        / f"{method_name}_control_false_positives.tsv",
                        sep="\t",
                        index=False,
                    )
            except Exception as e:
                print(f"Error in control FP analysis: {e}")

        # 4. Permutation Tests
        if run_permutation:
            print(f"Running permutation tests...")
            try:
                perm_results = permutation_test_analysis(
                    count_table=count_table,
                    control_ids=control_ids,
                    treatment_ids=treatment_ids,
                    full_design_matrix=full_design_matrix,
                    output_dir=method_dir / "permutations",
                    prefix=method_name,
                    run_mageck_func=run_func,
                    method_params=params,
                    n_permutations=n_permutations,
                )
                results[f"{method_name}_permutations"] = perm_results
            except Exception as e:
                print(f"Error in permutation analysis: {e}")

    # Create comparative summary
    print(f"\n{'='*80}")
    print("Creating comparative summary...")
    print(f"{'='*80}\n")

    summary_rows = []

    for method_name in methods.keys():
        row = {"method": method_name}

        # Leave-one-out stats
        loo_key = f"{method_name}_leave_one_out"
        if loo_key in results:
            loo_df = results[loo_key]
            if len(loo_df) > 0:
                row["mean_spearman"] = loo_df["mean_spearman"].iloc[0]
                row["median_spearman"] = loo_df["median_spearman"].iloc[0]
                for n in top_n_list:
                    row[f"mean_jaccard_top_{n}"] = loo_df[
                        f"mean_jaccard_top_{n}"
                    ].iloc[0]

        # Coherence stats
        coh_key = f"{method_name}_coherence"
        if coh_key in results:
            coh_df = results[coh_key]
            if len(coh_df) > 0:
                row["mean_direction_consistency"] = coh_df[
                    "direction_consistency"
                ].mean()
                row["mean_fraction_significant_sgrnas"] = coh_df[
                    "fraction_significant"
                ].mean()

        # Control FP stats
        fp_key = f"{method_name}_control_fp"
        if fp_key in results:
            fp_df = results[fp_key]
            if len(fp_df) > 0:
                for n in top_n_list:
                    top_n_row = fp_df[fp_df["top_n"] == n]
                    if len(top_n_row) > 0:
                        row[f"controls_in_top_{n}"] = top_n_row[
                            "n_controls"
                        ].iloc[0]

        # Permutation stats
        perm_key = f"{method_name}_permutations"
        if perm_key in results:
            perm_df = results[perm_key]
            if len(perm_df) > 0:
                row["mean_perm_sig_genes"] = perm_df[
                    "n_significant_genes"
                ].mean()
                row["std_perm_sig_genes"] = perm_df["n_significant_genes"].std()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    results["comparison_summary"] = summary_df

    # Save summary
    summary_df.to_csv(
        output_dir / "method_comparison_summary.tsv", sep="\t", index=False
    )

    print("\nComparison complete!")
    print(f"Results saved to: {output_dir}")

    return results


###############################################################################
# Rank comparisons
###############################################################################


def rbo_score(list1: List[str], list2: List[str], p: float = 0.9) -> float:
    """
    Rank-Biased Overlap (finite approximation).
    p in (0,1): higher p -> deeper ranks matter more. 0.9 is a common default.
    """
    if not (0 < p < 1):
        raise ValueError("rbo_p muss in (0,1) liegen, z.B. 0.9")

    l1 = list(list1)
    l2 = list(list2)
    d = max(len(l1), len(l2))
    if d == 0:
        return np.nan

    s1, s2 = set(), set()
    summation = 0.0

    for i in range(1, d + 1):
        if i <= len(l1):
            s1.add(l1[i - 1])
        if i <= len(l2):
            s2.add(l2[i - 1])

        overlap = len(s1.intersection(s2))
        a_i = overlap / i
        summation += (p ** (i - 1)) * a_i

    return (1 - p) * summation


def compare_rankings_simple(
    frames: Dict[str, pd.DataFrame],
    specs: Dict[str, Dict[str, str]],
    outdir: str,
    run_prefix: str,
    top_x: int = 100,
    fdr_thresh: Optional[float] = 0.05,
    ascending: Optional[
        Dict[str, bool]
    ] = None,  # per condition: True if smaller is better
    rbo_p: float = 0.9,
    make_combined_plot: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compute pairwise similarity between ranked gene lists across multiple conditions.

    Inputs:
      frames: {"cond": DataFrame}
      specs: {cond: {"id_column": "...", "rankby_column": "...", "fdr_column": "..."}}
             fdr_column can be None/"" if not available (then no filtering).
      outdir: output directory
      run_prefix: filename prefix
      top_x: take top X after optional FDR filter
      fdr_thresh: filter rows where fdr_column <= threshold (if fdr_column present)
      ascending: optional {cond: bool} to control sorting direction by rankby_column
      rbo_p: RBO parameter (0<p<1)
      make_combined_plot: also save a 2x2 combined heatmap PNG

    Outputs:
      long_table: tidy table with columns [metric, cond1, cond2, value]
      matrices: dict metric -> square matrix DataFrame (conds x conds)
    """
    os.makedirs(outdir, exist_ok=True)

    # Validate inputs
    missing_specs = [c for c in frames.keys() if c not in specs]
    if missing_specs:
        raise ValueError(f"Specs fehlen für conditions: {missing_specs}")

    conds = list(frames.keys())
    if len(conds) < 2:
        raise ValueError("Bitte mindestens 2 conditions in frames übergeben.")

    # Build ranked lists per condition
    ranked: Dict[str, list[str]] = {}
    for cond in conds:
        df = pd.read_csv(frames[cond], sep="\t")
        sp = specs[cond]

        id_col = sp["id_column"]
        rank_col = sp["rankby_column"]
        fdr_col = sp.get("fdr_column", None)
        if fdr_col in ("", "None", None):
            fdr_col = None

        # basic checks
        for col in [id_col, rank_col]:
            if col not in df.columns:
                raise ValueError(f"[{cond}] Spalte nicht gefunden: {col}")
        if fdr_col is not None and fdr_col not in df.columns:
            raise ValueError(
                f"[{cond}] fdr_column angegeben, aber nicht im DF: {fdr_col}"
            )

        # optional fdr filter
        if fdr_thresh is not None and fdr_col is not None:
            df = df[df[fdr_col] <= fdr_thresh]

        # drop missing ids
        df = df.dropna(subset=[id_col])

        # choose sorting direction
        asc = False
        if ascending is not None and cond in ascending:
            asc = ascending[cond]
        else:
            # sensible heuristic: if rank column name looks like p/q/fdr => smaller is better
            rc = rank_col.lower()
            if any(
                k in rc
                for k in [
                    "pval",
                    "p_value",
                    "pvalue",
                    "fdr",
                    "qval",
                    "q_value",
                    "qvalue",
                    "padj",
                ]
            ):
                asc = True

        # sort and de-duplicate by best rank
        df = df.sort_values(rank_col, ascending=asc)
        df = df.drop_duplicates(subset=[id_col], keep="first")

        # take top_x
        df = df.head(top_x)

        ranked[cond] = df[id_col].astype(str).tolist()

    # Prepare outputs
    metrics = ["jaccard", "rbo", "kendall_tau", "spearman"]
    n = len(conds)
    mats: Dict[str, pd.DataFrame] = {
        m: pd.DataFrame(np.eye(n), index=conds, columns=conds) for m in metrics
    }

    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            a, b = conds[i], conds[j]
            A = ranked[a]
            B = ranked[b]
            setA, setB = set(A), set(B)

            # Jaccard on top-X sets
            union = setA | setB
            jac = (len(setA & setB) / len(union)) if union else np.nan

            # RBO on ranked lists
            rbo = rbo_score(A, B, p=rbo_p) if (len(A) and len(B)) else np.nan

            # Rank correlations on union, penalize missing with rank (top_x+1)
            genes_union = sorted(union)
            missing_rank = top_x + 1
            posA = {g: k + 1 for k, g in enumerate(A)}
            posB = {g: k + 1 for k, g in enumerate(B)}
            ra = np.array(
                [posA.get(g, missing_rank) for g in genes_union], dtype=float
            )
            rb = np.array(
                [posB.get(g, missing_rank) for g in genes_union], dtype=float
            )

            sp = spearmanr(ra, rb).correlation
            kt = kendalltau(ra, rb).correlation

            mats["jaccard"].loc[a, b] = mats["jaccard"].loc[b, a] = jac
            mats["rbo"].loc[a, b] = mats["rbo"].loc[b, a] = rbo
            mats["spearman"].loc[a, b] = mats["spearman"].loc[b, a] = sp
            mats["kendall_tau"].loc[a, b] = mats["kendall_tau"].loc[b, a] = kt

    # Long table (tidy)
    rows = []
    for m in metrics:
        mat = mats[m]
        for a in conds:
            for b in conds:
                rows.append(
                    {
                        "metric": m,
                        "cond1": a,
                        "cond2": b,
                        "value": mat.loc[a, b],
                    }
                )
    long_table = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(outdir, f"{run_prefix}.ranking_similarity_long.csv")
    long_table.to_csv(csv_path, index=False)

    # Plot + save heatmaps (one per metric)
    def _save_heatmap(metric: str, mat: pd.DataFrame) -> str:
        fig, ax = plt.subplots(
            figsize=(max(6, 0.6 * n + 2), max(5, 0.6 * n + 2))
        )
        im = ax.imshow(mat.values, aspect="auto")

        ax.set_title(f"{run_prefix} — {metric}")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(conds, rotation=45, ha="right")
        ax.set_yticklabels(conds)

        # annotate
        vals = mat.values
        for r in range(n):
            for c in range(n):
                v = vals[r, c]
                txt = "nan" if np.isnan(v) else f"{v:.2f}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        outpath = os.path.join(outdir, f"{run_prefix}.heatmap_{metric}.png")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return outpath

    heatmap_paths = {}
    for m in metrics:
        heatmap_paths[m] = _save_heatmap(m, mats[m])

    # Combined plot (2x2)
    if make_combined_plot:
        fig, axes = plt.subplots(
            2, 2, figsize=(14, 12), constrained_layout=True
        )
        axes = axes.ravel()
        for ax, m in zip(axes, metrics):
            mat = mats[m]
            im = ax.imshow(mat.values, aspect="auto")
            ax.set_title(m)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(conds, rotation=45, ha="right")
            ax.set_yticklabels(conds)

            vals = mat.values
            for r in range(n):
                for c in range(n):
                    v = vals[r, c]
                    txt = "nan" if np.isnan(v) else f"{v:.2f}"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=8)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        combined_path = os.path.join(
            outdir, f"{run_prefix}.heatmaps_combined.png"
        )
        fig.savefig(combined_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Also save matrices as wide CSVs (handy for downstream)
    for m, mat in mats.items():
        mat.to_csv(os.path.join(outdir, f"{run_prefix}.matrix_{m}.csv"))

    return long_table, mats


# ----------------------------
# Example
# ----------------------------
# frames = {"A": dfA, "B": dfB, "C": dfC}
# specs = {
#   "A": {"id_column":"Gene", "rankby_column":"logFC", "fdr_column":"FDR"},
#   "B": {"id_column":"ID",   "rankby_column":"score", "fdr_column":"padj"},
#   "C": {"id_column":"Gene", "rankby_column":"t",     "fdr_column":"qval"},
# }
# ascending = {"A": False, "B": False, "C": False}  # set True if smaller is better
# long, mats = compare_rankings_simple(frames, specs, outdir="out", run_prefix="run1", top_x=50, fdr_thresh=0.05, ascending=ascending)
