"""
Example script demonstrating control sgRNA quality control analysis.

This script shows how to use the control QC functionality both standalone
and integrated with pypipegraph workflows.
"""

from pathlib import Path
from crisprscreens.core.qc import (
    control_sgrna_qc,
    generate_control_qc_report,
)
from crisprscreens.jobs.qc_jobs import control_qc_job


def example_standalone_analysis():
    """
    Example: Run control QC analysis standalone (no pypipegraph).
    """
    # Define paths
    count_table = "results/mageck_count/all/counts.count.tsv"
    control_file = "incoming/control_sgRNAs.txt"
    output_dir = "results/qc/control_sgrnas"
    baseline_condition = "Total"  # or "T0", "unsorted", etc.

    # Generate full QC report
    print("Running control sgRNA QC analysis...")
    report = generate_control_qc_report(
        count_table=count_table,
        control_sgrnas=control_file,
        baseline_condition=baseline_condition,
        output_dir=output_dir,
        prefix="control_qc",
        sgrna_col="sgRNA",
        gene_col="Gene",
        delimiter="_",
        save_formats=["png", "pdf"],
    )

    # Access results
    qc_results = report["qc_results"]
    metrics = qc_results["metrics"]

    # Print summary
    print("\n" + "="*60)
    print("Control sgRNA QC Summary")
    print("="*60)

    for condition, metric in metrics.items():
        print(f"\n{condition} vs {baseline_condition}:")
        print(f"  Median Δc: {metric['median']:.3f}")
        print(f"  IQR: {metric['iqr']:.3f}")
        print(f"  Tail rate (|Δc|>1): {metric['tail_rate_1.0']*100:.1f}%")
        print(f"  Wilcoxon p-value: {metric['wilcoxon_pvalue']:.2e}")

        # Interpretation
        if abs(metric['median']) < 0.1 and metric['tail_rate_1.0'] < 0.05:
            print("  ✓ Controls look stable")
        elif abs(metric['median']) < 0.3 and metric['tail_rate_1.0'] < 0.1:
            print("  ⚠ Controls show slight drift")
        else:
            print("  ✗ Controls show strong systematic shift!")

    print(f"\n\nFull report saved to: {output_dir}")

    return report


def example_programmatic_analysis():
    """
    Example: Run QC analysis programmatically and access all results.
    """
    # Run QC
    qc_results = control_sgrna_qc(
        count_table="results/mageck_count/all/counts.count.tsv",
        control_sgrnas="incoming/control_sgRNAs.txt",
        baseline_condition="Total",
    )

    # Access individual components
    print(f"Found {len(qc_results['raw_counts'])} control sgRNAs")
    print(f"Conditions: {list(qc_results['conditions'].keys())}")

    # Check pairwise shifts
    pairwise = qc_results["pairwise_median"]
    max_shift = pairwise.abs().max().max()
    print(f"Maximum pairwise shift: {max_shift:.3f}")

    # Check replicate correlations
    for cond, corr_matrix in qc_results["replicate_correlations"].items():
        if len(corr_matrix) > 1:
            # Get off-diagonal (inter-replicate) correlations
            n = len(corr_matrix)
            off_diag = corr_matrix.values[np.triu_indices(n, k=1)]
            min_corr = off_diag.min()
            print(f"{cond} min replicate correlation: {min_corr:.3f}")

    return qc_results


def example_pypipegraph_integration():
    """
    Example: Integrate control QC into pypipegraph workflow.

    This would typically be added to your main run.py script.
    """
    import pypipegraph2 as ppg2
    from crisprscreens.jobs.mageck_jobs import mageck_count_job

    # Initialize pypipegraph
    ppg2.new_pipeline()

    # Define paths
    results_dir = Path("results")
    mageck_count_dir = results_dir / "mageck_count" / "all"
    qc_dir = results_dir / "qc" / "control_sgrnas"

    prefix = "counts"
    control_file = Path("incoming/control_sgRNAs.txt")

    # Create mageck count job (assuming it exists)
    # count_job = mageck_count_job(...)

    # Create control QC job
    qc_job = control_qc_job(
        count_table=mageck_count_dir / f"{prefix}.count.tsv",
        control_sgrnas=control_file,
        baseline_condition="Total",  # Adjust to your baseline
        output_dir=qc_dir,
        prefix="control_qc",
        sgrna_col="sgRNA",
        gene_col="Gene",
        delimiter="_",
        save_formats=["png", "pdf"],
        dependencies=[],  # Add [count_job] if you have it
    )

    # Run
    ppg2.run()

    print(f"QC analysis complete. Results in {qc_dir}")


def example_custom_baseline_detection():
    """
    Example: Auto-detect baseline condition from column names.
    """
    import pandas as pd

    # Load count table to inspect columns
    count_df = pd.read_csv(
        "results/mageck_count/all/counts.count.tsv",
        sep="\t",
        nrows=5
    )

    # Get sample columns
    sample_cols = [col for col in count_df.columns 
                   if col not in ["sgRNA", "Gene"]]

    # Parse conditions
    conditions = set()
    for col in sample_cols:
        condition = col.rsplit("_", 1)[0]
        conditions.add(condition)

    print(f"Available conditions: {conditions}")

    # Auto-detect baseline (common patterns)
    baseline_patterns = ["Total", "T0", "unsorted", "Plasmid", "Input"]
    baseline = None
    for pattern in baseline_patterns:
        for cond in conditions:
            if pattern.lower() in cond.lower():
                baseline = cond
                break
        if baseline:
            break

    if baseline:
        print(f"Auto-detected baseline: {baseline}")
    else:
        print("Could not auto-detect baseline. Please specify manually.")
        baseline = list(conditions)[0]
        print(f"Using first condition as baseline: {baseline}")

    return baseline


def example_interpretation_guidelines():
    """
    Print interpretation guidelines for QC metrics.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         Control sgRNA QC Interpretation Guide                ║
    ╚══════════════════════════════════════════════════════════════╝

    1. UNIVARIATE CHECKS (Distribution per condition vs baseline)
    ───────────────────────────────────────────────────────────────
    ✓ GOOD:
      - Median Δc ≈ 0 (± 0.1)
      - Symmetric distribution around 0
      - Tail rate (|Δc| > 1) < 5%
      - Similar across replicates

    ⚠ ACCEPTABLE:
      - Median Δc between 0.1-0.3
      - Tail rate 5-10%
      - Consistent across replicates

    ✗ PROBLEMATIC:
      - Median Δc > 0.5
      - Strong skew
      - Tail rate > 15%
      - High variation between replicates
      → Controls show systematic shifts or instability


    2. PAIRWISE COMPARISONS (Heatmap)
    ───────────────────────────────────────────────────────────────
    ✓ GOOD:
      - All pairwise median shifts ≈ 0
      - Symmetric around diagonal

    ✗ PROBLEMATIC:
      - One condition shows consistent offset
      → Global drift in that condition
      → May indicate normalization issue


    3. REPLICATE CONSISTENCY (Correlation heatmap)
    ───────────────────────────────────────────────────────────────
    ✓ GOOD:
      - High correlation (> 0.9) between replicates
      - Consistent across conditions

    ⚠ WATCH:
      - Correlation 0.8-0.9
      → Some biological/technical variation

    ✗ PROBLEMATIC:
      - Correlation < 0.8
      - One replicate deviates
      → Batch effect, contamination, or technical failure


    4. PCA ANALYSIS
    ───────────────────────────────────────────────────────────────
    ✓ GOOD (Controls are neutral):
      - Samples cluster by replicate
      - Conditions do NOT separate clearly
      - PC1/PC2 explain technical variation, not biology

    ✗ PROBLEMATIC (Controls show treatment effect):
      - Conditions separate on PC1
      - Strong biological signal in controls
      → Controls are NOT neutral
      → May be responding to treatment
      → Not suitable for normalization


    5. WILCOXON TEST
    ───────────────────────────────────────────────────────────────
    Tests H0: Median Δc = 0

    ✓ GOOD:
      - p-value > 0.05 (fail to reject H0)
      → No significant shift from baseline

    ✗ PROBLEMATIC:
      - p-value < 0.01
      → Significant systematic shift

    Note: With large N, even small shifts can be "significant"
          → Always check effect size (median, IQR)


    6. OVERALL DECISION TREE
    ───────────────────────────────────────────────────────────────

    IF all conditions show:
       - |median Δc| < 0.2
       - tail rate < 10%
       - high replicate correlation (>0.85)
       - PCA does NOT separate by condition
    THEN: ✓ Controls are suitable for normalization

    ELSE IF:
       - One condition shows systematic shift
       → Consider excluding that condition
       → Or use median normalization instead

    ELSE IF:
       - Controls show treatment response in PCA
       → Consider using different control set
       → Or use median/total-count normalization

    """)


if __name__ == "__main__":
    # Run standalone analysis
    example_standalone_analysis()

    # Print interpretation guide
    example_interpretation_guidelines()
