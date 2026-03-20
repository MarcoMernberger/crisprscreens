"""
Example: Comprehensive comparison of MAGeCK RRA and MLE methods.

This script demonstrates how to use the method comparison tools to determine
which MAGeCK analysis method is more robust for your dataset.

The comparison includes:
1. Leave-one-replicate-out analysis (replicate consistency)
2. sgRNA coherence analysis (multiple guides per gene)
3. Control sgRNA false-positive checks
4. Permutation tests (negative controls)

Author: CRISPR Screens Package
Date: 2026-01-28
"""

import pypipegraph2 as ppg
from pathlib import Path
from crisprscreens import (
    mageck_method_comparison_job,
    mageck_count_job,
)
from crisprscreens.core.mageck import mageck_test, mageck_mle

# Initialize PyPipeGraph
ppg.new()

###############################################################################
# Configuration
###############################################################################

# Input files
count_table = Path("results/mageck_count/brunello.count.txt")
control_sgrnas = Path(
    "cache/input/consolidated/brunello.genes.sgrnas.controls.txt"
)
design_matrix = Path("incoming/design_matrix.tsv")
design_matrix_batch = Path("incoming/design_matrix_batch.tsv")

# Sample IDs
control_ids = ["Total_Rep1", "Total_Rep2", "Total_Rep3"]
treatment_ids = ["Sort_Rep1", "Sort_Rep2", "Sort_Rep3"]

# Output directory
comparison_dir = Path("results/method_comparison")
comparison_dir.mkdir(parents=True, exist_ok=True)

###############################################################################
# Define Methods to Compare
###############################################################################

methods = {
    # RRA: Robust Rank Aggregation (original MAGeCK)
    "RRA_paired_median": {
        "run_func": mageck_test,
        "params": {
            "paired": True,
            "norm_method": "median",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "id",  # RRA uses 'id' as gene column
    },
    "RRA_unpaired_median": {
        "run_func": mageck_test,
        "params": {
            "paired": False,
            "norm_method": "median",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "id",
    },
    "RRA_paired_control_norm": {
        "run_func": mageck_test,
        "params": {
            "paired": True,
            "norm_method": "control",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "id",
    },
    # MLE: Maximum Likelihood Estimation (for batch effects)
    "MLE_median_norm": {
        "run_func": mageck_mle,
        "params": {
            "design_matrix": str(design_matrix),
            "norm_method": "median",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "Gene",  # MLE uses 'Gene' as gene column
    },
    "MLE_batch_median": {
        "run_func": mageck_mle,
        "params": {
            "design_matrix": str(design_matrix_batch),
            "norm_method": "median",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "Gene",
    },
    "MLE_batch_control_norm": {
        "run_func": mageck_mle,
        "params": {
            "design_matrix": str(design_matrix_batch),
            "norm_method": "control",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "Gene",
    },
}

###############################################################################
# Run Comprehensive Comparison
###############################################################################

comparison_job = mageck_method_comparison_job(
    count_table=str(count_table),
    control_ids=control_ids,
    treatment_ids=treatment_ids,
    output_dir=str(comparison_dir),
    control_sgrnas=str(control_sgrnas),
    methods=methods,
    top_n_list=[50, 100, 200],
    run_leave_one_out=True,
    run_coherence=True,
    run_control_fp=True,
    run_permutation=True,
    n_permutations=5,
    dependencies=[],  # Add count_job here if needed
)

# Run the pipeline
ppg.run()

###############################################################################
# Interpretation Guide
###############################################################################

print("\n" + "=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)

print(
    """
Results are saved in: results/method_comparison/

Key files to check:
1. method_comparison_summary.tsv - Overall comparison summary
   
   Look for:
   - Higher mean_spearman: More consistent rankings between replicates
   - Higher mean_jaccard_top_N: More overlap in top hits
   - Higher mean_direction_consistency: sgRNAs agree in direction
   - Lower controls_in_top_N: Fewer false positives
   - Lower mean_perm_sig_genes: Fewer hits in permuted data

2. Per-method directories (e.g., RRA_paired_median/, MLE_batch_median/):
   - *_leave_one_out_comparison.tsv: Pairwise replicate consistency
   - *_sgrna_coherence.tsv: Per-gene sgRNA consistency
   - *_control_false_positives.tsv: Control sgRNAs in top hits
   - permutations/: Results from permuted data

INTERPRETATION:

The "better" method shows:
✓ Higher Spearman correlations (more stable rankings)
✓ Higher Jaccard indices (more consistent top hits)
✓ Higher sgRNA direction consistency (multiple guides agree)
✓ Fewer control sgRNAs in top hits (lower false positives)
✓ Fewer significant genes in permuted data (better calibrated)

Common findings:
- MLE is better when you have batch effects (e.g., different sequencing runs)
- RRA paired is better for simple designs with matched replicates
- Control normalization helps if controls are stable

Decision tree:
1. If leave-one-out shows low consistency → try different normalization
2. If controls appear in top hits → method poorly calibrated for your data
3. If sgRNA coherence is low → check library quality or increase stringency
4. If permutation test shows many hits → method too liberal

Next steps:
- Review method_comparison_summary.tsv
- Choose the method with best overall metrics
- Use that method for final analysis
"""
)

###############################################################################
# Optional: Run only specific analyses
###############################################################################

# Example: Only run leave-one-out for two specific methods
if False:  # Set to True to run
    from crisprscreens import leave_one_replicate_out_job

    rra_loo_job = leave_one_replicate_out_job(
        count_table=str(count_table),
        control_ids=control_ids,
        treatment_ids=treatment_ids,
        output_dir=str(comparison_dir / "RRA_paired_median_LOO"),
        prefix="RRA_paired_median",
        run_mageck_func=mageck_test,
        method_params={"paired": True, "norm_method": "median"},
        top_n_list=[50, 100, 200],
        gene_col="id",
    )

    mle_loo_job = leave_one_replicate_out_job(
        count_table=str(count_table),
        control_ids=control_ids,
        treatment_ids=treatment_ids,
        output_dir=str(comparison_dir / "MLE_batch_LOO"),
        prefix="MLE_batch",
        run_mageck_func=mageck_mle,
        method_params={
            "design_matrix": str(design_matrix_batch),
            "norm_method": "median",
        },
        top_n_list=[50, 100, 200],
        gene_col="Gene",
    )

# Example: Only check control false positives
if False:  # Set to True to run
    from crisprscreens import control_false_positive_job

    rra_fp_job = control_false_positive_job(
        gene_summary="results/mageck_test/sorted_vs_unsorted/RRA.gene_summary.tsv",
        control_sgrnas=str(control_sgrnas),
        output_file=str(comparison_dir / "RRA_control_fp.tsv"),
        top_n_list=[50, 100, 200, 500],
        gene_col="id",
    )

# Example: Only run sgRNA coherence
if False:  # Set to True to run
    from crisprscreens import sgrna_coherence_job

    mle_coherence_job = sgrna_coherence_job(
        gene_summary="results/mageck_mle/sorted_vs_unsorted/MLE.gene_summary.tsv",
        sgrna_summary="results/mageck_mle/sorted_vs_unsorted/MLE.sgrna_summary.txt",
        output_file=str(comparison_dir / "MLE_sgrna_coherence.tsv"),
        top_n=200,
        gene_col="Gene",
    )
