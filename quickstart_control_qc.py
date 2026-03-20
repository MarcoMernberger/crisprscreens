#!/usr/bin/env python3
"""
Quick Start Script - Control sgRNA QC Analysis

This script demonstrates how to run control QC on your actual data.
Adjust paths and parameters as needed.
"""

from pathlib import Path
from crisprscreens.core.qc import generate_control_qc_report

# ============================================================================
# CONFIGURATION - ADJUST THESE PATHS FOR YOUR DATA
# ============================================================================

# Path to your MAGeCK count table
COUNT_TABLE = Path("results/mageck_count/all/counts.count.tsv")

# Path to your control sgRNA file
CONTROL_FILE = Path("incoming/control_sgRNAs.txt")

# Output directory for QC reports
OUTPUT_DIR = Path("results/qc/control_sgrnas")

# Baseline condition name (must match column prefix before "_")
# Check your count table columns to identify the correct name
# Common options: "Total", "T0", "unsorted", "Plasmid", "Input"
BASELINE_CONDITION = "Total"

# Column names in count table (usually don't need to change)
SGRNA_COL = "sgRNA"
GENE_COL = "Gene"

# Delimiter for condition_replicate parsing (usually "_")
DELIMITER = "_"

# Output formats for plots
SAVE_FORMATS = ["png", "pdf"]

# ============================================================================
# OPTIONAL: CHECK YOUR DATA FIRST
# ============================================================================

def check_data():
    """Check your data before running QC."""
    import pandas as pd

    print("="*60)
    print("Data Check")
    print("="*60)

    # Check count table
    if not COUNT_TABLE.exists():
        print(f"❌ Count table not found: {COUNT_TABLE}")
        print("   Please adjust COUNT_TABLE path")
        return False

    print(f"✓ Found count table: {COUNT_TABLE}")

    # Check control file
    if not CONTROL_FILE.exists():
        print(f"❌ Control file not found: {CONTROL_FILE}")
        print("   Please adjust CONTROL_FILE path")
        return False

    print(f"✓ Found control file: {CONTROL_FILE}")

    # Check columns
    print("\nReading count table (first 5 rows)...")
    df = pd.read_csv(COUNT_TABLE, sep="\t", nrows=5)

    print(f"\n  Columns found: {list(df.columns)}")
    print(f"  sgRNA column present: {SGRNA_COL in df.columns}")
    print(f"  Gene column present: {GENE_COL in df.columns}")

    # Parse conditions
    sample_cols = [col for col in df.columns if col not in [SGRNA_COL, GENE_COL]]
    conditions = set()
    for col in sample_cols:
        condition = col.rsplit(DELIMITER, 1)[0]
        conditions.add(condition)

    print(f"\n  Detected conditions: {sorted(conditions)}")
    print(f"  Your baseline: {BASELINE_CONDITION}")

    if BASELINE_CONDITION not in conditions:
        print(f"\n❌ WARNING: Baseline '{BASELINE_CONDITION}' not found in conditions!")
        print(f"   Available conditions: {sorted(conditions)}")
        print(f"   Please update BASELINE_CONDITION")
        return False

    print(f"✓ Baseline condition found")

    # Check control sgRNAs
    with open(CONTROL_FILE) as f:
        control_ids = [line.strip() for line in f if line.strip()]

    print(f"\n  Control sgRNAs in file: {len(control_ids)}")

    # Check if controls are in count table
    full_df = pd.read_csv(COUNT_TABLE, sep="\t")
    controls_in_table = full_df[SGRNA_COL].isin(control_ids).sum()
    print(f"  Controls found in count table: {controls_in_table}")

    if controls_in_table == 0:
        print(f"\n❌ WARNING: No control sgRNAs found in count table!")
        print(f"   Check that sgRNA IDs match between files")
        return False

    print(f"✓ Found {controls_in_table} / {len(control_ids)} controls in count table")

    return True


# ============================================================================
# RUN QC ANALYSIS
# ============================================================================

def run_qc():
    """Run the complete QC analysis."""
    print("\n" + "="*60)
    print("Running Control sgRNA QC Analysis")
    print("="*60)

    report = generate_control_qc_report(
        count_table=COUNT_TABLE,
        control_sgrnas=CONTROL_FILE,
        baseline_condition=BASELINE_CONDITION,
        output_dir=OUTPUT_DIR,
        prefix="control_qc",
        sgrna_col=SGRNA_COL,
        gene_col=GENE_COL,
        delimiter=DELIMITER,
        save_formats=SAVE_FORMATS,
    )

    print("\n" + "="*60)
    print("QC Analysis Complete!")
    print("="*60)

    # Print summary
    qc_results = report["qc_results"]
    metrics = qc_results["metrics"]

    print("\nResults Summary:")
    print("-" * 60)

    for condition, metric in metrics.items():
        print(f"\n{condition} vs {BASELINE_CONDITION}:")
        print(f"  Median Δc: {metric['median']:>8.3f}")
        print(f"  IQR:       {metric['iqr']:>8.3f}")
        print(f"  Tail (|Δc|>1): {metric['tail_rate_1.0']*100:>5.1f}%")
        print(f"  Wilcoxon p: {metric['wilcoxon_pvalue']:>7.2e}")

        # Quick interpretation
        if abs(metric['median']) < 0.1 and metric['tail_rate_1.0'] < 0.05:
            status = "✓ Controls look stable"
        elif abs(metric['median']) < 0.3 and metric['tail_rate_1.0'] < 0.1:
            status = "⚠ Controls show slight drift"
        else:
            status = "✗ Controls show strong systematic shift!"
        print(f"  → {status}")

    print("\n" + "="*60)
    print(f"Full report saved to: {OUTPUT_DIR}")
    print("="*60)

    print("\nGenerated files:")
    for key, filepath in report["files"].items():
        print(f"  {Path(filepath).name}")

    print("\n📊 Check the plots to validate control stability!")
    print("   Priority: Look at PCA and distribution plots first")

    return report


# ============================================================================
# INTERPRETATION HELPER
# ============================================================================

def print_interpretation_guide():
    """Print quick interpretation guide."""
    print("\n" + "="*60)
    print("Quick Interpretation Guide")
    print("="*60)
    print("""
    ✅ GOOD Controls (use control normalization):
       • Median Δc close to 0 (± 0.1)
       • Low tail rate (< 5%)
       • High replicate correlation (> 0.9)
       • PCA: samples cluster by replicate, not condition

    ⚠️  ACCEPTABLE (can proceed with caution):
       • Median Δc: 0.1-0.3
       • Tail rate: 5-10%
       • Moderate correlation: 0.8-0.9

    ❌ PROBLEMATIC (don't use control normalization):
       • Median Δc > 0.5
       • High tail rate (> 15%)
       • Low correlation (< 0.8)
       • PCA: conditions separate on PC1
       → Use median normalization instead
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check data first
    if not check_data():
        print("\n❌ Data check failed. Please fix the issues above.")
        sys.exit(1)

    # Ask for confirmation
    print("\n" + "="*60)
    response = input("Proceed with QC analysis? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Aborted.")
        sys.exit(0)

    # Run QC
    try:
        report = run_qc()
        print_interpretation_guide()
        print("\n✅ Success!")
    except Exception as e:
        print(f"\n❌ Error during QC analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
