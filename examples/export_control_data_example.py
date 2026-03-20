#!/usr/bin/env python3
"""
Example: Export Control sgRNA Counts and CPM

This script demonstrates how to export raw counts and CPM values
for control sgRNAs only.
"""

from pathlib import Path

# ============================================================================
# Example 1: Standalone usage (direct Python)
# ============================================================================


def example_standalone():
    """Export control data without pypipegraph."""
    from crisprscreens.core.qc import export_control_counts_and_cpm

    files = export_control_counts_and_cpm(
        count_table="results/mageck_count/all/counts.count.tsv",
        control_sgrnas="incoming/control_sgRNAs.txt",
        output_dir="results/control_data",
        prefix="control_data",
    )

    print("\n✓ Exported control data:")
    print(f"  Raw counts: {files['raw_counts']}")
    print(f"  CPM:        {files['cpm']}")

    # Load and inspect
    import pandas as pd

    raw_counts = pd.read_csv(files["raw_counts"], sep="\t")
    cpm = pd.read_csv(files["cpm"], sep="\t")

    print(f"\n📊 Data summary:")
    print(f"  Control sgRNAs: {len(raw_counts)}")
    print(
        f"  Samples:        {len(raw_counts.columns) - 2}"
    )  # Exclude sgRNA, Gene
    print(
        f"\n  Raw count range: {raw_counts.iloc[:, 2:].min().min():.0f} - {raw_counts.iloc[:, 2:].max().max():.0f}"
    )
    print(
        f"  CPM range:       {cpm.iloc[:, 2:].min().min():.2f} - {cpm.iloc[:, 2:].max().max():.2f}"
    )


# ============================================================================
# Example 2: PyPipeGraph integration
# ============================================================================


def example_pypipegraph():
    """Integrate control data export into pypipegraph workflow."""
    import pypipegraph2 as ppg2
    from crisprscreens import export_control_data_job

    # Initialize pypipegraph
    ppg2.new_pipeline()

    # Create job to export control data
    export_job = export_control_data_job(
        count_table="results/mageck_count/all/counts.count.tsv",
        control_sgrnas="incoming/control_sgRNAs.txt",
        output_dir="results/control_data",
        prefix="control_data",
        dependencies=[],  # Add [count_job] if you have it
    )

    # Run
    ppg2.run()

    print("✓ Control data exported via pypipegraph")


# ============================================================================
# Example 3: Combined with QC analysis
# ============================================================================


def example_combined():
    """Export data and run QC together."""
    from crisprscreens.core.qc import (
        export_control_counts_and_cpm,
        generate_control_qc_report,
    )

    # Export raw data
    print("📤 Exporting control data...")
    files = export_control_counts_and_cpm(
        count_table="results/mageck_count/all/counts.count.tsv",
        control_sgrnas="incoming/control_sgRNAs.txt",
        output_dir="results/control_data",
    )

    # Run QC
    print("\n📊 Running QC analysis...")
    report = generate_control_qc_report(
        count_table="results/mageck_count/all/counts.count.tsv",
        control_sgrnas="incoming/control_sgRNAs.txt",
        baseline_condition="Total",
        output_dir="results/qc/control_sgrnas",
    )

    print("\n✓ Complete!")
    print(f"  Control data: results/control_data/")
    print(f"  QC report:    results/qc/control_sgrnas/")


# ============================================================================
# Example 4: Verify CPM calculation is correct
# ============================================================================


def example_verify_cpm():
    """Verify that CPM is calculated on full library."""
    import pandas as pd
    import numpy as np

    # Load full count table
    full_counts = pd.read_csv(
        "results/mageck_count/all/counts.count.tsv", sep="\t"
    )

    # Load control sgRNAs
    with open("incoming/control_sgRNAs.txt") as f:
        controls = {line.strip() for line in f if line.strip()}

    # Sample column (first data column)
    sample_col = [c for c in full_counts.columns if c not in ["sgRNA", "Gene"]][
        0
    ]

    # Calculate total library size (all sgRNAs)
    total_library_size = full_counts[sample_col].sum()

    # Get one control sgRNA
    control_mask = full_counts["sgRNA"].isin(controls)
    control_sgRNA = full_counts[control_mask].iloc[0]
    raw_count = control_sgRNA[sample_col]

    # Manual CPM calculation (on full library)
    expected_cpm = (raw_count / total_library_size) * 1e6

    # Load exported CPM
    cpm_df = pd.read_csv("results/control_data/control_data_cpm.tsv", sep="\t")
    exported_cpm = cpm_df[cpm_df["sgRNA"] == control_sgRNA["sgRNA"]][
        sample_col
    ].values[0]

    print("\n🔍 CPM Calculation Verification:")
    print(f"  Sample:             {sample_col}")
    print(f"  Control sgRNA:      {control_sgRNA['sgRNA']}")
    print(f"  Raw count:          {raw_count}")
    print(f"  Total library size: {total_library_size:,}")
    print(f"\n  Expected CPM:       {expected_cpm:.4f}")
    print(f"  Exported CPM:       {exported_cpm:.4f}")
    print(
        f"  Match:              {'✓ YES' if np.isclose(expected_cpm, exported_cpm) else '✗ NO'}"
    )

    # Verify it's NOT calculated on controls only
    control_only_size = full_counts[control_mask][sample_col].sum()
    wrong_cpm = (raw_count / control_only_size) * 1e6

    print(f"\n  If calculated on controls only:")
    print(f"    Control library size: {control_only_size:,}")
    print(f"    Wrong CPM:           {wrong_cpm:.4f}")
    print(
        f"    Would be different:  {'✓ Correct' if not np.isclose(wrong_cpm, exported_cpm) else '✗ Bug!'}"
    )


# ============================================================================
# Example 5: PyPipeGraph workflow with both jobs
# ============================================================================


def example_full_workflow():
    """Complete workflow: export data + QC analysis."""
    import pypipegraph2 as ppg2
    from crisprscreens import export_control_data_job, control_qc_job

    ppg2.new_pipeline()

    count_table = "results/mageck_count/all/counts.count.tsv"
    control_file = "incoming/control_sgRNAs.txt"

    # Job 1: Export control data
    export_job = export_control_data_job(
        count_table=count_table,
        control_sgrnas=control_file,
        output_dir="results/control_data",
        prefix="control_data",
        dependencies=[],  # Add count_job if you have it
    )

    # Job 2: Run QC analysis (can run in parallel with export)
    qc_job = control_qc_job(
        count_table=count_table,
        control_sgrnas=control_file,
        baseline_condition="Total",
        output_dir="results/qc/control_sgrnas",
        prefix="control_qc",
        dependencies=[],  # Both depend on count_job, can run in parallel
    )

    ppg2.run()

    print("✓ Complete workflow finished")
    print("  - Control counts/CPM exported")
    print("  - QC analysis completed")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Control sgRNA Data Export Examples")
    print("=" * 60)

    # Run standalone example
    example_standalone()

    # Verify CPM calculation
    example_verify_cpm()

    print("\n" + "=" * 60)
    print("✓ Examples complete")
    print("=" * 60)
    print("\n💡 Usage in your workflow:")
    print("   from crisprscreens import export_control_data_job")
    print("   export_job = export_control_data_job(...)")
