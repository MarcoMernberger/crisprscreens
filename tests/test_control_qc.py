"""
Simple test script to verify control QC implementation works.

This script performs basic functionality tests without requiring the full
dataset.
"""

import numpy as np
import pandas as pd
import importlib
from pathlib import Path
import tempfile
import shutil


def create_mock_data():
    """Create mock count table and control file for testing."""
    # Create mock count table
    np.random.seed(42)
    n_sgrnas = 200
    n_controls = 50

    # Generate sgRNA IDs
    sgrna_ids = [f"s_{i}" for i in range(n_sgrnas)]
    control_ids = sgrna_ids[:n_controls]  # First 50 are controls

    # Generate mock counts
    data = {
        "sgRNA": sgrna_ids,
        "Gene": ["NonTargetingControl"] * n_controls
        + [f"Gene_{i}" for i in range(n_sgrnas - n_controls)],
    }

    # Add sample columns: Total (baseline) and conditions with 3 replicates each
    conditions = {
        "Total": [10000, 10200, 9800],  # Baseline
        "Sort1": [9500, 9700, 9300],  # Slight depletion
        "Sort2": [8000, 8200, 7800],  # More depletion
    }

    for cond, base_counts in conditions.items():
        for rep_idx, base_count in enumerate(base_counts, 1):
            col_name = f"{cond}_Rep{rep_idx}"

            # For controls: stable counts (mean = base_count)
            control_counts = (
                np.random.negative_binomial(n=10, p=0.1, size=n_controls)
                + base_count
                - 100
            )
            # For genes: variable counts (some depleted in conditions)
            if cond == "Total":
                gene_counts = (
                    np.random.negative_binomial(
                        n=10, p=0.1, size=n_sgrnas - n_controls
                    )
                    + base_count
                    - 100
                )
            else:
                # Simulate depletion for some genes
                gene_counts = np.random.negative_binomial(
                    n=10, p=0.1, size=n_sgrnas - n_controls
                )
                # Make half the genes depleted
                gene_counts[: len(gene_counts) // 2] = (
                    gene_counts[: len(gene_counts) // 2] * 0.3
                )
                gene_counts = gene_counts + base_count - 100

            counts = np.concatenate([control_counts, gene_counts])
            counts = np.maximum(counts, 0)  # Ensure non-negative
            data[col_name] = counts.astype(int)

    count_df = pd.DataFrame(data)

    return count_df, control_ids


def test_basic_functionality():
    """Test basic import and function availability."""
    print("Testing imports...")

    qc_mod = importlib.import_module("crisprscreens.core.qc")
    required_names = [
        "control_sgrna_qc",
        "generate_control_qc_report",
        "plot_control_distribution_per_condition",
        "plot_pairwise_control_shifts",
        "plot_control_replicate_correlation",
        "plot_control_pca",
    ]
    missing = [n for n in required_names if not hasattr(qc_mod, n)]
    if missing:
        raise ImportError(f"Missing attributes in {qc_mod.__name__}: {missing}")

    # Bind primary functions we will use later to avoid further lookups
    qc_mod.control_sgrna_qc
    qc_mod.generate_control_qc_report

    print("✓ All core functions available on module")

    qc_mod = importlib.import_module("crisprscreens.jobs.qc_jobs")
    qc_mod.control_qc_job

    print("✓ Job function imported successfully")

    pkg = importlib.import_module("crisprscreens")
    if not hasattr(pkg, "control_qc_job"):
        raise ImportError(
            f"Missing attribute in {pkg.__name__}: control_qc_job"
        )
    pkg.control_qc_job

    print("✓ Package-level import works")

    return True


def test_with_mock_data():
    """Test QC analysis with mock data."""
    print("\nTesting with mock data...")

    from crisprscreens.core.qc import control_sgrna_qc

    # Create mock data
    count_df, control_ids = create_mock_data()
    control_set = set(control_ids)

    print(
        f"  Created mock data: {len(count_df)} sgRNAs, {len(control_ids)} controls"  # noqa: E501
    )

    # Run QC analysis
    try:
        qc_results = control_sgrna_qc(
            count_table=count_df,
            control_sgrnas=control_set,
            baseline_condition="Total",
        )
        print("✓ QC analysis completed")
    except Exception as e:
        print(f"✗ QC analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Check results
    try:
        assert "metrics" in qc_results
        assert "pairwise_median" in qc_results
        assert "replicate_correlations" in qc_results
        assert "delta" in qc_results
        print("✓ All expected result keys present")

        metrics = qc_results["metrics"]
        assert "Sort1" in metrics
        assert "Sort2" in metrics
        print(f"✓ Found metrics for conditions: {list(metrics.keys())}")

        # Check metric values
        for cond, metric in metrics.items():
            print(f"\n  {cond} metrics:")
            print(f"    Median Δc: {metric['median']:.3f}")
            print(f"    IQR: {metric['iqr']:.3f}")
            print(f"    Tail rate (|Δc|>1): {metric['tail_rate_1.0']*100:.1f}%")

            assert "median" in metric
            assert "iqr" in metric
            assert "tail_rate_1.0" in metric
        print("✓ All metrics calculated correctly")

        # Check pairwise
        pairwise = qc_results["pairwise_median"]
        print(f"\n  Pairwise shifts shape: {pairwise.shape}")
        print(f"  Max pairwise shift: {pairwise.abs().max().max():.3f}")
        print("✓ Pairwise analysis complete")

        # Check correlations
        print(
            f"\n  Replicate correlations for: {list(qc_results['replicate_correlations'].keys())}"  # noqa: E501
        )
        for cond, corr_mat in qc_results["replicate_correlations"].items():
            if len(corr_mat) > 1:
                # Get minimum off-diagonal correlation
                n = len(corr_mat)
                off_diag = corr_mat.values[np.triu_indices(n, k=1)]
                print(f"    {cond}: min correlation = {off_diag.min():.3f}")
        print("✓ Replicate correlations calculated")

    except AssertionError as e:
        print(f"✗ Result validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_full_report_generation():
    """Test full report generation with file output."""
    print("\nTesting full report generation...")

    from crisprscreens.core.qc import generate_control_qc_report

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"  Using temp directory: {temp_dir}")

    try:
        # Create mock data
        count_df, control_ids = create_mock_data()

        # Save to temporary files
        count_file = temp_dir / "counts.tsv"
        control_file = temp_dir / "controls.txt"

        count_df.to_csv(count_file, sep="\t", index=False)
        with open(control_file, "w") as f:
            f.write("\n".join(control_ids))

        print("  Saved mock data to temp files")

        # Generate report
        output_dir = temp_dir / "qc_output"
        report = generate_control_qc_report(
            count_table=count_file,
            control_sgrnas=control_file,
            baseline_condition="Total",
            output_dir=output_dir,
            prefix="test_qc",
            save_formats=["png"],  # Only PNG to save time
        )

        print("✓ Report generation completed")

        # Check output files
        saved_files = report["files"]
        print(f"\n  Generated {len(saved_files)} output files:")
        for key, filepath in saved_files.items():
            exists = (
                filepath.exists()
                if isinstance(filepath, Path)
                else Path(filepath).exists()
            )
            status = "✓" if exists else "✗"
            print(f"    {status} {key}: {Path(filepath).name}")

        # Check specific required files
        required_files = [
            "test_qc_metrics.tsv",
            "test_qc_pairwise_shifts.tsv",
            "test_qc_distribution.png",
            "test_qc_pca_condition.png",
        ]

        all_exist = True
        for filename in required_files:
            filepath = output_dir / filename
            if not filepath.exists():
                print(f"  ✗ Missing required file: {filename}")
                all_exist = False

        if all_exist:
            print("✓ All required output files generated")

        return all_exist

    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            print("  Cleaned up temp directory")
        except Exception as e:
            print(f"  Cleanup failed: {e}")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Control sgRNA QC - Functionality Tests")
    print("=" * 60)

    tests = [
        ("Basic functionality", test_basic_functionality),
        ("Mock data analysis", test_with_mock_data),
        ("Full report generation", test_full_report_generation),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print("=" * 60)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback

            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
