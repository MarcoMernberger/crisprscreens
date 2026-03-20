"""
Test module for MAGeCK method comparison functionality.

This test uses mock data to verify that all comparison functions work correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pytest


def create_mock_count_table(tmp_dir, n_genes=100, n_samples=6):
    """Create a mock MAGeCK count table."""
    sgrnas = []
    genes = []
    for i in range(n_genes):
        gene = f"GENE{i:04d}"
        for j in range(6):  # 6 sgRNAs per gene
            sgrnas.append(f"sgRNA_{i:04d}_{j}")
            genes.append(gene)

    data = {
        "sgRNA": sgrnas,
        "Gene": genes,
    }

    # Add sample columns
    sample_names = [
        "Total_Rep1",
        "Total_Rep2",
        "Total_Rep3",
        "Sort_Rep1",
        "Sort_Rep2",
        "Sort_Rep3",
    ]

    np.random.seed(42)
    for sample in sample_names:
        # Simulate counts
        data[sample] = np.random.poisson(100, len(sgrnas))

    df = pd.DataFrame(data)
    outfile = Path(tmp_dir) / "mock_counts.txt"
    df.to_csv(outfile, sep="\t", index=False)
    return outfile


def create_mock_gene_summary(tmp_dir, n_genes=100):
    """Create a mock MAGeCK gene summary."""
    np.random.seed(42)

    data = {
        "id": [f"GENE{i:04d}" for i in range(n_genes)],
        "neg|score": np.random.uniform(0, 1, n_genes),
        "neg|fdr": np.random.uniform(0, 0.5, n_genes),
        "pos|score": np.random.uniform(0, 1, n_genes),
        "pos|fdr": np.random.uniform(0, 0.5, n_genes),
    }

    df = pd.DataFrame(data)
    df = df.sort_values("neg|fdr")

    outfile = Path(tmp_dir) / "mock_gene_summary.tsv"
    df.to_csv(outfile, sep="\t", index=False)
    return outfile


def create_mock_sgrna_summary(tmp_dir, n_genes=100):
    """Create a mock MAGeCK sgRNA summary."""
    sgrnas = []
    genes = []
    lfcs = []
    fdrs = []

    np.random.seed(42)

    for i in range(n_genes):
        gene = f"GENE{i:04d}"
        # Simulate 6 sgRNAs per gene with some consistency
        base_lfc = np.random.normal(0, 1)
        for j in range(6):
            sgrnas.append(f"sgRNA_{i:04d}_{j}")
            genes.append(gene)
            # Add noise to base LFC
            lfcs.append(base_lfc + np.random.normal(0, 0.3))
            fdrs.append(np.random.uniform(0, 0.5))

    data = {
        "sgrna": sgrnas,
        "Gene": genes,
        "LFC": lfcs,
        "FDR": fdrs,
    }

    df = pd.DataFrame(data)
    outfile = Path(tmp_dir) / "mock_sgrna_summary.txt"
    df.to_csv(outfile, sep="\t", index=False)
    return outfile


def create_mock_control_sgrnas(tmp_dir, n_controls=20):
    """Create a mock control sgRNAs file."""
    controls = [f"Control_{i:03d}" for i in range(n_controls)]
    outfile = Path(tmp_dir) / "mock_controls.txt"
    with open(outfile, "w") as f:
        f.write("\n".join(controls))
    return outfile


def test_imports():
    """Test that all modules can be imported."""
    from crisprscreens.core.method_comparison import (
        get_top_n_genes,
        compute_rank_correlation,
        compute_overlap,
        analyze_sgrna_coherence,
        analyze_control_false_positives,
    )

    from crisprscreens.jobs.method_comparison_jobs import (
        sgrna_coherence_job,
        control_false_positive_job,
    )

    print("✓ All imports successful")


def test_get_top_n_genes():
    """Test get_top_n_genes function."""
    from crisprscreens.core.method_comparison import get_top_n_genes

    with tempfile.TemporaryDirectory() as tmp_dir:
        gene_summary = create_mock_gene_summary(tmp_dir, n_genes=100)

        # Test with default parameters
        top_genes = get_top_n_genes(gene_summary, n=50)

        assert isinstance(top_genes, set)
        assert len(top_genes) <= 50

        print(f"✓ get_top_n_genes returned {len(top_genes)} genes")


def test_compute_overlap():
    """Test Jaccard index computation."""
    from crisprscreens.core.method_comparison import compute_overlap

    set1 = {"A", "B", "C", "D"}
    set2 = {"B", "C", "D", "E"}

    jaccard = compute_overlap(set1, set2)

    # Intersection: {B, C, D} = 3
    # Union: {A, B, C, D, E} = 5
    # Expected: 3/5 = 0.6
    assert abs(jaccard - 0.6) < 0.01

    print(f"✓ compute_overlap returned {jaccard:.2f} (expected 0.60)")


def test_compute_rank_correlation():
    """Test Spearman rank correlation."""
    from crisprscreens.core.method_comparison import compute_rank_correlation

    with tempfile.TemporaryDirectory() as tmp_dir:
        gs1 = create_mock_gene_summary(tmp_dir, n_genes=100)
        gs2 = create_mock_gene_summary(tmp_dir, n_genes=100)

        corr, pval = compute_rank_correlation(gs1, gs2, gene_col="id")

        assert isinstance(corr, float)
        assert isinstance(pval, float)
        assert -1 <= corr <= 1

        print(f"✓ compute_rank_correlation returned {corr:.3f} (p={pval:.3f})")


def test_analyze_sgrna_coherence():
    """Test sgRNA coherence analysis."""
    from crisprscreens.core.method_comparison import analyze_sgrna_coherence

    with tempfile.TemporaryDirectory() as tmp_dir:
        gene_summary = create_mock_gene_summary(tmp_dir, n_genes=100)
        sgrna_summary = create_mock_sgrna_summary(tmp_dir, n_genes=100)

        coherence_df = analyze_sgrna_coherence(
            gene_summary=gene_summary,
            sgrna_summary=sgrna_summary,
            top_n=20,
            gene_col="id",
            sgrna_gene_col="Gene",
            sgrna_col="sgrna",
            lfc_col="LFC",
            fdr_col="FDR",
        )

        assert isinstance(coherence_df, pd.DataFrame)
        assert len(coherence_df) > 0
        assert "gene" in coherence_df.columns
        assert "direction_consistency" in coherence_df.columns

        print(f"✓ analyze_sgrna_coherence returned {len(coherence_df)} genes")
        print(
            f"  Mean direction consistency: {coherence_df['direction_consistency'].mean():.2f}"
        )


def test_analyze_control_false_positives():
    """Test control false-positive analysis."""
    from crisprscreens.core.method_comparison import (
        analyze_control_false_positives,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create gene summary with some control genes
        np.random.seed(42)
        genes = [f"GENE{i:04d}" for i in range(80)]
        controls = [f"Control_{i:03d}" for i in range(20)]
        all_genes = genes + controls

        data = {
            "id": all_genes,
            "neg|score": np.random.uniform(0, 1, len(all_genes)),
            "neg|fdr": np.random.uniform(0, 0.5, len(all_genes)),
        }
        df = pd.DataFrame(data).sort_values("neg|fdr")

        gene_summary = Path(tmp_dir) / "gene_summary_with_controls.tsv"
        df.to_csv(gene_summary, sep="\t", index=False)

        control_sgrnas = controls

        fp_df = analyze_control_false_positives(
            gene_summary=gene_summary,
            control_sgrnas=control_sgrnas,
            top_n_list=[50, 100],
            gene_col="id",
        )

        assert isinstance(fp_df, pd.DataFrame)
        assert len(fp_df) == 2  # Two top-N values
        assert "n_controls" in fp_df.columns
        assert "fraction_controls" in fp_df.columns

        print(f"✓ analyze_control_false_positives completed")
        print(f"  Controls in Top 50: {fp_df.iloc[0]['n_controls']}")
        print(f"  Controls in Top 100: {fp_df.iloc[1]['n_controls']}")


def test_create_permuted_count_table():
    """Test permuted count table creation."""
    from crisprscreens.core.method_comparison import (
        create_permuted_count_table,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        count_table = create_mock_count_table(tmp_dir, n_genes=20, n_samples=6)

        # Read original
        df_orig = pd.read_csv(count_table, sep="\t")

        # Create permuted version
        perm_file = Path(tmp_dir) / "permuted_counts.txt"
        sample_columns = ["Total_Rep1", "Total_Rep2", "Total_Rep3"]

        create_permuted_count_table(
            count_table=count_table,
            output_file=perm_file,
            sample_columns=sample_columns,
            permutation_type="within_sample",
        )

        # Read permuted
        df_perm = pd.read_csv(perm_file, sep="\t")

        assert len(df_orig) == len(df_perm)
        assert set(df_orig.columns) == set(df_perm.columns)

        # Check that values were actually permuted
        for col in sample_columns:
            # Sums should be the same
            assert df_orig[col].sum() == df_perm[col].sum()
            # But individual values should differ
            n_different = (df_orig[col] != df_perm[col]).sum()
            assert n_different > 0, f"Column {col} was not permuted"

        print(f"✓ create_permuted_count_table completed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING METHOD COMPARISON TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Get Top-N Genes", test_get_top_n_genes),
        ("Compute Overlap", test_compute_overlap),
        ("Rank Correlation", test_compute_rank_correlation),
        ("sgRNA Coherence", test_analyze_sgrna_coherence),
        ("Control False-Positives", test_analyze_control_false_positives),
        ("Permuted Count Table", test_create_permuted_count_table),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n[{name}]")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
