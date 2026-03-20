"""
Quick test of spike evaluation functions (without pypipegraph).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crisprscreens.services.spike_evaluation import (
    classify_spike_genes,
    calculate_precision_recall,
    calculate_separation_metrics,
    calculate_ranking_power,
    calculate_auc_metrics,
    calculate_spike_consistency,
    evaluate_mageck_result,
    evaluate_multiple_mageck_results,
    rank_mageck_methods,
)


def create_mock_mageck_output(seed=42):
    """Create a mock MAGeCK gene summary for testing."""
    np.random.seed(seed)
    
    genes = []
    
    # Add spike-ins
    for group in ["POS", "NEG", "NEUTRAL"]:
        for num in range(1, 21):
            genes.append(f"SPIKE_{group}_G{num:03d}")
    
    # Add some random genes
    for i in range(100):
        genes.append(f"GENE_{i:04d}")
    
    n_genes = len(genes)
    
    # Simulate scores based on spike type
    def get_lfc(gene):
        if "SPIKE_NEG" in gene:
            return np.random.normal(-2.5, 0.5)  # Depleted
        elif "SPIKE_POS" in gene:
            return np.random.normal(2.0, 0.5)   # Enriched
        elif "SPIKE_NEUTRAL" in gene:
            return np.random.normal(0, 0.3)     # No change
        else:
            return np.random.normal(0, 1.5)     # Random
    
    def get_fdr(lfc):
        # More extreme LFC -> lower FDR
        abs_lfc = abs(lfc)
        if abs_lfc > 2:
            return np.random.uniform(0, 0.01)
        elif abs_lfc > 1:
            return np.random.uniform(0.01, 0.1)
        else:
            return np.random.uniform(0.1, 1.0)
    
    lfcs = [get_lfc(g) for g in genes]
    fdrs = [get_fdr(lfc) for lfc in lfcs]
    
    # Create ranks
    sorted_indices = np.argsort([-abs(lfc) for lfc in lfcs])
    ranks = np.empty(n_genes, dtype=int)
    ranks[sorted_indices] = np.arange(n_genes) + 1
    
    df = pd.DataFrame({
        "id": genes,
        "neg|lfc": lfcs,
        "neg|fdr": fdrs,
        "neg|rank": ranks,
        "neg|score": [-np.log10(max(fdr, 1e-10)) for fdr in fdrs],
        "pos|lfc": lfcs,
        "pos|fdr": fdrs,
        "pos|rank": ranks,
        "pos|score": [-np.log10(max(fdr, 1e-10)) for fdr in fdrs],
    })
    
    return df


def test_classify():
    """Test gene classification."""
    print("\n" + "="*80)
    print("TEST: classify_spike_genes")
    print("="*80)
    
    genes = pd.Series([
        "SPIKE_POS_G001",
        "SPIKE_NEG_G012",
        "SPIKE_NEUTRAL_G020",
        "ACTB",
        "TP53",
    ])
    
    classes = classify_spike_genes(genes)
    print(f"\nInput genes:\n{genes.tolist()}")
    print(f"\nClassification:\n{classes.tolist()}")
    
    assert classes.tolist() == ["POS", "NEG", "NEUTRAL", "OTHER", "OTHER"]
    print("\n✓ Classification test passed")


def test_precision_recall():
    """Test precision/recall calculation."""
    print("\n" + "="*80)
    print("TEST: calculate_precision_recall")
    print("="*80)
    
    df = create_mock_mageck_output(seed=42)
    
    metrics = calculate_precision_recall(
        df=df,
        fdr_col="neg|fdr",
        lfc_col="neg|lfc",
        gene_col="id",
        fdr_threshold=0.05,
        lfc_threshold=1.0,
    )
    
    print("\nMetrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
    
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    print("\n✓ Precision/Recall test passed")


def test_separation():
    """Test separation metrics."""
    print("\n" + "="*80)
    print("TEST: calculate_separation_metrics")
    print("="*80)
    
    df = create_mock_mageck_output(seed=42)
    
    metrics = calculate_separation_metrics(
        df=df,
        lfc_col="neg|lfc",
        fdr_col="neg|fdr",
        gene_col="id",
    )
    
    print("\nSeparation metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    assert "neg_vs_neutral_cohens_d" in metrics
    assert "neg_vs_neutral_pvalue" in metrics
    print("\n✓ Separation test passed")


def test_ranking():
    """Test ranking power."""
    print("\n" + "="*80)
    print("TEST: calculate_ranking_power")
    print("="*80)
    
    df = create_mock_mageck_output(seed=42)
    
    metrics = calculate_ranking_power(
        df=df,
        rank_col="neg|rank",
        gene_col="id",
    )
    
    print("\nRanking metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    assert "median_rank" in metrics
    assert "aucc" in metrics
    print("\n✓ Ranking power test passed")


def test_auc():
    """Test AUC metrics."""
    print("\n" + "="*80)
    print("TEST: calculate_auc_metrics")
    print("="*80)
    
    df = create_mock_mageck_output(seed=42)
    
    metrics = calculate_auc_metrics(
        df=df,
        score_col="neg|score",
        fdr_col="neg|fdr",
        gene_col="id",
    )
    
    print("\nAUC metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    
    if "auc_roc" in metrics:
        assert 0 <= metrics["auc_roc"] <= 1
    print("\n✓ AUC test passed")


def test_consistency():
    """Test consistency metrics."""
    print("\n" + "="*80)
    print("TEST: calculate_spike_consistency")
    print("="*80)
    
    df = create_mock_mageck_output(seed=42)
    
    metrics = calculate_spike_consistency(
        df=df,
        lfc_col="neg|lfc",
        gene_col="id",
    )
    
    print("\nConsistency metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    assert "neg_cv" in metrics
    assert "neg_std_lfc" in metrics
    print("\n✓ Consistency test passed")


def test_full_evaluation():
    """Test full evaluation pipeline."""
    print("\n" + "="*80)
    print("TEST: Full evaluation pipeline")
    print("="*80)
    
    # Create mock files
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock MAGeCK outputs
        methods = {}
        for method_name in ["Method_A", "Method_B", "Method_C"]:
            df = create_mock_mageck_output(seed=hash(method_name) % 10000)
            filepath = tmpdir / f"{method_name}.tsv"
            df.to_csv(filepath, sep="\t", index=False)
            methods[method_name] = filepath
        
        # Evaluate
        eval_df = evaluate_multiple_mageck_results(
            results_dict=methods,
            direction="neg",
            fdr_threshold=0.05,
            lfc_threshold=1.0,
        )
        
        print(f"\nEvaluated {len(eval_df)} methods")
        print("\nColumns:")
        print(eval_df.columns.tolist())
        
        # Rank
        ranked_df = rank_mageck_methods(eval_df)
        
        print("\n\nRanked methods:")
        print(ranked_df[["rank", "comparison", "composite_score", "f1", "precision", "recall"]].to_string())
        
        assert len(ranked_df) == 3
        assert "composite_score" in ranked_df.columns
        assert ranked_df["rank"].tolist() == [1, 2, 3]
        
        print("\n✓ Full evaluation test passed")


def test_combine_both_when_both_present_small(tmp_path):
    # small test for combine behavior
    df = pd.DataFrame(
        {
            "id": ["SPIKE_POS_1", "SPIKE_NEG_1", "SPIKE_NEUTRAL_1"],
            "pos|fdr": [0.01, 1.0, 1.0],
            "pos|lfc": [2.0, 0.0, 0.0],
            "pos|rank": [1, 3, 2],
            "pos|score": [10, 0, 0],
            "neg|fdr": [1.0, 0.01, 1.0],
            "neg|lfc": [0.0, -2.0, 0.0],
            "neg|rank": [3, 1, 2],
            "neg|score": [0, 10, 0],
        }
    )
    p = tmp_path / "both_small.tsv"
    df.to_csv(p, sep="\t", index=False)

    res = evaluate_multiple_mageck_results({"cmp_small": p}, combine_directions=True)

    assert set(res["direction"].tolist()).issuperset({"pos", "neg", "both"})

    both = res[res["direction"] == "both"].iloc[0]
    assert both["n_expected_hits"] == 2
    assert pytest.approx(both["precision"]) == 1.0
    assert pytest.approx(both["recall"]) == 1.0
    assert pytest.approx(both["f1"]) == 1.0


def test_combined_behavior_only_pos(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["SPIKE_POS_1", "SPIKE_POS_2", "SPIKE_NEUTRAL_1"],
            "pos|fdr": [0.01, 0.02, 1.0],
            "pos|lfc": [2.0, 1.5, 0.0],
            "pos|rank": [1, 2, 3],
            "pos|score": [10, 9, 0],
            "neg|fdr": [1.0, 1.0, 1.0],
            "neg|lfc": [0.0, 0.0, 0.0],
            "neg|rank": [3, 4, 5],
            "neg|score": [0, 0, 0],
        }
    )
    p = tmp_path / "only_pos_small.tsv"
    df.to_csv(p, sep="\t", index=False)

    res = evaluate_multiple_mageck_results({"cmp_pos": p}, combine_directions=True)

    both = res[res["direction"] == "both"].iloc[0]
    pos = res[res["direction"] == "pos"].iloc[0]

    assert both["n_expected_hits"] == pos["n_expected_hits"]
    assert pytest.approx(both["precision"]) == pytest.approx(pos["precision"])
    assert pytest.approx(both["recall"]) == pytest.approx(pos["recall"])


def test_combined_behavior_only_neg(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["SPIKE_NEG_1", "SPIKE_NEG_2", "SPIKE_NEUTRAL_1"],
            "pos|fdr": [1.0, 1.0, 1.0],
            "pos|lfc": [0.0, 0.0, 0.0],
            "pos|rank": [3, 4, 5],
            "pos|score": [0, 0, 0],
            "neg|fdr": [0.01, 0.02, 1.0],
            "neg|lfc": [-2.0, -1.5, 0.0],
            "neg|rank": [1, 2, 3],
            "neg|score": [10, 9, 0],
        }
    )
    p = tmp_path / "only_neg_small.tsv"
    df.to_csv(p, sep="\t", index=False)

    res = evaluate_multiple_mageck_results({"cmp_neg": p}, combine_directions=True)

    both = res[res["direction"] == "both"].iloc[0]
    neg = res[res["direction"] == "neg"].iloc[0]

    assert both["n_expected_hits"] == neg["n_expected_hits"]
    assert pytest.approx(both["precision"]) == pytest.approx(neg["precision"])
    assert pytest.approx(both["recall"]) == pytest.approx(neg["recall"])


if __name__ == "__main__":

    print("\n" + "="*80)
    print("SPIKE-IN EVALUATION TEST SUITE")
    print("="*80)
    
    test_classify()
    test_precision_recall()
    test_separation()
    test_ranking()
    test_auc()
    test_consistency()
    test_full_evaluation()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80 + "\n")
