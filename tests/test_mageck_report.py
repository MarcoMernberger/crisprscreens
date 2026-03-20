"""
Tests for core/mageck_report.py module.

This module tests the comprehensive MAGeCK reporting functionality.
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from crisprscreens.core.mageck_report import generate_mageck_report


# Test data paths
DATA_DIR = Path(__file__).parent / "data"
COUNT_FILE = DATA_DIR / "brunello.count.tsv"


@pytest.fixture
def sample_mle_summary(tmp_path):
    """Create sample MLE gene summary file."""
    data = pd.DataFrame({
        "Gene": [f"Gene{i}" for i in range(100)],
        "Sorted1|beta": np.random.randn(100),
        "Sorted1|z": np.random.randn(100),
        "Sorted1|wald-fdr": np.random.uniform(0, 1, 100),
        "Sorted1|wald-p-value": np.random.uniform(0, 1, 100),
        "Sorted1|beta-se": np.abs(np.random.randn(100)) * 0.5,
        "Sorted2|beta": np.random.randn(100),
        "Sorted2|z": np.random.randn(100),
        "Sorted2|wald-fdr": np.random.uniform(0, 1, 100),
        "Sorted2|wald-p-value": np.random.uniform(0, 1, 100),
        "Sorted2|beta-se": np.abs(np.random.randn(100)) * 0.5,
    })

    output_file = tmp_path / "mle_gene_summary.txt"
    data.to_csv(output_file, sep="\t", index=False)
    return output_file


@pytest.fixture
def sample_rra_summary(tmp_path):
    """Create sample RRA gene summary file."""
    data = pd.DataFrame({
        "id": [f"Gene{i}" for i in range(100)],
        "neg|lfc": np.random.randn(100),
        "pos|lfc": np.random.randn(100),
        "neg|fdr": np.random.uniform(0, 1, 100),
        "pos|fdr": np.random.uniform(0, 1, 100),
        "neg|rank": np.arange(100),
        "pos|rank": np.arange(100),
        "neg|score": np.random.uniform(0, 1, 100),
        "pos|score": np.random.uniform(0, 1, 100),
    })

    output_file = tmp_path / "rra_gene_summary.txt"
    data.to_csv(output_file, sep="\t", index=False)
    return output_file


@pytest.fixture
def sample_pathway_enrichment(tmp_path):
    """Create sample pathway enrichment file."""
    data = pd.DataFrame({
        "pathway": [f"Pathway{i}" for i in range(20)],
        "nes": np.random.randn(20) * 2,
        "fdr": np.random.uniform(0, 0.5, 20),
        "size": np.random.randint(10, 200, 20),
        "pvalue": np.random.uniform(0, 0.1, 20),
    })

    output_file = tmp_path / "pathway_enrichment.tsv"
    data.to_csv(output_file, sep="\t", index=False)
    return output_file


class TestGenerateMageckReportMLE:
    """Test generate_mageck_report with MLE data."""

    def test_mle_minimal_report(self, tmp_path, sample_mle_summary):
        """Test minimal MLE report generation."""
        output_dir = tmp_path / "report_mle_minimal"

        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=sample_mle_summary,
            readout="minimal",
            effect_cols=["Sorted1|beta", "Sorted2|beta"],
            gene_col="Gene",
            fdr_threshold=0.25
        )

        # Check that result dict has expected keys
        assert "plots" in result
        assert "metrics" in result
        assert "executive_summary" in result

        # Check that minimal plots were created
        plots = result["plots"]
        assert "volcano_plot" in plots or "volcano_plots" in plots
        assert "effect_size_vs_reproducibility" in plots

        # Check that output directory was created
        assert output_dir.exists()

    def test_mle_full_report(self, tmp_path, sample_mle_summary):
        """Test full MLE report generation."""
        output_dir = tmp_path / "report_mle_full"

        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=sample_mle_summary,
            readout="full",
            effect_cols=["Sorted1|beta", "Sorted2|beta"],
            gene_col="Gene",
            fdr_threshold=0.25,
            top_n=20
        )

        assert "plots" in result
        assert "metrics" in result

        # Full report should have more plots
        plots = result["plots"]
        assert len(plots) > 0

    def test_mle_report_with_pathway_enrichment(
        self, tmp_path, sample_mle_summary, sample_pathway_enrichment
    ):
        """Test MLE report with pathway enrichment."""
        output_dir = tmp_path / "report_mle_pathway"

        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=sample_mle_summary,
            pathway_enrichment_path=sample_pathway_enrichment,
            readout="full",
            effect_cols=["Sorted1|beta", "Sorted2|beta"],
            gene_col="Gene"
        )

        assert "plots" in result
        plots = result["plots"]

        # Should include pathway enrichment plot
        assert "pathway_enrichment" in plots or "pathway_enrichment_summary" in plots


class TestGenerateMageckReportRRA:
    """Test generate_mageck_report with RRA data."""

    def test_rra_minimal_report(self, tmp_path, sample_rra_summary):
        """Test minimal RRA report generation."""
        output_dir = tmp_path / "report_rra_minimal"

        result = generate_mageck_report(
            output_dir=output_dir,
            rra_summary_path=sample_rra_summary,
            readout="minimal",
            gene_col="id",
            fdr_threshold=0.25
        )

        assert "plots" in result
        assert "metrics" in result
        assert output_dir.exists()

    def test_rra_full_report(self, tmp_path, sample_rra_summary):
        """Test full RRA report generation."""
        output_dir = tmp_path / "report_rra_full"

        result = generate_mageck_report(
            output_dir=output_dir,
            rra_summary_path=sample_rra_summary,
            readout="full",
            gene_col="id",
            top_n=30
        )

        assert "plots" in result
        assert len(result["plots"]) > 0


class TestGenerateMageckReportCombined:
    """Test generate_mageck_report with both RRA and MLE data."""

    def test_combined_rra_mle_report(
        self, tmp_path, sample_rra_summary, sample_mle_summary
    ):
        """Test report with both RRA and MLE results."""
        output_dir = tmp_path / "report_combined"

        result = generate_mageck_report(
            output_dir=output_dir,
            rra_summary_path=sample_rra_summary,
            mle_summary_path=sample_mle_summary,
            readout="full",
            effect_cols=["Sorted1|beta", "Sorted2|beta"],
            gene_col="Gene"
        )

        assert "plots" in result
        plots = result["plots"]

        # Combined report should include rank stability plot
        # (comparing RRA and MLE rankings)
        assert len(plots) > 0


class TestGenerateMageckReportEdgeCases:
    """Test generate_mageck_report with edge cases."""

    def test_empty_gene_summary(self, tmp_path):
        """Test report with empty gene summary."""
        empty_file = tmp_path / "empty_summary.txt"
        pd.DataFrame(columns=["Gene", "Sorted1|beta", "Sorted1|wald-fdr"]).to_csv(
            empty_file, sep="\t", index=False
        )

        output_dir = tmp_path / "report_empty"

        # Should handle empty data gracefully
        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=empty_file,
            readout="minimal",
            gene_col="Gene"
        )

        assert "plots" in result
        assert "metrics" in result

    def test_missing_effect_columns(self, tmp_path, sample_mle_summary):
        """Test report when specified effect columns don't exist."""
        output_dir = tmp_path / "report_missing_cols"

        # Request non-existent columns
        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=sample_mle_summary,
            readout="minimal",
            effect_cols=["NonExistent|beta"],
            gene_col="Gene"
        )

        # Should still produce some results
        assert "plots" in result or "metrics" in result

    def test_no_significant_genes(self, tmp_path):
        """Test report when no genes are significant."""
        data = pd.DataFrame({
            "Gene": [f"Gene{i}" for i in range(100)],
            "Sorted1|beta": np.random.randn(100) * 0.1,  # Small effects
            "Sorted1|wald-fdr": np.random.uniform(0.5, 1.0, 100),  # High FDRs
        })

        no_sig_file = tmp_path / "no_sig_summary.txt"
        data.to_csv(no_sig_file, sep="\t", index=False)

        output_dir = tmp_path / "report_no_sig"

        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=no_sig_file,
            readout="minimal",
            effect_cols=["Sorted1|beta"],
            gene_col="Gene",
            fdr_threshold=0.05
        )

        # Should still generate report
        assert "metrics" in result

        # Metrics should reflect no significant genes
        if "executive_summary" in result:
            summary = result["executive_summary"]
            # Could check for indicators of no significant findings


class TestGenerateMageckReportGeneSetAnalysis:
    """Test generate_mageck_report with gene set analysis."""

    def test_report_with_gene_sets(self, tmp_path, sample_mle_summary):
        """Test report with custom gene sets."""
        output_dir = tmp_path / "report_gene_sets"

        # Define some gene sets
        gene_sets = {
            "GeneSet1": [f"Gene{i}" for i in range(10)],
            "GeneSet2": [f"Gene{i}" for i in range(10, 25)],
            "GeneSet3": [f"Gene{i}" for i in range(25, 40)],
        }

        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=sample_mle_summary,
            readout="full",
            effect_cols=["Sorted1|beta"],
            gene_col="Gene",
            gene_sets=gene_sets
        )

        assert "plots" in result
        # Should include gene set score distribution plots
        plots = result["plots"]
        assert len(plots) > 0


class TestGenerateMageckReportMetrics:
    """Test metrics generation in reports."""

    def test_metrics_structure(self, tmp_path, sample_mle_summary):
        """Test that metrics have expected structure."""
        output_dir = tmp_path / "report_metrics"

        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=sample_mle_summary,
            readout="full",
            effect_cols=["Sorted1|beta", "Sorted2|beta"],
            gene_col="Gene",
            fdr_threshold=0.25
        )

        assert "metrics" in result
        metrics = result["metrics"]

        # Metrics should be a dictionary or DataFrame
        assert isinstance(metrics, (dict, pd.DataFrame))

    def test_executive_summary(self, tmp_path, sample_mle_summary):
        """Test executive summary generation."""
        output_dir = tmp_path / "report_exec_summary"

        result = generate_mageck_report(
            output_dir=output_dir,
            gene_summary_path=sample_mle_summary,
            readout="full",
            effect_cols=["Sorted1|beta"],
            gene_col="Gene",
            fdr_threshold=0.05,
            top_n=10
        )

        assert "executive_summary" in result
        summary = result["executive_summary"]

        # Summary should contain key information
        assert isinstance(summary, dict)
