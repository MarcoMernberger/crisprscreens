"""
Tests for core/plots.py module.

This module tests plotting functions for CRISPR screen analysis.
Since plotting functions create figures, we focus on testing that they:
1. Accept correct input formats
2. Return expected figure objects
3. Don't crash with edge cases
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from pathlib import Path
from crisprscreens.core.plots import (
    volcano_plot,
    plot_control_distribution_per_condition,
    plot_pairwise_control_shifts,
    plot_control_replicate_correlation,
    plot_ma,
    plot_ma_grid,
    plot_library_pca,
    plot_sample_correlations,
    plot_control_pca,
    plot_effect_size_vs_reproducibility,
    plot_rank_stability,
    plot_direction_consistency,
    plot_replicate_effect_heatmap,
    plot_effect_decomposition,
    plot_contrast,
    plot_pathway_enrichment_summary,
    plot_gene_set_score_distribution,
    plot_beta_vs_standard_error,
    plot_wald_z_distribution,
    plot_qq,
)


# Test data paths
DATA_DIR = Path(__file__).parent / "data"
COUNT_FILE = DATA_DIR / "brunello.count.tsv"


@pytest.fixture
def sample_rra_data():
    """Create sample RRA-style data for testing."""
    return pd.DataFrame({
        "id": [f"Gene{i}" for i in range(100)],
        "neg|lfc": np.random.randn(100),
        "pos|lfc": np.random.randn(100),
        "neg|fdr": np.random.uniform(0, 1, 100),
        "pos|fdr": np.random.uniform(0, 1, 100),
        "neg|rank": np.arange(100),
        "pos|rank": np.arange(100),
    })


@pytest.fixture
def sample_mle_data():
    """Create sample MLE-style data for testing."""
    return pd.DataFrame({
        "Gene": [f"Gene{i}" for i in range(100)],
        "Condition1|beta": np.random.randn(100),
        "Condition1|z": np.random.randn(100),
        "Condition1|wald-fdr": np.random.uniform(0, 1, 100),
        "Condition1|wald-p-value": np.random.uniform(0, 1, 100),
        "Condition2|beta": np.random.randn(100),
        "Condition2|z": np.random.randn(100),
        "Condition2|wald-fdr": np.random.uniform(0, 1, 100),
    })


@pytest.fixture
def count_df():
    """Load real count data for testing."""
    df = pd.read_csv(COUNT_FILE, sep="\t")
    # Take subset for faster tests
    return df.head(1000)


@pytest.fixture
def sample_ids():
    """Sample IDs for testing."""
    return {
        "baseline": ["Total_Rep1", "Total_Rep2", "Total_Rep3"],
        "sorted1": ["Sort1_Rep1", "Sort1_Rep2", "Sort1_Rep3"],
        "sorted2": ["Sort2_Rep1", "Sort2_Rep2", "Sort2_Rep3"],
    }


class TestVolcanoPlot:
    """Test volcano_plot function."""

    def test_volcano_basic(self, sample_rra_data):
        """Test basic volcano plot creation."""
        fig = volcano_plot(
            sample_rra_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id",
            transform_y=True
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_volcano_with_labels(self, sample_rra_data):
        """Test volcano plot with top gene labels."""
        fig = volcano_plot(
            sample_rra_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id",
            top_n_labels=10,
            transform_y=True
        )

        assert fig is not None
        plt.close(fig)

    def test_volcano_with_two_fdr_columns(self, sample_rra_data):
        """Test volcano plot with positive and negative FDR."""
        fig = volcano_plot(
            sample_rra_data,
            log_fc_column="neg|lfc",
            fdr_column=("pos|fdr", "neg|fdr"),
            name_column="id",
            transform_y=True
        )

        assert fig is not None
        plt.close(fig)

    def test_volcano_with_thresholds(self, sample_rra_data):
        """Test volcano plot with custom thresholds."""
        fig = volcano_plot(
            sample_rra_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id",
            log_threshold=1.5,
            fdr_threshold=0.01,
            transform_y=True
        )

        assert fig is not None
        plt.close(fig)


class TestControlPlots:
    """Test control sgRNA plotting functions."""

    def test_plot_control_distribution(self, count_df, sample_ids):
        """Test control distribution plotting."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
        }

        # Create simple control set
        control_sgrnas = set(count_df["sgRNA"].head(100))

        fig = plot_control_distribution_per_condition(
            count_df,
            control_sgrnas,
            sample_cols=sample_cols,
            conditions_dict=conditions_dict,
            baseline_condition="Total"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_control_replicate_correlation(self, count_df, sample_ids):
        """Test control replicate correlation plotting."""
        # Create simple control set
        control_sgrnas = set(count_df["sgRNA"].head(100))

        fig = plot_control_replicate_correlation(
            count_df,
            control_sgrnas,
            condition_samples=sample_ids["baseline"]
        )

        assert fig is not None
        plt.close(fig)


class TestMAPlots:
    """Test MA plot functions."""

    def test_plot_ma_basic(self, count_df):
        """Test basic MA plot."""
        fig = plot_ma(
            count_df,
            sample1="Total_Rep1",
            sample2="Total_Rep2",
            pseudocount=1.0
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_ma_grid(self, count_df, sample_ids):
        """Test MA plot grid."""
        fig = plot_ma_grid(
            count_df,
            samples=sample_ids["baseline"][:3],  # Use 3 samples for 3x3 grid
            pseudocount=1.0
        )

        assert fig is not None
        plt.close(fig)


class TestPCAPlots:
    """Test PCA plotting functions."""

    def test_plot_library_pca(self, count_df, sample_ids):
        """Test library PCA plotting."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]

        fig = plot_library_pca(
            count_df,
            sample_cols=sample_cols,
            log_transform=True,
            pseudocount=1.0
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_control_pca(self, count_df, sample_ids):
        """Test control PCA plotting."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        control_sgrnas = set(count_df["sgRNA"].head(100))

        fig = plot_control_pca(
            count_df,
            control_sgrnas,
            sample_cols=sample_cols,
            log_transform=True,
            pseudocount=1.0
        )

        assert fig is not None
        plt.close(fig)


class TestSampleCorrelations:
    """Test sample correlation plotting."""

    def test_plot_sample_correlations(self, count_df, sample_ids):
        """Test sample correlation heatmap."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]

        fig = plot_sample_correlations(
            count_df,
            sample_cols=sample_cols,
            log_transform=True,
            pseudocount=1.0
        )

        assert fig is not None
        plt.close(fig)


class TestMageckReportPlots:
    """Test specialized MAGeCK report plotting functions."""

    def test_plot_effect_size_vs_reproducibility(self, sample_mle_data):
        """Test effect size vs reproducibility plot."""
        fig = plot_effect_size_vs_reproducibility(
            sample_mle_data,
            effect_columns=["Condition1|beta", "Condition2|beta"],
            fdr_columns=["Condition1|wald-fdr", "Condition2|wald-fdr"],
            gene_column="Gene"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_rank_stability(self, sample_rra_data):
        """Test rank stability plot."""
        # Create second comparison data
        comparison_data = {
            "Comp1": sample_rra_data.copy(),
            "Comp2": sample_rra_data.copy(),
        }
        comparison_data["Comp2"]["neg|rank"] = np.random.permutation(100)

        fig = plot_rank_stability(
            comparison_data,
            rank_columns={"Comp1": "neg|rank", "Comp2": "neg|rank"},
            gene_column="id"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_direction_consistency(self, sample_rra_data):
        """Test direction consistency plot."""
        comparison_data = {
            "Comp1": sample_rra_data.copy(),
            "Comp2": sample_rra_data.copy(),
        }

        fig = plot_direction_consistency(
            comparison_data,
            logfc_columns={"Comp1": "neg|lfc", "Comp2": "neg|lfc"},
            gene_column="id"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_replicate_effect_heatmap(self, sample_mle_data):
        """Test replicate effect heatmap."""
        fig = plot_replicate_effect_heatmap(
            sample_mle_data,
            effect_columns=["Condition1|beta", "Condition2|beta"],
            gene_column="Gene",
            top_n=50
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_effect_decomposition(self, sample_mle_data):
        """Test effect decomposition plot."""
        fig = plot_effect_decomposition(
            sample_mle_data,
            effect_column="Condition1|beta",
            gene_column="Gene"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_contrast(self, sample_mle_data):
        """Test contrast plot."""
        fig = plot_contrast(
            sample_mle_data,
            effect1_column="Condition1|beta",
            effect2_column="Condition2|beta",
            fdr1_column="Condition1|wald-fdr",
            fdr2_column="Condition2|wald-fdr",
            gene_column="Gene"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_beta_vs_standard_error(self, sample_mle_data):
        """Test beta vs standard error plot."""
        # Add standard error column
        sample_mle_data["Condition1|beta-se"] = np.abs(sample_mle_data["Condition1|beta"]) / 2

        fig = plot_beta_vs_standard_error(
            sample_mle_data,
            beta_column="Condition1|beta",
            stderr_column="Condition1|beta-se",
            gene_column="Gene"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_wald_z_distribution(self, sample_mle_data):
        """Test Wald Z distribution plot."""
        fig = plot_wald_z_distribution(
            sample_mle_data,
            z_column="Condition1|z"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_qq(self, sample_mle_data):
        """Test QQ plot."""
        fig = plot_qq(sample_mle_data["Condition1|wald-p-value"].values)

        assert fig is not None
        plt.close(fig)


class TestPathwayPlots:
    """Test pathway enrichment plotting functions."""

    def test_plot_pathway_enrichment_summary(self):
        """Test pathway enrichment summary plot."""
        pathway_data = pd.DataFrame({
            "pathway": [f"Pathway{i}" for i in range(20)],
            "nes": np.random.randn(20) * 2,
            "fdr": np.random.uniform(0, 0.5, 20),
            "size": np.random.randint(10, 200, 20),
        })

        fig = plot_pathway_enrichment_summary(
            pathway_data,
            pathway_column="pathway",
            nes_column="nes",
            fdr_column="fdr",
            size_column="size"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_gene_set_score_distribution(self):
        """Test gene set score distribution plot."""
        # Create sample gene set data
        gene_set_data = {
            "Pathway1": np.random.randn(100),
            "Pathway2": np.random.randn(100) + 1,
            "Pathway3": np.random.randn(100) - 0.5,
        }

        fig = plot_gene_set_score_distribution(gene_set_data)

        assert fig is not None
        plt.close(fig)


class TestPlotEdgeCases:
    """Test plotting functions with edge cases."""

    def test_empty_dataframe(self):
        """Test volcano plot with empty dataframe."""
        empty_df = pd.DataFrame(columns=["id", "neg|lfc", "neg|fdr"])

        # Should handle empty data gracefully
        fig = volcano_plot(
            empty_df,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id"
        )

        assert fig is not None
        plt.close(fig)

    def test_single_gene(self):
        """Test volcano plot with single gene."""
        single_gene = pd.DataFrame({
            "id": ["Gene1"],
            "neg|lfc": [2.0],
            "neg|fdr": [0.01],
        })

        fig = volcano_plot(
            single_gene,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id"
        )

        assert fig is not None
        plt.close(fig)

    def test_all_significant(self, sample_rra_data):
        """Test volcano plot when all genes are significant."""
        sample_rra_data["neg|fdr"] = 0.001  # All significant

        fig = volcano_plot(
            sample_rra_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id",
            fdr_threshold=0.05
        )

        assert fig is not None
        plt.close(fig)
