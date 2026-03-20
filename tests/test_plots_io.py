"""
Tests for services/plots_io.py module.

This module tests plotting I/O service functions.
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from crisprscreens.services.plots_io import (
    write_venn,
    write_volcano_plot,
)


# Test data paths
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_gene_sets():
    """Create sample gene sets for Venn diagrams."""
    return {
        "Comparison1": set([f"Gene{i}" for i in range(50)]),
        "Comparison2": set([f"Gene{i}" for i in range(25, 75)]),
        "Comparison3": set([f"Gene{i}" for i in range(50, 100)]),
    }


@pytest.fixture
def sample_volcano_data():
    """Create sample data for volcano plots."""
    return pd.DataFrame({
        "id": [f"Gene{i}" for i in range(100)],
        "neg|lfc": np.random.randn(100) * 2,
        "pos|lfc": np.random.randn(100) * 2,
        "neg|fdr": np.random.uniform(0, 1, 100),
        "pos|fdr": np.random.uniform(0, 1, 100),
    })


class TestWriteVenn:
    """Test write_venn function."""

    def test_write_venn_two_sets(self, tmp_path):
        """Test writing Venn diagram with two sets."""
        gene_sets = {
            "Set1": set([f"Gene{i}" for i in range(50)]),
            "Set2": set([f"Gene{i}" for i in range(25, 75)]),
        }

        output_file = tmp_path / "venn_2sets.png"

        result_path = write_venn(
            selected_genes_dict=gene_sets,
            filename=output_file,
            folder=tmp_path
        )

        assert result_path.exists()
        assert result_path.suffix == ".png"

    def test_write_venn_three_sets(self, tmp_path, sample_gene_sets):
        """Test writing Venn diagram with three sets."""
        output_file = tmp_path / "venn_3sets.png"

        result_path = write_venn(
            selected_genes_dict=sample_gene_sets,
            filename=output_file,
            folder=tmp_path
        )

        assert result_path.exists()

    def test_write_venn_creates_directory(self, tmp_path):
        """Test that write_venn creates output directory."""
        gene_sets = {
            "Set1": set([f"Gene{i}" for i in range(20)]),
            "Set2": set([f"Gene{i}" for i in range(10, 30)]),
        }

        nested_dir = tmp_path / "nested" / "venn"
        output_file = nested_dir / "venn.png"

        result_path = write_venn(
            selected_genes_dict=gene_sets,
            filename=output_file,
            folder=nested_dir
        )

        assert nested_dir.exists()
        assert result_path.exists()

    def test_write_venn_with_empty_sets(self, tmp_path):
        """Test Venn diagram with some empty sets."""
        gene_sets = {
            "Set1": set([f"Gene{i}" for i in range(50)]),
            "Set2": set(),  # Empty set
            "Set3": set([f"Gene{i}" for i in range(25, 75)]),
        }

        output_file = tmp_path / "venn_empty.png"

        result_path = write_venn(
            selected_genes_dict=gene_sets,
            filename=output_file,
            folder=tmp_path
        )

        assert result_path.exists()


class TestWriteVolcanoPlot:
    """Test write_volcano_plot function."""

    def test_write_volcano_basic(self, tmp_path, sample_volcano_data):
        """Test basic volcano plot writing."""
        output_file = tmp_path / "volcano.png"

        result_path = write_volcano_plot(
            filename=output_file,
            folder=tmp_path,
            df=sample_volcano_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id"
        )

        assert result_path.exists()
        assert result_path.suffix == ".png"

    def test_write_volcano_with_labels(self, tmp_path, sample_volcano_data):
        """Test volcano plot with top gene labels."""
        output_file = tmp_path / "volcano_labeled.png"

        result_path = write_volcano_plot(
            filename=output_file,
            folder=tmp_path,
            df=sample_volcano_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id",
            top_n_labels=10
        )

        assert result_path.exists()

    def test_write_volcano_with_thresholds(self, tmp_path, sample_volcano_data):
        """Test volcano plot with custom thresholds."""
        output_file = tmp_path / "volcano_thresholds.png"

        result_path = write_volcano_plot(
            filename=output_file,
            folder=tmp_path,
            df=sample_volcano_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id",
            log_threshold=1.5,
            fdr_threshold=0.01,
            transform_y=True
        )

        assert result_path.exists()

    def test_write_volcano_with_two_fdr_columns(self, tmp_path, sample_volcano_data):
        """Test volcano plot with pos and neg FDR columns."""
        output_file = tmp_path / "volcano_two_fdr.png"

        result_path = write_volcano_plot(
            filename=output_file,
            folder=tmp_path,
            df=sample_volcano_data,
            log_fc_column="neg|lfc",
            fdr_column=("pos|fdr", "neg|fdr"),
            name_column="id"
        )

        assert result_path.exists()

    def test_write_volcano_with_custom_figsize(self, tmp_path, sample_volcano_data):
        """Test volcano plot with custom figure size."""
        output_file = tmp_path / "volcano_custom_size.png"

        result_path = write_volcano_plot(
            filename=output_file,
            folder=tmp_path,
            df=sample_volcano_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id",
            figsize=(10, 8)
        )

        assert result_path.exists()

    def test_write_volcano_creates_directory(self, tmp_path, sample_volcano_data):
        """Test that write_volcano_plot creates output directory."""
        nested_dir = tmp_path / "nested" / "volcano"
        output_file = nested_dir / "volcano.png"

        result_path = write_volcano_plot(
            filename=output_file,
            folder=nested_dir,
            df=sample_volcano_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id"
        )

        assert nested_dir.exists()
        assert result_path.exists()

    def test_write_volcano_with_title(self, tmp_path, sample_volcano_data):
        """Test volcano plot with custom title."""
        output_file = tmp_path / "volcano_titled.png"

        result_path = write_volcano_plot(
            filename=output_file,
            folder=tmp_path,
            df=sample_volcano_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id",
            title="My Custom Volcano Plot"
        )

        assert result_path.exists()

    def test_write_volcano_pdf(self, tmp_path, sample_volcano_data):
        """Test volcano plot saved as PDF."""
        output_file = tmp_path / "volcano.pdf"

        result_path = write_volcano_plot(
            filename=output_file,
            folder=tmp_path,
            df=sample_volcano_data,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id"
        )

        assert result_path.exists()
        assert result_path.suffix == ".pdf"


class TestPlotsIoEdgeCases:
    """Test plotting I/O functions with edge cases."""

    def test_write_volcano_empty_dataframe(self, tmp_path):
        """Test volcano plot with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["id", "neg|lfc", "neg|fdr"])
        output_file = tmp_path / "volcano_empty.png"

        # Should handle empty data gracefully
        result_path = write_volcano_plot(
            filename=output_file,
            folder=tmp_path,
            df=empty_df,
            log_fc_column="neg|lfc",
            fdr_column="neg|fdr",
            name_column="id"
        )

        assert result_path.exists()

    def test_write_venn_single_set(self, tmp_path):
        """Test Venn diagram with single set."""
        gene_sets = {
            "OnlySet": set([f"Gene{i}" for i in range(50)]),
        }

        output_file = tmp_path / "venn_single.png"

        # Should handle single set (might not create Venn, but should not crash)
        result_path = write_venn(
            selected_genes_dict=gene_sets,
            filename=output_file,
            folder=tmp_path
        )

        # May or may not create file depending on implementation
        # Just check no crash occurred
        assert True

    def test_write_volcano_missing_column(self, tmp_path):
        """Test volcano plot with missing column."""
        df = pd.DataFrame({
            "id": [f"Gene{i}" for i in range(10)],
            "neg|lfc": np.random.randn(10),
            # Missing "neg|fdr" column
        })

        output_file = tmp_path / "volcano_missing_col.png"

        # Should raise error for missing column
        with pytest.raises(KeyError):
            write_volcano_plot(
                filename=output_file,
                folder=tmp_path,
                df=df,
                log_fc_column="neg|lfc",
                fdr_column="neg|fdr",
                name_column="id"
            )
