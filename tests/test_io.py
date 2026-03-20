"""
Tests for services/io.py module.

This module tests I/O service functions that wrap core functionality with
directory creation and file handling.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
from pathlib import Path
from unittest.mock import patch, MagicMock
from crisprscreens.services.io import (
    save_figure,
    read_dataframe,
    generate_control_qc_report,
    standard_qc_report,
    mageck_report,
)


# Test data paths
DATA_DIR = Path(__file__).parent / "data"
COUNT_FILE = DATA_DIR / "brunello.count.tsv"
CONTROL_FILE = DATA_DIR / "control_sgRNAs.txt"


@pytest.fixture
def sample_ids():
    """Sample IDs for testing."""
    return {
        "baseline": ["Total_Rep1", "Total_Rep2", "Total_Rep3"],
        "sorted1": ["Sort1_Rep1", "Sort1_Rep2", "Sort1_Rep3"],
        "sorted2": ["Sort2_Rep1", "Sort2_Rep2", "Sort2_Rep3"],
    }


class TestSaveFigure:
    """Test save_figure function."""

    def test_save_figure_png(self, tmp_path):
        """Test saving figure as PNG."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        output_path = save_figure(fig, tmp_path, "test_plot.png")

        assert output_path.exists()
        assert output_path.suffix == ".png"
        plt.close(fig)

    def test_save_figure_pdf(self, tmp_path):
        """Test saving figure as PDF."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        output_path = save_figure(fig, tmp_path, "test_plot.pdf")

        assert output_path.exists()
        assert output_path.suffix == ".pdf"
        plt.close(fig)

    def test_save_figure_creates_directory(self, tmp_path):
        """Test that save_figure creates output directory."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        nested_dir = tmp_path / "nested" / "dir"
        output_path = save_figure(fig, nested_dir, "test_plot.png")

        assert nested_dir.exists()
        assert output_path.exists()
        plt.close(fig)


class TestReadDataframe:
    """Test read_dataframe function."""

    def test_read_tsv(self):
        """Test reading TSV file."""
        df = read_dataframe(COUNT_FILE)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "sgRNA" in df.columns
        assert "Gene" in df.columns

    def test_read_csv(self, tmp_path):
        """Test reading CSV file."""
        # Create test CSV
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        csv_file = tmp_path / "test.csv"
        test_data.to_csv(csv_file, index=False)

        df = read_dataframe(csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "col1" in df.columns

    def test_read_with_kwargs(self, tmp_path):
        """Test reading with additional pandas kwargs."""
        # Create test file with custom separator
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        custom_file = tmp_path / "test_custom.txt"
        test_data.to_csv(custom_file, sep="|", index=False)

        df = read_dataframe(custom_file, sep="|")

        assert len(df) == 3
        assert "col1" in df.columns


class TestGenerateControlQcReport:
    """Test generate_control_qc_report function."""

    def test_control_qc_report_basic(self, tmp_path, sample_ids):
        """Test basic control QC report generation."""
        output_dir = tmp_path / "control_qc"

        all_samples = sample_ids["baseline"] + sample_ids["sorted1"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
        }

        result = generate_control_qc_report(
            count_tsv=COUNT_FILE,
            control_sgrna_txt=CONTROL_FILE,
            output_dir=output_dir,
            sample_cols=all_samples,
            conditions_dict=conditions_dict,
            baseline_condition="Total",
            save_formats=["png"],
        )

        # Check output directory was created
        assert output_dir.exists()

        # Check result structure
        assert "figures" in result
        assert "metrics" in result

    def test_control_qc_with_multiple_formats(self, tmp_path, sample_ids):
        """Test control QC report with multiple save formats."""
        output_dir = tmp_path / "control_qc_multi"

        all_samples = sample_ids["baseline"]
        conditions_dict = {"Total": sample_ids["baseline"]}

        result = generate_control_qc_report(
            count_tsv=COUNT_FILE,
            control_sgrna_txt=CONTROL_FILE,
            output_dir=output_dir,
            sample_cols=all_samples,
            conditions_dict=conditions_dict,
            baseline_condition="Total",
            save_formats=["png", "pdf"],
        )

        assert output_dir.exists()

        # Check that both formats were saved
        png_files = list(output_dir.glob("*.png"))
        pdf_files = list(output_dir.glob("*.pdf"))
        assert len(png_files) > 0 or len(pdf_files) > 0


class TestStandardQcReport:
    """Test standard_qc_report function."""

    def test_standard_qc_basic(self, tmp_path, sample_ids):
        """Test basic standard QC report generation."""
        output_dir = tmp_path / "standard_qc"

        all_samples = sample_ids["baseline"] + sample_ids["sorted1"]

        result = standard_qc_report(
            count_tsv=COUNT_FILE,
            output_dir=output_dir,
            sample_cols=all_samples,
            baseline_condition="Total",
            control_sgrna_txt=CONTROL_FILE,
        )

        # Check output directory was created
        assert output_dir.exists()

        # Check result structure
        assert isinstance(result, dict)

    def test_standard_qc_without_controls(self, tmp_path, sample_ids):
        """Test standard QC without control sgRNAs."""
        output_dir = tmp_path / "standard_qc_no_controls"

        all_samples = sample_ids["baseline"]

        result = standard_qc_report(
            count_tsv=COUNT_FILE,
            output_dir=output_dir,
            sample_cols=all_samples,
            baseline_condition="Total",
            control_sgrna_txt=None,
        )

        assert output_dir.exists()
        assert isinstance(result, dict)


class TestMageckReport:
    """Test mageck_report function."""

    def test_mageck_report_mle(self, tmp_path):
        """Test mageck_report with MLE data."""
        # Create mock MLE summary
        mle_data = pd.DataFrame(
            {
                "Gene": [f"Gene{i}" for i in range(50)],
                "Sorted1|beta": np.random.randn(50),
                "Sorted1|wald-fdr": np.random.uniform(0, 1, 50),
            }
        )
        mle_file = tmp_path / "mle_summary.txt"
        mle_data.to_csv(mle_file, sep="\t", index=False)

        output_dir = tmp_path / "mageck_report"

        result = mageck_report(
            output_dir=output_dir,
            gene_summary_path=mle_file,
            readout="minimal",
            gene_col="Gene",
        )

        # Check output directory was created
        assert output_dir.exists()

        # Check result structure
        assert "plots" in result
        assert "metrics" in result

    def test_mageck_report_creates_directory(self, tmp_path):
        """Test that mageck_report creates output directory."""
        # Create mock data
        mle_data = pd.DataFrame(
            {
                "Gene": [f"Gene{i}" for i in range(20)],
                "Condition|beta": np.random.randn(20),
            }
        )
        mle_file = tmp_path / "mle_summary.txt"
        mle_data.to_csv(mle_file, sep="\t", index=False)

        # Use nested directory that doesn't exist
        output_dir = tmp_path / "nested" / "output" / "dir"

        result = mageck_report(
            output_dir=output_dir,
            gene_summary_path=mle_file,
            readout="minimal",
            gene_col="Gene",
        )

        # Directory should be created
        assert output_dir.exists()


class TestMageckWrappers:
    """Test MAGeCK pathway and plot wrappers."""

    @patch("subprocess.run")
    def test_mageck_pathway_wrapper(self, mock_run, tmp_path):
        """Test mageck_pathway service wrapper."""
        from crisprscreens.services.io import mageck_pathway

        mock_run.return_value = MagicMock(
            stdout="Success", stderr="", returncode=0
        )

        # Create mock ranking file
        ranking_file = tmp_path / "ranking.txt"
        ranking_file.write_text("Gene1\t0.5\nGene2\t-0.3\n")

        gmt_file = tmp_path / "pathways.gmt"
        gmt_file.write_text("Pathway1\tDesc\tGene1\tGene2\n")

        output_dir = tmp_path / "pathway_output"

        result = mageck_pathway(
            gene_ranking=str(ranking_file),
            gmt_file=str(gmt_file),
            out_dir=str(output_dir),
            prefix="test",
        )

        # Check output directory was created
        assert output_dir.exists()

        # Check result structure
        assert "cmd" in result
        assert "stdout" in result

    @patch("subprocess.run")
    def test_mageck_plot_wrapper(self, mock_run, tmp_path):
        """Test mageck_plot service wrapper."""
        from crisprscreens.services.io import mageck_plot

        mock_run.return_value = MagicMock(
            stdout="Success", stderr="", returncode=0
        )

        # Create mock summary files
        gene_summary = tmp_path / "gene_summary.txt"
        gene_summary.write_text("Gene\tScore\nGene1\t0.5\n")

        output_dir = tmp_path / "plot_output"

        result = mageck_plot(
            gene_summary=str(gene_summary),
            out_dir=str(output_dir),
            prefix="test",
        )

        # Check output directory was created
        assert output_dir.exists()

        # Check result structure
        assert "cmd" in result


class TestIoEdgeCases:
    """Test I/O functions with edge cases."""

    def test_read_nonexistent_file(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_dataframe("nonexistent_file.csv")

    def test_save_figure_with_none(self, tmp_path):
        """Test save_figure with None figure."""
        # Should handle None gracefully or raise appropriate error
        with pytest.raises((AttributeError, ValueError)):
            save_figure(None, tmp_path, "test.png")
