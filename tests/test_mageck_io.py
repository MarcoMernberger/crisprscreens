"""
Tests for services/mageck_io.py module.

This module tests MAGeCK-specific I/O service functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from crisprscreens.services.mageck_io import (
    write_filtered_mageck_comparison,
    combine_comparison_output,
    create_query_control_sgrna_frames,
)


# Test data paths
DATA_DIR = Path(__file__).parent / "data"
COUNT_FILE = DATA_DIR / "brunello.count.tsv"
CONTROL_FILE = DATA_DIR / "control_sgRNAs.txt"


@pytest.fixture
def sample_mageck_comparison():
    """Create sample MAGeCK comparison data."""
    return pd.DataFrame(
        {
            "Comp1|id": [f"Gene{i}" for i in range(100)],
            "Comp1|pos|fdr": np.random.uniform(0, 1, 100),
            "Comp1|neg|fdr": np.random.uniform(0, 1, 100),
            "Comp1|pos|lfc": np.random.randn(100),
            "Comp1|neg|lfc": np.random.randn(100),
            "Comp2|id": [f"Gene{i}" for i in range(100)],
            "Comp2|pos|fdr": np.random.uniform(0, 1, 100),
            "Comp2|neg|fdr": np.random.uniform(0, 1, 100),
            "Comp2|pos|lfc": np.random.randn(100),
            "Comp2|neg|lfc": np.random.randn(100),
        }
    )


@pytest.fixture
def sample_mageck_results(tmp_path):
    """Create sample MAGeCK result files."""
    results = {}

    # Create Comparison 1
    df1 = pd.DataFrame(
        {
            "id": [f"Gene{i}" for i in range(50)],
            "pos|score": np.random.uniform(0, 1, 50),
            "neg|score": np.random.uniform(0, 1, 50),
            "pos|fdr": np.random.uniform(0, 1, 50),
            "neg|fdr": np.random.uniform(0, 1, 50),
            "pos|lfc": np.random.randn(50),
            "neg|lfc": np.random.randn(50),
        }
    )
    comp1_file = tmp_path / "comparison1.txt"
    df1.to_csv(comp1_file, sep="\t", index=False)
    results["Comparison1"] = comp1_file

    # Create Comparison 2
    df2 = pd.DataFrame(
        {
            "id": [f"Gene{i}" for i in range(50)],
            "pos|score": np.random.uniform(0, 1, 50),
            "neg|score": np.random.uniform(0, 1, 50),
            "pos|fdr": np.random.uniform(0, 1, 50),
            "neg|fdr": np.random.uniform(0, 1, 50),
            "pos|lfc": np.random.randn(50),
            "neg|lfc": np.random.randn(50),
        }
    )
    comp2_file = tmp_path / "comparison2.txt"
    df2.to_csv(comp2_file, sep="\t", index=False)
    results["Comparison2"] = comp2_file

    return results


class TestWriteFilteredMageckComparison:
    """Test write_filtered_mageck_comparison function."""

    def test_write_filtered_basic(self, tmp_path, sample_mageck_comparison):
        """Test basic filtered comparison writing."""
        output_file = tmp_path / "filtered_comparison.tsv"

        result_path = write_filtered_mageck_comparison(
            combined_frame=sample_mageck_comparison,
            output_file=output_file,
            comparisons_to_filter=["Comp1", "Comp2"],
            fdr_threshold=0.05,
            fdr_columns={
                "Comp1": ("Comp1|pos|fdr", "Comp1|neg|fdr"),
                "Comp2": ("Comp2|pos|fdr", "Comp2|neg|fdr"),
            },
            filter_direction="both",
        )

        assert result_path.exists()

        # Read back and check filtering worked
        filtered_df = pd.read_csv(result_path, sep="\t")
        assert len(filtered_df) <= len(sample_mageck_comparison)

    def test_write_filtered_positive_only(
        self, tmp_path, sample_mageck_comparison
    ):
        """Test filtering only positive direction."""
        output_file = tmp_path / "filtered_positive.tsv"

        result_path = write_filtered_mageck_comparison(
            combined_frame=sample_mageck_comparison,
            output_file=output_file,
            comparisons_to_filter=["Comp1"],
            fdr_threshold=0.05,
            fdr_columns={"Comp1": ("Comp1|pos|fdr", "Comp1|neg|fdr")},
            filter_direction="positive",
        )

        assert result_path.exists()

        # Check that filtering was applied
        filtered_df = pd.read_csv(result_path, sep="\t")
        assert all(filtered_df["Comp1|pos|fdr"] <= 0.05)

    def test_write_filtered_creates_directory(
        self, tmp_path, sample_mageck_comparison
    ):
        """Test that directory is created if needed."""
        nested_dir = tmp_path / "nested" / "output"
        output_file = nested_dir / "filtered.tsv"

        result_path = write_filtered_mageck_comparison(
            combined_frame=sample_mageck_comparison,
            output_file=output_file,
            comparisons_to_filter=["Comp1"],
            fdr_threshold=0.1,
            fdr_columns={"Comp1": ("Comp1|pos|fdr", "Comp1|neg|fdr")},
            filter_direction="both",
        )

        assert nested_dir.exists()
        assert result_path.exists()

    def test_write_filtered_different_thresholds(
        self, tmp_path, sample_mageck_comparison
    ):
        """Test filtering with different thresholds per comparison."""
        output_file = tmp_path / "filtered_diff_thresh.tsv"

        result_path = write_filtered_mageck_comparison(
            combined_frame=sample_mageck_comparison,
            output_file=output_file,
            comparisons_to_filter=["Comp1", "Comp2"],
            fdr_threshold={"Comp1": 0.05, "Comp2": 0.1},
            fdr_columns={
                "Comp1": ("Comp1|pos|fdr", "Comp1|neg|fdr"),
                "Comp2": ("Comp2|pos|fdr", "Comp2|neg|fdr"),
            },
            filter_direction="both",
        )

        assert result_path.exists()


class TestCombineComparisonOutput:
    """Test combine_comparison_output function."""

    def test_combine_two_comparisons(self, tmp_path, sample_mageck_results):
        """Test combining two MAGeCK comparison outputs."""
        output_file = tmp_path / "combined_output.tsv"

        result_path = combine_comparison_output(
            mageck_results=sample_mageck_results,
            output_file=output_file,
            combine_on="id",
            how="inner",
        )

        assert result_path.exists()

        # Read and check combined data
        combined_df = pd.read_csv(result_path, sep="\t")
        assert "Comparison1|pos|fdr" in combined_df.columns
        assert "Comparison2|pos|fdr" in combined_df.columns

    def test_combine_outer_join(self, tmp_path, sample_mageck_results):
        """Test combining with outer join."""
        output_file = tmp_path / "combined_outer.tsv"

        result_path = combine_comparison_output(
            mageck_results=sample_mageck_results,
            output_file=output_file,
            combine_on="id",
            how="outer",
        )

        assert result_path.exists()

        # Outer join should have all genes
        combined_df = pd.read_csv(result_path, sep="\t")
        assert len(combined_df) >= 50  # At least as many as each individual

    def test_combine_creates_directory(self, tmp_path, sample_mageck_results):
        """Test that combine_comparison_output creates directory."""
        nested_dir = tmp_path / "nested" / "combined"
        output_file = nested_dir / "combined.tsv"

        result_path = combine_comparison_output(
            mageck_results=sample_mageck_results,
            output_file=output_file,
            combine_on="id",
            how="inner",
        )

        assert nested_dir.exists()
        assert result_path.exists()


class TestCreateQueryControlSgrnaFrames:
    """Test create_query_control_sgrna_frames function."""

    def test_split_count_dataframe(self, tmp_path):
        """Test splitting count DataFrame into control and query."""
        # Load real count data
        count_df = pd.read_csv(COUNT_FILE, sep="\t")

        # Load control sgRNAs
        with open(CONTROL_FILE, "r") as f:
            control_sgrnas = set(line.strip() for line in f)

        control_output = tmp_path / "control_sgrnas.tsv"
        query_output = tmp_path / "query_sgrnas.tsv"

        result = create_query_control_sgrna_frames(
            count_table=COUNT_FILE,
            control_sgrnas_file=CONTROL_FILE,
            output_control=control_output,
            output_query=query_output,
            id_column="sgRNA",
        )

        assert result["control_file"].exists()
        assert result["query_file"].exists()

        # Read back and verify split
        control_df = pd.read_csv(result["control_file"], sep="\t")
        query_df = pd.read_csv(result["query_file"], sep="\t")

        # All control sgRNAs should be in control file
        assert all(sgrna in control_sgrnas for sgrna in control_df["sgRNA"])

        # No control sgRNAs should be in query file
        assert all(sgrna not in control_sgrnas for sgrna in query_df["sgRNA"])

        # Total should equal original
        assert len(control_df) + len(query_df) == len(count_df)

    def test_split_creates_directory(self, tmp_path):
        """Test that output directories are created."""
        nested_dir = tmp_path / "nested" / "output"
        control_output = nested_dir / "control.tsv"
        query_output = nested_dir / "query.tsv"

        result = create_query_control_sgrna_frames(
            count_table=COUNT_FILE,
            control_sgrnas_file=CONTROL_FILE,
            output_control=control_output,
            output_query=query_output,
            id_column="sgRNA",
        )

        assert nested_dir.exists()
        assert result["control_file"].exists()
        assert result["query_file"].exists()


class TestMageckIoEdgeCases:
    """Test MAGeCK I/O functions with edge cases."""

    def test_combine_empty_results(self, tmp_path):
        """Test combining empty MAGeCK results."""
        # Create empty files
        empty_df = pd.DataFrame(columns=["id", "pos|fdr", "neg|fdr"])
        file1 = tmp_path / "empty1.txt"
        file2 = tmp_path / "empty2.txt"
        empty_df.to_csv(file1, sep="\t", index=False)
        empty_df.to_csv(file2, sep="\t", index=False)

        mageck_results = {"Comp1": file1, "Comp2": file2}
        output_file = tmp_path / "combined_empty.tsv"

        result_path = combine_comparison_output(
            mageck_results=mageck_results,
            output_file=output_file,
            combine_on="id",
            how="outer",
        )

        assert result_path.exists()
        combined_df = pd.read_csv(result_path, sep="\t")
        assert len(combined_df) == 0

    def test_filter_no_significant_genes(self, tmp_path):
        """Test filtering when no genes pass threshold."""
        # Create data with high FDRs
        df = pd.DataFrame(
            {
                "Comp1|id": [f"Gene{i}" for i in range(20)],
                "Comp1|pos|fdr": np.random.uniform(0.5, 1.0, 20),
                "Comp1|neg|fdr": np.random.uniform(0.5, 1.0, 20),
            }
        )

        output_file = tmp_path / "filtered_none.tsv"

        result_path = write_filtered_mageck_comparison(
            combined_frame=df,
            output_file=output_file,
            comparisons_to_filter=["Comp1"],
            fdr_threshold=0.01,  # Very strict threshold
            fdr_columns={"Comp1": ("Comp1|pos|fdr", "Comp1|neg|fdr")},
            filter_direction="both",
        )

        assert result_path.exists()

        # Should still create file, just empty or very few rows
        filtered_df = pd.read_csv(result_path, sep="\t")
        assert len(filtered_df) <= len(df)

    def test_split_with_no_controls(self, tmp_path):
        """Test splitting when no controls are in the data."""
        # Create count data with sgRNAs not in control file
        count_data = pd.DataFrame(
            {
                "sgRNA": [f"s_novel_{i}" for i in range(50)],
                "Gene": [f"Gene{i}" for i in range(50)],
                "Sample1": np.random.randint(0, 1000, 50),
            }
        )
        count_file = tmp_path / "count_no_controls.tsv"
        count_data.to_csv(count_file, sep="\t", index=False)

        control_output = tmp_path / "control_empty.tsv"
        query_output = tmp_path / "query_all.tsv"

        result = create_query_control_sgrna_frames(
            count_table=count_file,
            control_sgrnas_file=CONTROL_FILE,
            output_control=control_output,
            output_query=query_output,
            id_column="sgRNA",
        )

        # Control file should be empty
        control_df = pd.read_csv(result["control_file"], sep="\t")
        assert len(control_df) == 0

        # All sgRNAs should be in query file
        query_df = pd.read_csv(result["query_file"], sep="\t")
        assert len(query_df) == 50
