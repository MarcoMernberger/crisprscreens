"""
Tests for core/mageck.py module.

This module tests MAGeCK utility functions for combining and filtering comparisons,
as well as wrapper functions for MAGeCK CLI tools.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from crisprscreens.core.mageck import (
    combine_comparisons,
    filter_multiple_mageck_comparisons,
    split_frame_to_control_and_query,
)


@pytest.fixture
def sample_mageck_results():
    """Create sample MAGeCK result DataFrames for testing."""
    df1 = pd.DataFrame({
        "id": ["Gene1", "Gene2", "Gene3"],
        "pos|score": [0.01, 0.5, 0.8],
        "neg|score": [0.9, 0.3, 0.1],
        "pos|fdr": [0.001, 0.6, 0.9],
        "neg|fdr": [0.9, 0.4, 0.01],
    })
    df2 = pd.DataFrame({
        "id": ["Gene1", "Gene2", "Gene4"],
        "pos|score": [0.02, 0.4, 0.7],
        "neg|score": [0.8, 0.2, 0.2],
        "pos|fdr": [0.002, 0.5, 0.8],
        "neg|fdr": [0.8, 0.3, 0.02],
    })
    return {"Comparison1": df1, "Comparison2": df2}


class TestCombineComparisons:
    """Test the combine_comparisons function."""

    def test_combine_two_comparisons_inner(self, sample_mageck_results):
        """Test combining two comparisons with inner join."""
        result = combine_comparisons(sample_mageck_results, combine_on="id", how="inner")

        # Should only have genes present in both comparisons
        assert len(result) == 2
        assert set(result["Comparison1|id"].values) == {"Gene1", "Gene2"}

        # Check that columns are renamed correctly
        assert "Comparison1|pos|score" in result.columns
        assert "Comparison2|pos|score" in result.columns
        assert "Comparison1|neg|fdr" in result.columns
        assert "Comparison2|neg|fdr" in result.columns

    def test_combine_two_comparisons_outer(self, sample_mageck_results):
        """Test combining two comparisons with outer join."""
        result = combine_comparisons(sample_mageck_results, combine_on="id", how="outer")

        # Should have all genes from both comparisons
        assert len(result) == 4  # Gene1, Gene2, Gene3, Gene4

    def test_combine_three_comparisons(self):
        """Test combining three comparisons."""
        df1 = pd.DataFrame({"id": ["Gene1", "Gene2"], "value": [1, 2]})
        df2 = pd.DataFrame({"id": ["Gene1", "Gene2"], "value": [3, 4]})
        df3 = pd.DataFrame({"id": ["Gene1", "Gene2"], "value": [5, 6]})

        mageck_results = {"Comp1": df1, "Comp2": df2, "Comp3": df3}
        result = combine_comparisons(mageck_results, combine_on="id", how="inner")

        assert len(result) == 2
        assert "Comp1|value" in result.columns
        assert "Comp2|value" in result.columns
        assert "Comp3|value" in result.columns

    def test_combine_with_dict_columns(self):
        """Test combining with different column names per comparison."""
        df1 = pd.DataFrame({"gene_id": ["Gene1", "Gene2"], "value": [1, 2]})
        df2 = pd.DataFrame({"gene_name": ["Gene1", "Gene2"], "value": [3, 4]})

        mageck_results = {"Comp1": df1, "Comp2": df2}
        combine_on = {"Comp1": "gene_id", "Comp2": "gene_name"}

        result = combine_comparisons(mageck_results, combine_on=combine_on, how="inner")
        assert len(result) == 2


class TestFilterMultipleMageckComparisons:
    """Test the filter_multiple_mageck_comparisons function."""

    def test_filter_by_single_fdr_threshold(self, sample_mageck_results):
        """Test filtering with a single FDR threshold."""
        combined = combine_comparisons(sample_mageck_results, combine_on="id", how="outer")

        filtered = filter_multiple_mageck_comparisons(
            combined,
            comparisons_to_filter=["Comparison1"],
            fdr_threshold=0.05,
            fdr_columns={"Comparison1": ("Comparison1|pos|fdr", "Comparison1|neg|fdr")},
            filter_direction="both"
        )

        # Should filter genes with FDR > 0.05 in either direction
        assert len(filtered) <= len(combined)

    def test_filter_by_dict_fdr_thresholds(self, sample_mageck_results):
        """Test filtering with different FDR thresholds per comparison."""
        combined = combine_comparisons(sample_mageck_results, combine_on="id", how="outer")

        filtered = filter_multiple_mageck_comparisons(
            combined,
            comparisons_to_filter=["Comparison1", "Comparison2"],
            fdr_threshold={"Comparison1": 0.05, "Comparison2": 0.1},
            fdr_columns={
                "Comparison1": ("Comparison1|pos|fdr", "Comparison1|neg|fdr"),
                "Comparison2": ("Comparison2|pos|fdr", "Comparison2|neg|fdr")
            },
            filter_direction="both"
        )

        assert len(filtered) <= len(combined)

    def test_filter_positive_direction_only(self, sample_mageck_results):
        """Test filtering only in positive direction."""
        combined = combine_comparisons(sample_mageck_results, combine_on="id", how="outer")

        filtered = filter_multiple_mageck_comparisons(
            combined,
            comparisons_to_filter=["Comparison1"],
            fdr_threshold=0.05,
            fdr_columns={"Comparison1": ("Comparison1|pos|fdr", "Comparison1|neg|fdr")},
            filter_direction="positive"
        )

        # Should only filter based on positive FDR
        assert all(filtered["Comparison1|pos|fdr"] <= 0.05)


class TestSplitFrameToControlAndQuery:
    """Test the split_frame_to_control_and_query function."""

    def test_split_by_control_list(self):
        """Test splitting DataFrame by control sgRNA list."""
        df = pd.DataFrame({
            "sgRNA": ["s_76442", "s_1", "s_76443", "s_2"],
            "Gene": ["Control1", "Gene1", "Control2", "Gene2"],
            "value": [1, 2, 3, 4]
        })
        control_sgrnas = {"s_76442", "s_76443"}

        control_df, query_df = split_frame_to_control_and_query(
            df, control_sgrnas, id_column="sgRNA"
        )

        assert len(control_df) == 2
        assert len(query_df) == 2
        assert set(control_df["sgRNA"].values) == {"s_76442", "s_76443"}
        assert set(query_df["sgRNA"].values) == {"s_1", "s_2"}

    def test_split_with_empty_controls(self):
        """Test splitting when no controls are present."""
        df = pd.DataFrame({
            "sgRNA": ["s_1", "s_2"],
            "Gene": ["Gene1", "Gene2"],
            "value": [1, 2]
        })
        control_sgrnas = set()

        control_df, query_df = split_frame_to_control_and_query(
            df, control_sgrnas, id_column="sgRNA"
        )

        assert len(control_df) == 0
        assert len(query_df) == 2


class TestMageckWrappers:
    """Test MAGeCK CLI wrapper functions."""

    @patch('subprocess.run')
    def test_mageck_pathway_basic_call(self, mock_run):
        """Test basic mageck_pathway wrapper call."""
        from crisprscreens.core.mageck import mageck_pathway

        # Mock subprocess result
        mock_run.return_value = MagicMock(
            stdout="Success",
            stderr="",
            returncode=0
        )

        result = mageck_pathway(
            gene_ranking="ranking.txt",
            gmt_file="pathways.gmt",
            out_dir="/tmp/output",
            prefix="test",
            method="gsea"
        )

        # Check that subprocess was called
        assert mock_run.called

        # Check result structure
        assert "cmd" in result
        assert "stdout" in result
        assert "stderr" in result
        assert "outputs" in result

    @patch('subprocess.run')
    def test_mageck_plot_basic_call(self, mock_run):
        """Test basic mageck_plot wrapper call."""
        from crisprscreens.core.mageck import mageck_plot

        mock_run.return_value = MagicMock(
            stdout="Success",
            stderr="",
            returncode=0
        )

        result = mageck_plot(
            gene_summary="gene_summary.txt",
            sgrna_summary="sgrna_summary.txt",
            out_dir="/tmp/output",
            prefix="test"
        )

        assert mock_run.called
        assert "cmd" in result
        assert "stdout" in result
