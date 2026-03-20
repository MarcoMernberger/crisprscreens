"""
Tests for core/qc.py module.

This module tests quality control functions for CRISPR screen analysis.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from crisprscreens.core.qc import (
    load_control_sgrnas,
    parse_condition_replicate,
    read_counts,
    parse_metadata_from_columns,
    calculate_cpm,
    calculate_delta_logfc,
    compute_library_stats,
    compute_size_factors_total,
    compute_size_factors_median_ratio,
    select_stable_guides,
    compute_size_factors_stable_set,
    compute_size_factors_control,
    apply_size_factors,
    qc_logfc_distribution,
    qc_replicate_consistency,
    qc_controls_neutrality,
    compare_size_factors,
    choose_best_normalization,
    recommend_analysis_method,
    control_sgrna_qc,
)


# Test data paths
DATA_DIR = Path(__file__).parent / "data"
COUNT_FILE = DATA_DIR / "brunello.count.tsv"
CONTROL_FILE = DATA_DIR / "control_sgRNAs.txt"
DESIGN_MATRIX = DATA_DIR / "design_matrix_3.tsv"


@pytest.fixture
def sample_ids():
    """Sample IDs for testing."""
    return {
        "baseline": ["Total_Rep1", "Total_Rep2", "Total_Rep3"],
        "sorted1": ["Sort1_Rep1", "Sort1_Rep2", "Sort1_Rep3"],
        "sorted2": ["Sort2_Rep1", "Sort2_Rep2", "Sort2_Rep3"],
    }


@pytest.fixture
def count_df():
    """Load real count data for testing."""
    df = pd.read_csv(COUNT_FILE, sep="\t")
    return df


@pytest.fixture
def control_sgrnas():
    """Load real control sgRNAs for testing."""
    return load_control_sgrnas(CONTROL_FILE)


class TestLoadControlSgrnas:
    """Test load_control_sgrnas function."""

    def test_load_control_sgrnas_from_file(self):
        """Test loading control sgRNAs from file."""
        controls = load_control_sgrnas(CONTROL_FILE)

        assert isinstance(controls, set)
        assert len(controls) > 0
        # Check first few entries
        assert "s_76442" in controls
        assert "s_76443" in controls

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_control_sgrnas("nonexistent.txt")


class TestParseConditionReplicate:
    """Test parse_condition_replicate function."""

    def test_parse_standard_format(self):
        """Test parsing standard condition_replicate format."""
        condition, replicate = parse_condition_replicate("Total_Rep1")
        assert condition == "Total"
        assert replicate == "Rep1"

        condition, replicate = parse_condition_replicate("Sort1_Rep2")
        assert condition == "Sort1"
        assert replicate == "Rep2"

    def test_parse_with_custom_separator(self):
        """Test parsing with custom separator."""
        condition, replicate = parse_condition_replicate("Total-Rep1", separator="-")
        assert condition == "Total"
        assert replicate == "Rep1"

    def test_parse_single_word(self):
        """Test parsing sample name without separator."""
        condition, replicate = parse_condition_replicate("Sample1")
        assert condition == "Sample1"
        assert replicate == ""


class TestReadCounts:
    """Test read_counts function."""

    def test_read_counts_basic(self, sample_ids):
        """Test reading count file."""
        all_samples = sample_ids["baseline"] + sample_ids["sorted1"] + sample_ids["sorted2"]
        result = read_counts(COUNT_FILE, sample_cols=all_samples)

        assert "count_df" in result
        assert "sample_cols" in result
        assert "conditions_dict" in result

        df = result["count_df"]
        assert "sgRNA" in df.columns
        assert "Gene" in df.columns
        assert all(col in df.columns for col in all_samples)

    def test_read_counts_with_log2(self, sample_ids):
        """Test reading counts with log2 transformation."""
        all_samples = sample_ids["baseline"] + sample_ids["sorted1"]
        result = read_counts(COUNT_FILE, sample_cols=all_samples, log2_transform=True, pseudocount=1.0)

        df = result["count_df"]
        # Check that values are log2 transformed
        assert df[all_samples].min().min() >= 0  # log2(1 + count) should be >= 0


class TestParseMetadataFromColumns:
    """Test parse_metadata_from_columns function."""

    def test_parse_metadata_standard(self, sample_ids):
        """Test parsing metadata from standard column names."""
        all_samples = sample_ids["baseline"] + sample_ids["sorted1"] + sample_ids["sorted2"]
        metadata = parse_metadata_from_columns(all_samples)

        assert len(metadata) == len(all_samples)
        assert "sample" in metadata.columns
        assert "condition" in metadata.columns
        assert "replicate" in metadata.columns

        # Check specific values
        total_rows = metadata[metadata["condition"] == "Total"]
        assert len(total_rows) == 3

        sort1_rows = metadata[metadata["condition"] == "Sort1"]
        assert len(sort1_rows) == 3


class TestCalculateCpm:
    """Test calculate_cpm function."""

    def test_calculate_cpm(self, count_df, sample_ids):
        """Test CPM calculation."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        cpm_df = calculate_cpm(count_df, sample_cols)

        # Check that CPM sums to approximately 1 million per sample
        for col in sample_cols:
            total_cpm = cpm_df[col].sum()
            assert np.isclose(total_cpm, 1e6, rtol=0.01)


class TestCalculateDeltaLogfc:
    """Test calculate_delta_logfc function."""

    def test_calculate_delta_basic(self, count_df, sample_ids):
        """Test basic delta log fold change calculation."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
        }

        delta_df = calculate_delta_logfc(
            count_df,
            sample_cols=sample_cols,
            conditions_dict=conditions_dict,
            baseline_condition="Total",
            pseudocount=1.0
        )

        # Should only have Sort1 columns (not Total)
        for col in sample_ids["sorted1"]:
            assert col in delta_df.columns

        # Should not have baseline columns
        for col in sample_ids["baseline"]:
            assert col not in delta_df.columns

        # Check that values are reasonable (log2 fold changes)
        assert delta_df[sample_ids["sorted1"]].abs().max().max() < 20


class TestComputeLibraryStats:
    """Test compute_library_stats function."""

    def test_compute_library_stats(self, count_df, sample_ids):
        """Test library statistics computation."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        stats = compute_library_stats(count_df, sample_cols)

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == len(sample_cols)
        assert "total_reads" in stats.columns
        assert "zero_count_guides" in stats.columns
        assert "median_count" in stats.columns
        assert "gini_index" in stats.columns

        # Check that all samples have positive total reads
        assert (stats["total_reads"] > 0).all()


class TestComputeSizeFactors:
    """Test size factor computation functions."""

    def test_compute_size_factors_total(self, count_df, sample_ids):
        """Test total count size factor computation."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        size_factors = compute_size_factors_total(count_df, sample_cols)

        assert len(size_factors) == len(sample_cols)
        assert all(size_factors > 0)

        # Geometric mean should be close to 1
        geom_mean = np.exp(np.log(size_factors).mean())
        assert np.isclose(geom_mean, 1.0, rtol=0.01)

    def test_compute_size_factors_median_ratio(self, count_df, sample_ids):
        """Test median ratio size factor computation."""
        sample_cols = sample_ids["baseline"]
        size_factors = compute_size_factors_median_ratio(count_df, sample_cols)

        assert len(size_factors) == len(sample_cols)
        assert all(size_factors > 0)

    def test_compute_size_factors_control(self, count_df, control_sgrnas, sample_ids):
        """Test control-based size factor computation."""
        sample_cols = sample_ids["baseline"]
        size_factors = compute_size_factors_control(
            count_df, control_sgrnas, sample_cols, id_column="sgRNA"
        )

        assert len(size_factors) == len(sample_cols)
        assert all(size_factors > 0)


class TestSelectStableGuides:
    """Test select_stable_guides function."""

    def test_select_stable_guides(self, count_df, sample_ids):
        """Test selecting stable guides."""
        sample_cols = sample_ids["baseline"]
        stable = select_stable_guides(
            count_df, sample_cols, cv_quantile=0.25, min_count=10
        )

        assert isinstance(stable, set)
        assert len(stable) > 0
        assert len(stable) < len(count_df)  # Should be subset


class TestApplySizeFactors:
    """Test apply_size_factors function."""

    def test_apply_size_factors(self, count_df, sample_ids):
        """Test applying size factors to counts."""
        sample_cols = sample_ids["baseline"]
        size_factors = compute_size_factors_total(count_df, sample_cols)

        normalized_df = apply_size_factors(count_df, size_factors, sample_cols)

        # Check that sgRNA and Gene columns are preserved
        assert "sgRNA" in normalized_df.columns
        assert "Gene" in normalized_df.columns

        # Check that normalized values are different from original
        assert not normalized_df[sample_cols[0]].equals(count_df[sample_cols[0]])


class TestQcFunctions:
    """Test QC analysis functions."""

    def test_qc_logfc_distribution(self, count_df, sample_ids):
        """Test log fold change distribution QC."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
        }

        result = qc_logfc_distribution(
            count_df,
            sample_cols=sample_cols,
            conditions_dict=conditions_dict,
            baseline_condition="Total",
            pseudocount=1.0,
            make_plots=False  # Don't create plots in tests
        )

        assert "delta_df" in result
        assert "figures" in result
        assert "metrics" in result

    def test_qc_replicate_consistency(self, count_df, sample_ids):
        """Test replicate consistency QC."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
        }

        result = qc_replicate_consistency(
            count_df,
            sample_cols=sample_cols,
            conditions_dict=conditions_dict,
            baseline_condition="Total",
            pseudocount=1.0,
            make_plots=False
        )

        assert "delta_df" in result
        assert "figures" in result
        assert "metrics" in result

        # Check that baseline is skipped
        metrics = result["metrics"]
        assert "Total" not in metrics.index

    def test_qc_controls_neutrality(self, count_df, control_sgrnas, sample_ids):
        """Test control sgRNA neutrality QC."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
        }

        result = qc_controls_neutrality(
            count_df,
            control_sgrnas,
            sample_cols=sample_ids,
            conditions_dict=conditions_dict,
            baseline_condition="Total",
            pseudocount=1.0,
            make_plots=False
        )

        assert "delta_df" in result
        assert "figures" in result
        assert "metrics" in result


class TestCompareSizeFactors:
    """Test compare_size_factors function."""

    def test_compare_size_factors(self, count_df, control_sgrnas, sample_ids):
        """Test comparing different normalization methods."""
        sample_cols = sample_ids["baseline"]

        sf_total = compute_size_factors_total(count_df, sample_cols)
        sf_median = compute_size_factors_median_ratio(count_df, sample_cols)
        sf_control = compute_size_factors_control(count_df, control_sgrnas, sample_cols)

        size_factors_dict = {
            "total": sf_total,
            "median_ratio": sf_median,
            "control": sf_control,
        }

        comparison = compare_size_factors(size_factors_dict)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == len(sample_cols)
        assert "total" in comparison.columns
        assert "median_ratio" in comparison.columns
        assert "control" in comparison.columns


class TestChooseBestNormalization:
    """Test choose_best_normalization function."""

    def test_choose_best_normalization(self, count_df, control_sgrnas, sample_ids):
        """Test choosing best normalization method."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
        }

        result = choose_best_normalization(
            count_df,
            sample_cols=sample_cols,
            conditions_dict=conditions_dict,
            baseline_condition="Total",
            control_sgrnas=control_sgrnas,
            pseudocount=1.0,
            make_plots=False
        )

        assert "recommendation" in result
        assert "size_factors" in result
        assert "comparison_df" in result
        assert "qc_results" in result

        # Check recommendation is one of the expected methods
        assert result["recommendation"] in ["total", "median_ratio", "control", "stable_set"]


class TestRecommendAnalysisMethod:
    """Test recommend_analysis_method function."""

    def test_recommend_analysis_method(self, count_df, sample_ids):
        """Test recommending RRA vs MLE."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"] + sample_ids["sorted2"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
            "Sort2": sample_ids["sorted2"],
        }

        result = recommend_analysis_method(
            count_df,
            sample_cols=sample_cols,
            conditions_dict=conditions_dict,
            design_matrix_file=DESIGN_MATRIX
        )

        assert "recommendation" in result
        assert "reason" in result
        assert result["recommendation"] in ["RRA", "MLE", "Both"]


class TestControlSgrnaQc:
    """Test control_sgrna_qc function."""

    def test_control_sgrna_qc(self, count_df, control_sgrnas, sample_ids):
        """Test comprehensive control sgRNA QC."""
        sample_cols = sample_ids["baseline"] + sample_ids["sorted1"]
        conditions_dict = {
            "Total": sample_ids["baseline"],
            "Sort1": sample_ids["sorted1"],
        }

        result = control_sgrna_qc(
            count_df,
            control_sgrnas,
            sample_cols=sample_cols,
            conditions_dict=conditions_dict,
            baseline_condition="Total",
            pseudocount=1.0,
            make_plots=False
        )

        assert "figures" in result
        assert "metrics" in result
        assert "control_df" in result
        assert "query_df" in result
