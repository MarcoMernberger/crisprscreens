import pandas as pd  # type: ignore
from typing import Dict, Optional, Union, List, Tuple, Callable, Any, Iterable
from pathlib import Path
from crisprscreens.core.mageck import (
    filter_multiple_mageck_comparisons,
    combine_comparisons,
    split_frame_to_control_and_query,
    combine_gene_info_with_mageck_output,
    get_significant_genes,
    filter_mageck_counts,
)
from crisprscreens.core.mageck_spikein import create_spiked_count_table
from crisprscreens.core.qc import (
    read_counts,
    calculate_norm_cpms_and_ma,
    calculate_size_factors_for_method,
)


def write_filtered_mageck_comparison(
    output_file: Path,
    combined_frame_input_file: Path,
    comparisons_to_filter: List[str],
    fdr_threshold: Union[float, Dict[str, float]] = 0.05,
    change_threshold: Union[float, Dict[str, float]] = 1.0,
    z_thresholds: Optional[Union[float, Dict[str, float]]] = None,
    direction: str = "both",  # "both", "pos", "neg"
    require_all: bool = True,  # AND (True) vs OR (False)
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_frame = pd.read_csv(combined_frame_input_file, sep="\t")
    filtered_frame = filter_multiple_mageck_comparisons(
        combined_frame=combined_frame,
        comparisons_to_filter=comparisons_to_filter,
        fdr_threshold=fdr_threshold,
        change_threshold=change_threshold,
        z_thresholds=z_thresholds,
        direction=direction,
        require_all=require_all,
    )
    filtered_frame.to_csv(output_file, sep="\t", index=False)


def combine_comparison_output(
    output_file: Path,
    mageck_results: Dict[str, Path],
    combine_on: Union[str, Dict[str, str]] = "id",
    how: str = "inner",
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mageck_frames = {}
    for name, mageck_file in mageck_results.items():
        mageck_frames[name] = pd.read_csv(mageck_file, sep="\t")
    merged_frame = combine_comparisons(
        mageck_frames, combine_on=combine_on, how=how
    )
    merged_frame.to_csv(output_file, sep="\t", index=False)


def create_query_control_sgrna_frames(
    infile: Path,
    outfiles: Tuple[Path],
    control_prefix: str,
    id_col: Optional[str] = None,
    name_column: str = "name",
    sgRNA_column: str = "sgRNA",
    infer_genes: Optional[Callable] = None,
    read_csv_kwargs: Optional[Dict] = None,
):
    outfiles[0].parent.mkdir(parents=True, exist_ok=True)
    read_csv_kwargs = read_csv_kwargs or {"sep": "\t"}
    df = pd.read_csv(infile, **read_csv_kwargs)
    split_dfs = split_frame_to_control_and_query(
        df,
        control_prefix=control_prefix,
        id_col=id_col,
        name_column=name_column,
        sgRNA_column=sgRNA_column,
        infer_genes=infer_genes,
    )
    split_dfs["control"].to_csv(
        outfiles[1], sep="\t", index=False, header=False
    )
    split_dfs["query"].to_csv(outfiles[0], sep="\t", index=False, header=False)


def create_combine_gene_info_with_mageck_output(
    mageck_file: Path,
    gene_info_file: Path,
    output_file: Path,
    name_column_mageck: str = "id",
    name_column_genes: str = "name_given",
    how: str = "inner",
    columns_to_add: List[str] = [
        "gene_stable_id",
        "name",
        "chr",
        "start",
        "stop",
        "strand",
        "tss",
        "tes",
        "biotype",
    ],
    read_csv_kwargs: Optional[Dict] = None,
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    read_csv_kwargs = read_csv_kwargs or {"sep": "\t"}
    mageck_df = pd.read_csv(mageck_file, **read_csv_kwargs)
    gene_info_df = pd.read_csv(gene_info_file, **read_csv_kwargs)
    combined_df = combine_gene_info_with_mageck_output(
        mageck_df,
        gene_info_df,
        name_column_mageck=name_column_mageck,
        name_column_genes=name_column_genes,
        how=how,
        columns_to_add=columns_to_add,
    )
    combined_df.to_csv(output_file, sep="\t", index=False)


def write_spiked_count_table(
    outfile: Union[Path, str],
    count_file: Union[str, Path],
    replicate_of: Dict[str, str],
    sample_to_group: Dict[str, str],
    group_contrast: Tuple[str] = ("sorted", "total"),
    n_genes: int = 20,
    log_effect: float = 2.0,
    baseline_mean: float = 300.0,
    dispersion: float = 0.08,
):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    count_df = pd.read_csv(count_file, sep="\t")
    sample_cols = [c for c in count_df.columns if (c not in ("sgRNA", "Gene"))]
    count_spike = create_spiked_count_table(
        count_df,
        replicate_of=replicate_of,
        sample_to_group=sample_to_group,
        sample_cols=sample_cols,
        group_contrast=group_contrast,
        n_genes=n_genes,
        log_effect=log_effect,
        baseline_mean=baseline_mean,
        dispersion=dispersion,
    )
    count_spike.to_csv(outfile, sep="\t", index=False)


def write_significant_genes(
    outfile: Union[Path, str],
    mageck_file: Union[str, Path],
    fdr_column: str = "pos|fdr",
    fdr_threshold: float = 0.05,
    logfc_column: str = "pos|lfc",
    logfc_or_beta_threshold: float = 1.0,
    direction: str = "both",  # "both", "pos", "neg"
):
    """
    write_significant_genes from a mageck gene summary table.

    Parameters
    ----------
    outfile : Union[Path, str]
        The output file to write significant genes to.
    mageck_file : Union[str, Path]
        The input mageck gene summary file.
    fdr_column : str, optional
        FDR column name, by default "pos|fdr"
    fdr_threshold : float, optional
        FDR threshold, by default 0.05
    logfc_column : str, optional
        LogFC column name, by default "pos|lfc"
    logfc_or_beta_threshold : float, optional
        LogFC or beta threshold, by default 1.0
    direction : str, optional
        Direction of change: "both", "pos", "neg", by default "both"
    method : str, optional
        Method used: "rra" or "mle", by default "rra"

    """
    outfile.parent.mkdir(parents=True, exist_ok=True)
    mageck_df = pd.read_csv(mageck_file, sep="\t")
    sig_genes_df = get_significant_genes(
        mageck_df,
        fdr_column=fdr_column,
        fdr_threshold=fdr_threshold,
        logfc_column=logfc_column,
        logfc_or_beta_threshold=logfc_or_beta_threshold,
        direction=direction,
    )
    sig_genes_df.to_csv(outfile, sep="\t", index=False)
    return outfile


def write_significant_genes_mageck(
    mageck_file: Union[str, Path],
    fdr_threshold: float = 0.05,
    logfc_or_beta_threshold: float = 1.0,
    direction: str = "both",  # "both", "pos", "neg"
    fdr_column_pos: str = "pos|fdr",
    fdr_column_neg: Optional[str] = "neg|fdr",
    lfc_column_pos: str = "pos|lfc",
    lfc_column_neg: Optional[str] = "neg|lfc",
    method: str = "rra",
) -> List[Path]:
    """
    write_significant_genes_rra for RRA results from mageck.

    Parameters
    ----------
    mageck_file : Union[str, Path]
        The input mageck gene summary file.
    fdr_threshold : float, optional
        FDR threshold, by default 0.05
    logfc_or_beta_threshold : float, optional
        LogFC threshold or beta score threshold, by default 1.0
    direction : str, optional
        Direction of change: "both", "pos", "neg", by default "both"

    Returns
    -------
    List[Path]
        List of output files for enriched and depleted genes.
    """
    if method == "mle":
        lfc_column_neg = lfc_column_pos
        fdr_column_neg = fdr_column_pos
    elif method == "rra":
        pass
    else:
        raise ValueError(f"Invalid method: {method}")
    if direction == "both":
        outfiles = [
            mageck_file.with_suffix(".enriched.genes.tsv"),
            mageck_file.with_suffix(".depleted.genes.tsv"),
        ]
    elif direction == "pos":
        outfiles = [mageck_file.with_suffix(".enriched.genes.tsv")]
    elif direction == "neg":
        outfiles = [mageck_file.with_suffix(".depleted.genes.tsv")]
    else:
        raise ValueError(f"Invalid direction: {direction}")

    if direction == "both" or direction == "pos":
        write_significant_genes(
            outfiles[0],
            mageck_file,
            fdr_column=fdr_column_pos,
            fdr_threshold=fdr_threshold,
            logfc_column=lfc_column_pos,
            logfc_or_beta_threshold=logfc_or_beta_threshold,
            direction="pos",
        )
    if direction == "both" or direction == "neg":
        write_significant_genes(
            outfiles[1],
            mageck_file,
            fdr_column=fdr_column_neg,
            fdr_threshold=fdr_threshold,
            logfc_column=lfc_column_neg,
            logfc_or_beta_threshold=logfc_or_beta_threshold,
            direction="neg",
        )
    if direction == "pos":
        outfiles = [outfiles[0]]
    elif direction == "neg":
        outfiles = [outfiles[1]]
    return outfiles


def write_filter_mageck_counts(
    count_table_file: Union[str, Path],
    samples: Iterable[str],
    outfile: Union[str, Path],
    conditions: Dict[str, Any],
    baseline_samples: Iterable[str],
    aggregations: Optional[Dict[str, Tuple[List[str], Callable]]] = None,
    exclude_samples: Optional[Iterable[str]] = None,
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
) -> Path:
    """
    write_filter_mageck_counts to filter low count sgRNAs.

    Parameters
    ----------
    count_table_file : Union[str, Path]
        The input mageck count table file.
    samples : Iterable[str]
        Names of all sample columns (raw counts) like
        ["Total1","Total2","Total3","Sorted1","Sorted2","Sorted3"].
    outfile : Union[str, Path]
        The output filtered count table file.
    filter_threshold : int, optional
        The count threshold below which sgRNAs are filtered, by default 5
    paired_samples_before_after : Dict[str, str]
        A dictionary mapping 'before' sample names to 'after' sample names.
    min_baseline_above_threshold : int, optional
        Minimum number of 'before' samples that must be above the threshold,
        by default 1

    Returns
    -------
    Path
        The output filtered count table file path.
    """

    outfile = Path(outfile)
    outfile2 = Path(outfile.with_suffix(".full.tsv"))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    count_df = pd.read_csv(count_table_file, sep="\t")
    filtered_count_df = filter_mageck_counts(
        count_df,
        conditions=conditions,
        baseline_samples=baseline_samples,
        aggregations=aggregations,
        exclude_samples=exclude_samples,
    )
    colums_to_keep = [sgrna_col, gene_col] + samples
    if exclude_samples is not None:
        colums_to_keep = [c for c in colums_to_keep if c not in exclude_samples]
    filtered_count_df[colums_to_keep].to_csv(
        str(outfile), sep="\t", index=False
    )
    filtered_count_df.to_csv(outfile2, sep="\t", index=False)
    return outfile


def write_count_table_with_MA_CPM(
    count_tsv: Union[str, Path],
    method: str = "median",
    pseudocount: float = 1.0,
    paired_replicates: bool = False,
    conditions_dict: Dict[str, List[str]] = {},
    baseline_condition: str = "total",
    control_sgrna_txt: Optional[Union[Path, str]] = None,
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    delimiter: str = "_",
    suffix: str = "CPM_MA",
) -> Path:
    """
    write_count_table_with_MA_CPM to add MA and CPM columns to count table.

    Parameters
    """
    outfile = Path(count_tsv).with_suffix(f".{suffix}.tsv")
    count_df, sample_cols = read_counts(count_tsv, sgrna_col, gene_col)
    size_factors = calculate_size_factors_for_method(
        method=method,
        count_df=count_df,
        sample_cols=sample_cols,
        control_sgrna_txt=control_sgrna_txt,
        sgrna_col=sgrna_col,
    )
    df_out = calculate_norm_cpms_and_ma(
        count_df=count_df,
        sample_cols=sample_cols,
        method=method,
        size_factors=size_factors,
        pseudocount=pseudocount,
        paired_replicates=paired_replicates,
        conditions_dict=conditions_dict,
        baseline_condition=baseline_condition,
        delimiter=delimiter,
    )
    df_out.to_csv(outfile, sep="\t", index=False)
    return outfile
