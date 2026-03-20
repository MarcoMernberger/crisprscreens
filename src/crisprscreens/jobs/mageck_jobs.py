from pypipegraph2 import (
    Job,
    FileGeneratingJob,
    MultiFileGeneratingJob,
    FunctionInvariant,
    ParameterInvariant,
)
from pathlib import Path
from typing import (
    Dict,
    Optional,
    Union,
    List,
    Tuple,
    Literal,
    Callable,
    Any,
    Iterable,
)
from crisprscreens.services.mageck_io import (
    write_filtered_mageck_comparison,
    combine_comparison_output,
    create_query_control_sgrna_frames,
    create_combine_gene_info_with_mageck_output,
    write_spiked_count_table,
    write_significant_genes_mageck,
    write_filter_mageck_counts,
    write_count_table_with_MA_CPM,
)
from crisprscreens.services.spike_evaluation import (
    evaluate_multiple_mageck_results,
    rank_mageck_methods,
)
from crisprscreens.r_integration.mageck_wrapper import run_mageck_scatterview
from crisprscreens.core.mageck import mageck_count, mageck_test, mageck_mle, mageck_count2
from pandas import DataFrame
from crisprscreens.services.io import generate_spike_evaluation_report
import numpy as np
import shutil


def combine_comparison_output_job(
    mageck_results: Dict[str, Path],
    output_file: Path,
    combine_on: Union[str, Dict[str, str]] = "id",
    how: str = "inner",
    dependencies: List[Job] = [],
):
    def __dump(
        output_file,
        mageck_results=mageck_results,
        combine_on=combine_on,
        how=how,
    ):
        combine_comparison_output(
            output_file=output_file,
            mageck_results=mageck_results,
            combine_on=combine_on,
            how=how,
        )

    return FileGeneratingJob(output_file, __dump).depends_on(dependencies)


def write_filtered_mageck_comparison_job(
    output_file: Path,
    combined_frame_input_file: Path,
    comparisons_to_filter: List[str],
    fdr_threshold: Union[float, Dict[str, float]] = 0.05,
    change_threshold: Union[float, Dict[str, float]] = 1.0,
    z_thresholds: Optional[Union[float, Dict[str, float]]] = None,
    direction: str = "both",  # "both", "pos", "neg"
    require_all: bool = True,  # AND (True) vs OR (False)
    dependencies: List[Job] = [],
):

    def __dump(
        output_file,
        combined_frame_input_file=combined_frame_input_file,
        comparisons_to_filter=comparisons_to_filter,
        fdr_threshold=fdr_threshold,
        change_threshold=change_threshold,
        z_thresholds=z_thresholds,
        direction=direction,
        require_all=require_all,
    ):
        write_filtered_mageck_comparison(
            output_file=output_file,
            combined_frame_input_file=combined_frame_input_file,
            comparisons_to_filter=comparisons_to_filter,
            fdr_threshold=fdr_threshold,
            change_threshold=change_threshold,
            z_thresholds=z_thresholds,
            direction=direction,
            require_all=require_all,
        )

    return FileGeneratingJob(output_file, __dump).depends_on(dependencies)


def run_mageck_scatterview_job(
    input_file: Union[str, Path],
    x_col: str,
    y_col: str,
    output_dir: Union[str, Path] = ".",
    filebase_name: str = "scatterview",
    datatype: Literal["rra", "mle"] = "rra",
    gene_col: str = "id",
    sep: str = "\t",
    normalize: bool = True,
    top: int = 10,
    groups: Tuple[str] = (
        "bottomleft",
        "topcenter",
        "midright",
    ),  # selection only those groups
    select: Optional[
        Literal["positive", "negative", "both", "none"]
    ] = "negative",  # selection by diagonal None, "positive", "negative", "both", "none"
    neg_effect_cutoff: float = -0.4,
    pos_effect_cutoff: float = 0.4,
    delta_cutoff_k: float = 2,
    filter_fdr_x: bool = False,  # selection filter by FDR column
    filter_fdr_y: bool = False,  # selection filter by FDR column
    filter_groups: bool = True,  # if false groups will be displayed but not filtered
    fdr_cutoff: float = 0.05,  # FDR cutoff
    # plot parameters
    toplabels: bool = True,
    label_selected_only: bool = False,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    jpeg_width: int = 20,
    jpeg_height: int = 15,
    # additional plot parameter for MLE
    auto_cut_diag: float = 2,
    auto_cut_x: float = 2,
    auto_cut_y: float = 2,
    # additional plot parameter for RRA
    x_cut: Optional[Tuple[float, float]] = None,  # e.g. (-0.5, 0.5),
    y_cut: Optional[Tuple[float, float]] = None,  # e.g. (-0.5, 0.5),
    dependencies: List[Job] = [],
):
    outfiles = [
        Path(output_dir) / f"{filebase_name}_data.tsv",
        Path(output_dir) / f"{filebase_name}_hits_selected.tsv",
        Path(output_dir) / f"{filebase_name}.jpeg",
        Path(output_dir) / f"{filebase_name}_lmfit.jpeg",
    ]

    def __dump(
        outfiles,
        input_file=input_file,
        x_col=x_col,
        y_col=y_col,
        output_dir=output_dir,
        filebase_name=filebase_name,
        datatype=datatype,
        gene_col=gene_col,
        sep=sep,
        normalize=normalize,
        top=top,
        groups=groups,
        select=select,
        neg_effect_cutoff=neg_effect_cutoff,
        pos_effect_cutoff=pos_effect_cutoff,
        delta_cutoff_k=delta_cutoff_k,
        filter_fdr_x=filter_fdr_x,
        filter_fdr_y=filter_fdr_y,
        filter_groups=filter_groups,
        fdr_cutoff=fdr_cutoff,
        toplabels=toplabels,
        label_selected_only=label_selected_only,
        xlab=xlab,
        ylab=ylab,
        jpeg_width=jpeg_width,
        jpeg_height=jpeg_height,
        auto_cut_diag=auto_cut_diag,
        auto_cut_x=auto_cut_x,
        auto_cut_y=auto_cut_y,
        x_cut=x_cut,
        y_cut=y_cut,
    ):
        run_mageck_scatterview(
            input_file=input_file,
            x_col=x_col,
            y_col=y_col,
            output_dir=output_dir,
            filebase_name=filebase_name,
            datatype=datatype,
            gene_col=gene_col,
            sep=sep,
            normalize=normalize,
            top=top,
            groups=groups,
            select=select,
            neg_effect_cutoff=neg_effect_cutoff,
            pos_effect_cutoff=pos_effect_cutoff,
            delta_cutoff_k=delta_cutoff_k,
            filter_fdr_x=filter_fdr_x,
            filter_fdr_y=filter_fdr_y,
            filter_groups=filter_groups,
            fdr_cutoff=fdr_cutoff,
            toplabels=toplabels,
            label_selected_only=label_selected_only,
            xlab=xlab,
            ylab=ylab,
            jpeg_width=jpeg_width,
            jpeg_height=jpeg_height,
            auto_cut_diag=auto_cut_diag,
            auto_cut_x=auto_cut_x,
            auto_cut_y=auto_cut_y,
            x_cut=x_cut,
            y_cut=y_cut,
        )

    return MultiFileGeneratingJob(outfiles, __dump).depends_on(dependencies)


def mageck_count_job(
    sgrna_list: Union[Path, str],
    samples: dict,
    out_dir: Union[Path, str],
    prefix: str,
    control_sgrnas=Optional[: Union[Path, str]],
    norm_method: str = None,
    pdf_report: bool = False,
    other_parameter: Dict[str, str] = [],
    dependencies: List[Job] = [],
):
    outfile = Path(out_dir) / f"{prefix}.count.txt"

    def __dump(
        outfile,
        sgrna_list=sgrna_list,
        samples=samples,
        out_dir=out_dir,
        prefix=prefix,
        control_sgrnas=control_sgrnas,
        norm_method=norm_method,
        pdf_report=pdf_report,
        other_parameter=other_parameter,
    ):
        mageck_count(
            sgrna_list=sgrna_list,
            samples=samples,
            out_dir=out_dir,
            prefix=prefix,
            control_sgrnas=control_sgrnas,
            norm_method=norm_method,
            pdf_report=pdf_report,
            other_parameter=other_parameter,
        )

    return FileGeneratingJob(outfile, __dump).depends_on(dependencies)


def mageck_count_job2(
    sgrna_list: Union[Path, str],
    samples: dict,
    out_dir: Union[Path, str],
    prefix: str,
    count_table: Optional[Union[Path, str]] = None,
    control_sgrnas=Optional[: Union[Path, str]],
    norm_method: str = None,
    pdf_report: bool = False,
    other_parameter: Dict[str, str] = [],
    dependencies: List[Job] = [],
):
    outfile = Path(out_dir) / f"{prefix}.count.txt"

    def __dump(
        outfile,
        sgrna_list=sgrna_list,
        samples=samples,
        out_dir=out_dir,
        prefix=prefix,
        count_table=count_table,
        control_sgrnas=control_sgrnas,
        norm_method=norm_method,
        pdf_report=pdf_report,
        other_parameter=other_parameter,
    ):
        mageck_count2(
            sgrna_list=sgrna_list,
            samples=samples,
            out_dir=out_dir,
            prefix=prefix,
            count_table=count_table,
            control_sgrnas=control_sgrnas,
            norm_method=norm_method,
            pdf_report=pdf_report,
            other_parameter=other_parameter,
        )
        shutil.copyfile(count_table, outfile)

    return FileGeneratingJob(outfile, __dump).depends_on(dependencies)


def mageck_rra_job(
    count_table: Union[Path, str],
    treatment_ids: List[str],
    control_ids: List[str],
    out_dir: Union[Path, str],
    prefix: str,
    control_sgrnas: Optional[Union[Path, str]],
    norm_method: str = None,
    paired: bool = False,
    pdf_report: bool = False,
    other_parameter: Dict[str, str] = [],
    dependencies: List[Job] = [],
):
    outfile = Path(out_dir) / f"{prefix}.gene_summary.tsv"

    def __dump(
        outfile,
        count_table=count_table,
        treatment_ids=treatment_ids,
        control_ids=control_ids,
        out_dir=out_dir,
        prefix=prefix,
        control_sgrnas=control_sgrnas,
        norm_method=norm_method,
        paired=paired,
        pdf_report=pdf_report,
        other_parameter=other_parameter,
    ):
        mageck_test(
            count_table=count_table,
            treatment_ids=treatment_ids,
            control_ids=control_ids,
            out_dir=out_dir,
            prefix=prefix,
            control_sgrnas=control_sgrnas,
            norm_method=norm_method,
            paired=paired,
            pdf_report=pdf_report,
            other_parameter=other_parameter,
        )

    return FileGeneratingJob(outfile, __dump).depends_on(dependencies)


def mageck_mle_job(
    count_table: Union[Path, str],
    design_matrix: str,
    out_dir: Union[Path, str],
    prefix: str,
    control_sgrnas: Optional[Union[Path, str]] = None,
    norm_method: str = None,
    other_parameter: Dict[str, str] = [],
    dependencies: List[Job] = [],
):
    outfile = Path(out_dir) / f"{prefix}.gene_summary.tsv"

    def __dump(
        outfile,
        count_table=count_table,
        design_matrix=design_matrix,
        out_dir=out_dir,
        prefix=prefix,
        control_sgrnas=control_sgrnas,
        norm_method=norm_method,
        other_parameter=other_parameter,
    ):
        mageck_mle(
            count_table=count_table,
            design_matrix=design_matrix,
            out_dir=out_dir,
            prefix=prefix,
            control_sgrnas=control_sgrnas,
            norm_method=norm_method,
            other_parameter=other_parameter,
        )

    return FileGeneratingJob(outfile, __dump).depends_on(dependencies)


def create_query_control_sgrna_frames_job(
    infile: Path,
    outfiles: Tuple[Path],
    control_prefix: str,
    id_col: Optional[str] = None,
    name_column: str = "name",
    sgRNA_column: str = "sgRNA",
    infer_genes: Optional[Callable] = None,
    read_csv_kwargs: Optional[Dict] = None,
    dependencies: List[Job] = [],
):
    def __dump(
        outfiles,
        infile=infile,
        control_prefix=control_prefix,
        id_col=id_col,
        name_column=name_column,
        sgRNA_column=sgRNA_column,
        infer_genes=infer_genes,
        read_csv_kwargs=read_csv_kwargs,
    ):

        create_query_control_sgrna_frames(
            infile=infile,
            outfiles=outfiles,
            control_prefix=control_prefix,
            id_col=id_col,
            name_column=name_column,
            sgRNA_column=sgRNA_column,
            infer_genes=infer_genes,
            read_csv_kwargs=read_csv_kwargs,
        )

    return MultiFileGeneratingJob(outfiles, __dump).depends_on(
        dependencies
        + [
            ParameterInvariant(
                f"PI_{outfiles[0].name}",
                [
                    str(infile),
                    control_prefix,
                    name_column,
                    sgRNA_column,
                    read_csv_kwargs,
                ],
            )
        ]
    )


def create_combine_gene_info_with_mageck_output_job(
    mageck_file: Path,
    gene_info_file: Path,
    output_file: Path,
    name_column_mageck: str = "id",
    name_column_genes: str = "Gene",
    how: str = "inner",
    columns_to_add: List[str] = [
        "gene_stable_id",
        "name",
        "chr",
        "start",
        "stop",
        "strand",
        "biotype",
    ],
    read_csv_kwargs: Optional[Dict] = None,
    dependencies: List[Job] = [],
):
    def __dump(
        output_file,
        mageck_file=mageck_file,
        gene_info_file=gene_info_file,
        name_column_mageck=name_column_mageck,
        name_column_genes=name_column_genes,
        how=how,
        columns_to_add=columns_to_add,
        read_csv_kwargs=read_csv_kwargs,
    ):
        create_combine_gene_info_with_mageck_output(
            mageck_file=mageck_file,
            gene_info_file=gene_info_file,
            output_file=output_file,
            name_column_mageck=name_column_mageck,
            name_column_genes=name_column_genes,
            how=how,
            columns_to_add=columns_to_add,
            read_csv_kwargs=read_csv_kwargs,
        )

    return FileGeneratingJob(output_file, __dump).depends_on(dependencies)


def create_spiked_count_table_job(
    outfile: Union[Path, str],
    count_file: Union[Path, str],
    replicate_of: Dict[str, str],
    sample_to_group: Dict[str, str],
    group_contrast: Tuple[str] = ("sorted", "total"),
    n_genes: int = 20,
    log_effect: float = 2.0,
    baseline_mean: float = 300.0,
    dispersion: float = 0.08,
    dependencies: List[Job] = [],
) -> DataFrame:

    def __dump(
        outfile,
        count_file=count_file,
        replicate_of=replicate_of,
        sample_to_group=sample_to_group,
        group_contrast=group_contrast,
        n_genes=n_genes,
        log_effect=log_effect,
        baseline_mean=baseline_mean,
        dispersion=dispersion,
    ):
        write_spiked_count_table(
            outfile=outfile,
            count_file=count_file,
            replicate_of=replicate_of,
            sample_to_group=sample_to_group,
            group_contrast=group_contrast,
            n_genes=n_genes,
            log_effect=log_effect,
            baseline_mean=baseline_mean,
            dispersion=dispersion,
        )

    return FileGeneratingJob(outfile, __dump).depends_on(dependencies)


def evaluate_spike_in_performance_job(
    output_file: Union[Path, str],
    mageck_results: Dict[str, Path],
    # direction: str = "neg",
    combine_directions: bool = True,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
    gene_col: str = "id",
    weights: Optional[Dict[str, float]] = None,
    dependencies: List[Job] = [],
) -> Job:
    """
    Evaluate and compare multiple MAGeCK runs with spike-in controls.

    This job calculates comprehensive quality metrics for spike-in detection:
    - Precision, Recall, F1 score
    - AUC-ROC and AUC-PR
    - Separation metrics (Cohen's d, effect sizes)
    - Ranking power (AUCC, top-k enrichment)
    - Consistency (CV, IQR within spike-in groups)

    Parameters
    ----------
    output_file : Union[Path, str]
        Output TSV file with evaluation metrics
    mageck_results : Dict[str, Path]
        Dictionary mapping comparison names to gene_summary.tsv files
    direction : str
        "neg" for negative selection, "pos" for positive selection
    combine_directions : bool
        If True, add aggregated "both" rows combining pos and neg per comparison
    fdr_threshold : float
        FDR threshold for significance calling
    lfc_threshold : float
        Absolute log fold change threshold
    gene_col : str
        Column name for gene IDs (default: "id")
    weights : Optional[Dict[str, float]]
        Weights for composite score calculation (default: None = equal weights)
    dependencies : List[Job]
        Job dependencies

    Returns
    -------
    Job
        FileGeneratingJob that creates the evaluation report

    Examples
    --------
    >>> eval_job = evaluate_spike_in_performance_job(
    ...     output_file="results/spike_evaluation.tsv",
    ...     mageck_results={
    ...         "RRA_paired_median": Path("results/.../gene_summary.tsv"),
    ...         "RRA_paired_control": Path("results/.../gene_summary.tsv"),
    ...         "RRA_unpaired_median": Path("results/.../gene_summary.tsv"),
    ...     },
    ...     direction="neg",
    ...     dependencies=[rra1, rra2, rra3],
    ... )
    """

    def __dump(
        output_file,
        mageck_results=mageck_results,
        # direction=direction,
        combine_directions=combine_directions,
        fdr_threshold=fdr_threshold,
        lfc_threshold=lfc_threshold,
        gene_col=gene_col,
        weights=weights,
    ):
        # Evaluate all methods
        eval_df = evaluate_multiple_mageck_results(
            results_dict=mageck_results,
            # direction=direction,
            combine_directions=combine_directions,
            fdr_threshold=fdr_threshold,
            lfc_threshold=lfc_threshold,
            gene_col=gene_col,
        )
        if len(eval_df) == 0:
            raise ValueError("No valid MAGeCK results found for evaluation")

        # Rank methods
        ranked_df = rank_mageck_methods(eval_df, weights=weights)

        # Save to file
        ranked_df.to_csv(output_file, sep="\t", index=False)

        # Print summary to stdout
        print("\n" + "=" * 80)
        print("SPIKE-IN EVALUATION SUMMARY")
        print("=" * 80)
        print(f"FDR threshold: {fdr_threshold}")
        print(f"LFC threshold: {lfc_threshold}")
        print(f"\nNumber of methods evaluated: {len(ranked_df)}")
        # print(f"\nDirection: {direction} selection")

        print("\n" + "-" * 80)
        print("TOP 3 METHODS (by composite score):")
        print("-" * 80)

        for idx, row in ranked_df.head(3).iterrows():
            print(f"\n{int(row['rank'])}. {row['comparison']}")
            print(
                f"   Composite Score: {row.get('composite_score', np.nan):.3f}"
            )
            print(
                f"   F1: {row.get('f1', np.nan):.3f} | "
                f"Precision: {row.get('precision', np.nan):.3f} | "
                f"Recall: {row.get('recall', np.nan):.3f}"
            )
            print(
                f"   AUC-ROC: {row.get('auc_roc', np.nan):.3f} | "
                f"AUCC: {row.get('aucc', np.nan):.3f}"
            )
            print(f"   Median Rank: {row.get('median_rank', np.nan):.1f}")
            print(f"   Final Score: {row.get('final_score', np.nan):.3f}")
            # Detection stats
            n_detected = row.get("n_detected_hits", 0)
            n_expected = row.get("n_expected_hits", 0)
            print(
                f"   Detected: {int(n_detected)}/{int(n_expected)} expected hits"
            )

            # Separation
            cohens_d = row.get(
                "neg_vs_neutral_cohens_d",
                row.get("pos_vs_neutral_cohens_d", np.nan),
            )
            print(f"   Separation (Cohen's d): {cohens_d:.2f}")

        print("\n" + "=" * 80)
        print(f"Full results saved to: {output_file}")
        print("=" * 80 + "\n")

    return FileGeneratingJob(output_file, __dump).depends_on(dependencies)


def spike_evaluation_report_job(
    output_dir: Union[Path, str],
    evaluation_table: Union[Path, str],
    prefix: str = "spike_evaluation",
    top_n: int = 5,
    pdf_report: bool = True,
    dependencies: List[Job] = [],
) -> MultiFileGeneratingJob:
    """Job that creates a markdown (and optional PDF) report interpreting
    the spike-in evaluation table created by `evaluate_spike_in_performance_job`.

    Parameters
    ----------
    output_dir : Union[Path, str]
        Directory to save report outputs
    evaluation_table : Union[Path, str]
        Path to TSV produced by `evaluate_spike_in_performance_job`
    prefix : str
        Prefix for output files (default: 'spike_evaluation')
    top_n : int
        Number of top methods to show in the report
    pdf_report : bool
        Whether to attempt generating a PDF
    dependencies : List[Job]
        Job dependencies

    Returns
    -------
    MultiFileGeneratingJob
    """
    outfiles = [
        Path(output_dir) / f"{prefix}.md",
        Path(output_dir) / f"{prefix}_scores.png",
    ]
    if pdf_report:
        outfiles.append(Path(output_dir) / f"{prefix}.pdf")

    def __dump(
        outfiles,
        output_dir=output_dir,
        evaluation_table=evaluation_table,
        prefix=prefix,
        top_n=top_n,
        pdf_report=pdf_report,
    ):
        generate_spike_evaluation_report(
            eval_table=evaluation_table,
            output_dir=output_dir,
            prefix=prefix,
            top_n=top_n,
            pdf_report=pdf_report,
        )

    job = MultiFileGeneratingJob(outfiles, __dump).depends_on(dependencies)

    job.depends_on(
        FunctionInvariant(
            f"{prefix}_generate_spike_evaluation_report",
            generate_spike_evaluation_report,
        )
    )

    job.depends_on(
        ParameterInvariant(
            f"{prefix}_spike_params",
            (
                str(evaluation_table),
                prefix,
                top_n,
                pdf_report,
            ),
        )
    )

    return job


def write_significant_genes_rra_job(
    mageck_file: Union[str, Path],
    fdr_threshold: float = 0.05,
    logfc_threshold: float = 1.0,
    direction: str = "both",  # "both", "pos", "neg"
    dependencies: List[Job] = [],
) -> MultiFileGeneratingJob:
    """
    write_significant_genes_rra_job returns a job to write significant genes for
    RRA results from mageck.

    mageck_file : Union[str, Path]
        The input mageck gene summary file.
    fdr_threshold : float, optional
        FDR threshold, by default 0.05
    logfc_threshold : float, optional
        LogFC threshold, by default 1.0
    direction : str, optional
        Direction of change: "both", "pos", "neg", by default "both"

    Returns
    -------
    MultiFileGeneratingJob
        Job that creates output files for enriched and depleted genes.
    """
    outfiles = [
        mageck_file.with_suffix(".enriched.genes.tsv"),
        mageck_file.with_suffix(".depleted.genes.tsv"),
    ]

    def __dump(
        outfiles,
        mageck_file=mageck_file,
        fdr_threshold=fdr_threshold,
        logfc_threshold=logfc_threshold,
        direction=direction,
    ):
        write_significant_genes_rra(
            mageck_file=mageck_file,
            fdr_threshold=fdr_threshold,
            logfc_threshold=logfc_threshold,
            direction=direction,
        )

    return MultiFileGeneratingJob(outfiles, __dump).depends_on(dependencies)


def write_significant_genes_job(
    mageck_file: Union[str, Path],
    fdr_threshold: float = 0.05,
    logfc_or_beta_threshold: float = 1.0,
    direction: str = "both",  # "both", "pos", "neg"
    fdr_column_pos: str = "pos|fdr",
    fdr_column_neg: Optional[str] = "neg|fdr",
    lfc_column_pos: str = "pos|lfc",
    lfc_column_neg: Optional[str] = "neg|lfc",
    method: str = "rra",
    dependencies: List[Job] = [],
) -> MultiFileGeneratingJob:
    """
    write_significant_genes_rra_job returns a job to write significant genes for
    RRA results from mageck.

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
    MultiFileGeneratingJob
        Job that creates output files for enriched and depleted genes.
    """
    outfiles = [
        mageck_file.with_suffix(".enriched.genes.tsv"),
        mageck_file.with_suffix(".depleted.genes.tsv"),
    ]

    def __dump(
        outfiles,
        mageck_file=mageck_file,
        fdr_threshold=fdr_threshold,
        logfc_or_beta_threshold=logfc_or_beta_threshold,
        fdr_column_pos=fdr_column_pos,
        fdr_column_neg=fdr_column_neg,
        lfc_column_pos=lfc_column_pos,
        lfc_column_neg=lfc_column_neg,        
        direction=direction,
        method=method,
    ):
        write_significant_genes_mageck(
            mageck_file=mageck_file,
            fdr_threshold=fdr_threshold,
            logfc_or_beta_threshold=logfc_or_beta_threshold,
            fdr_column_pos=fdr_column_pos,
            fdr_column_neg=fdr_column_neg,
            lfc_column_pos=lfc_column_pos,
            lfc_column_neg=lfc_column_neg,        
            direction=direction,
            method=method,
        )


    return MultiFileGeneratingJob(outfiles, __dump).depends_on(dependencies)


def mageck_filter_count_job(
    count_table_file: Union[str, Path],
    samples: Iterable[str],
    outfile: Union[str, Path],
    conditions: Dict[str, Any],
    baseline_samples: Iterable[str],
    aggregations: Optional[Dict[str, Tuple[List[str], Callable]]] = None,
    exclude_samples: Optional[Iterable[str]] = None,
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    dependencies: List[Job] = [],
) -> Job:
    """
    mageck_filter_count_job returns a job to filter MAGeCK count table based on
    the number of baseline samples with minimum count.


    Parameters
    ----------
    count_table_file : Union[str, Path]
        file to count table
    samples : Iterable[str]
        Names of all sample columns (raw counts) like
        ["Total1","Total2","Total3","Sorted1","Sorted2","Sorted3"].
    outfile : Union[str, Path]
        new file to write filtered count table
    conditions : Dict[str, Any]
        Dict defining filtering rules. Supported keys:

            Column thresholds (simple):
            - "col_filters": list of dicts, each with:
                {"col": <column name>, "op": one of {"<","<="," >",">=","==","!="}, "value": number}
                Example:
                {"col_filters": [{"col": "Total_Rep1", "op": ">=", "value": 2}]}

            Baseline detectability:
            - "baseline_min_count": int (e.g. 20)
            - "baseline_min_n": int (e.g. 2)  # at least this many baseline samples pass the count
                Example:
                {"baseline_min_count": 20, "baseline_min_n": 2}

            Missing handling:
            - "na_policy": "drop" or "keep" (default "drop")
                "drop": rows with NA in any used filter metric are removed
    baseline_samples : Iterable[str]
        Names of baseline sample columns (raw counts) like
        ["Total1","Total2","Total3"].
    aggregations : Optional[Dict[str,, optional
        Optional dict defining new aggregated columns to create prior to filtering.
        You can specify in aggregations a new column name, a list of existing
        columns to aggregate and a function to use for aggregation.
        Supported functions are any pandas DataFrame aggregation functions like
        min, max, mean, median, etc., by default None
    exclude_samples : Optional[Iterable[str]], optional
        Samples to exclude (both from raw counts and any derived columns
        mentioning them), by default None
    sgrna_col : str, optional
        Name of the sgRNA column, by default "sgRNA"
    gene_col : str, optional
        Name of the gene column, by default "Gene"
    dependencies : List[Job], optional
        job dependencies, by default []

    Returns
    -------
    Job
        Job that creates filtered MAGeCK count table.
    """
    outfiles = [outfile, Path(outfile.with_suffix(".full.tsv"))]

    def __dump(
        outfiles,
        count_table_file=count_table_file,
        samples=samples,
        conditions=conditions,
        baseline_samples=baseline_samples,
        aggregations=aggregations,
        exclude_samples=exclude_samples,
        sgrna_col=sgrna_col,
        gene_col=gene_col,
    ):
        write_filter_mageck_counts(
            count_table_file=count_table_file,
            samples=samples,
            outfile=outfiles[0],
            conditions=conditions,
            baseline_samples=baseline_samples,
            aggregations=aggregations,
            exclude_samples=exclude_samples,
            sgrna_col=sgrna_col,
            gene_col=gene_col,
        )

    return MultiFileGeneratingJob(outfiles, __dump).depends_on(dependencies)


def write_count_table_with_MA_CPM_job(
    count_tsv: Union[str, Path],
    outfile: Union[str, Path],
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
    dependencies: List[Job] = [],
) -> Job:
    """
    write_count_table_with_MA_CPM to add MA and CPM columns to count table.

    Parameters
    ----------
    count_tsv : Union[str, Path]
        Input count table file.
    outdir : Union[str, Path]
        Output directory to save the modified count table.
    method : str, optional
        Normalization method for size factor calculation, by default "median"
    pseudocount : float, optional
        Pseudocount to add when calculating logCPM, by default 1.0
    paired_replicates : bool, optional
        Whether replicates are paired, by default False
    conditions_dict : Dict[str, List[str]], optional
        Dictionary mapping condition names to sample names, by default {}
    baseline_condition : str, optional
        Name of the baseline condition in conditions_dict, by default "total"
    control_sgrna_txt : Optional[Union[Path, str]], optional
        Path to control sgRNA file, by default None
    sgrna_col : str, optional
        Column name for sgRNA identifiers, by default "sgRNA"
    gene_col : str, optional
        Column name for gene identifiers, by default "Gene"
    delimiter : str, optional
        Delimiter used in sample names to separate conditions, by default "_"
    suffix : str, optional
        uffix for the output file name, by default "CPM_MA"
    dependencies : List[Job], optional
        Job dependencies, by default []

    Returns
    -------
    Job
        Job that creates the count table with added MA and CPM columns.
    """
    outfile = Path(outfile)

    def __dump(
        outfile,
        count_tsv=count_tsv,
        method=method,
        pseudocount=pseudocount,
        paired_replicates=paired_replicates,
        conditions_dict=conditions_dict,
        baseline_condition=baseline_condition,
        control_sgrna_txt=control_sgrna_txt,
        sgrna_col=sgrna_col,
        gene_col=gene_col,
        delimiter=delimiter,
        suffix=suffix,
    ):
        write_count_table_with_MA_CPM(
            count_tsv=count_tsv,
            method=method,
            pseudocount=pseudocount,
            paired_replicates=paired_replicates,
            conditions_dict=conditions_dict,
            baseline_condition=baseline_condition,
            control_sgrna_txt=control_sgrna_txt,
            sgrna_col=sgrna_col,
            gene_col=gene_col,
            delimiter=delimiter,
            suffix=suffix,
        )

    return FileGeneratingJob(outfile, __dump).depends_on(dependencies)
