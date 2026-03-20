from pypipegraph2 import (
    Job,
    MultiFileGeneratingJob,
    FileGeneratingJob,
    FunctionInvariant,
    ParameterInvariant,
)
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
from crisprscreens.services.plots_io import (
    write_venn,
    write_volcano_plot,
    create_plot_ranking_metric_heatmaps,
)
from crisprscreens.core.plots import volcano_plot
from pandas import DataFrame  # noqa F401


def write_venn_job(
    outdir: Union[Path, str],
    filebasename: str,
    label_to_file: Dict[str, str],
    id_cols: Union[List[str], str] = "id",
    sep: str = "\t",
    figsize: Tuple[float, float] = (6, 6),
    title: str | None = None,
    dependencies: List[Job] = [],
):
    out_venn = outdir / f"{filebasename}_venn.png"
    out_dataframe = outdir / f"{filebasename}.tsv"

    def __dump(
        outfiles,
        outdir=outdir,
        filebasename=filebasename,
        label_to_file=label_to_file,
        id_cols=id_cols,
        sep=sep,
        figsize=figsize,
        title=title,
    ):
        write_venn(
            outdir=outdir,
            filebasename=filebasename,
            label_to_file=label_to_file,
            id_cols=id_cols,
            sep=sep,
            figsize=figsize,
            title=title,
        )

    return MultiFileGeneratingJob([out_venn, out_dataframe], __dump).depends_on(
        dependencies
    )


def write_volcano_plot_job(
    filename: Union[Path, str],
    folder: Union[Path, str],
    df: Union[Path, str, DataFrame],
    log_fc_column: str,
    y_column: Union[str, Tuple[str, str]],
    fdr_column: Optional[Union[str, Tuple[str, str]]] = None,
    name_column: Optional[str] = None,
    top_n_labels: int = 0,
    transform_y: bool = True,  # True -> plot -log10(FDR), False -> plot raw FDR
    log_threshold: float = 1.0,  # abs(logFC) threshold
    fdr_threshold: float = 0.05,  # FDR threshold (always on raw FDR scale)
    point_size: float = 12,
    alpha: float = 0.75,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    y_clip_min: float = 1e-300,  # avoids -log10(0)
    y_clip_max: Optional[
        float
    ] = None,  # clips extreme y-values, shows as triangles
    label_fontsize: int = 9,
    dependencies: List[Job] = [],
):
    def __dump(
        outfiles,
        filename=Path(filename).stem,
        folder=folder,
        df=df,
        log_fc_column=log_fc_column,
        y_column=y_column,
        fdr_column=fdr_column,
        name_column=name_column,
        top_n_labels=top_n_labels,
        transform_y=transform_y,
        log_threshold=log_threshold,
        fdr_threshold=fdr_threshold,
        point_size=point_size,
        alpha=alpha,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        y_clip_min=y_clip_min,
        y_clip_max=y_clip_max,
        label_fontsize=label_fontsize,
    ):
        write_volcano_plot(
            filename=filename,
            folder=folder,
            df=df,
            log_fc_column=log_fc_column,
            y_column=y_column,
            fdr_column=fdr_column,
            name_column=name_column,
            top_n_labels=top_n_labels,
            transform_y=transform_y,
            log_threshold=log_threshold,
            fdr_threshold=fdr_threshold,
            point_size=point_size,
            alpha=alpha,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            y_clip_min=y_clip_min,
            y_clip_max=y_clip_max,
            label_fontsize=label_fontsize,
        )

    io_func_invariant = FunctionInvariant(
        f"write_volcano_plot_{filename}_{str(folder)}", write_volcano_plot
    )
    core_func_invariant = FunctionInvariant(
        f"volcano_plot_{filename}_{str(folder)}", volcano_plot
    )
    params = ParameterInvariant(
        f"write_volcano_plot_params{filename}_{str(folder)}",
        [
            filename,
            str(folder),
            log_fc_column,
            fdr_column,
            name_column,
            top_n_labels,
            transform_y,
            log_threshold,
            fdr_threshold,
            point_size,
            alpha,
            title,
            xlabel,
            ylabel,
            figsize,
            y_clip_min,
            y_clip_max,
            label_fontsize,
        ],
    )
    return (
        FileGeneratingJob(Path(folder) / filename, __dump)
        .depends_on(dependencies)
        .depends_on([io_func_invariant, core_func_invariant, params])
    )


def plot_ranking_metric_heatmaps_job(
    output_file: Union[Path, str],
    metrics_df_file: Union[str, Path],
    metrics: List[str] = ("kendall_tau", "spearman_r", "dcg"),
    figsize: tuple[float, float] | None = None,
    dependencies: Optional[List[Job]] = None,
) -> Job:
    """
    plot_ranking_metric_heatmaps_job creates a job that generates heatmaps for
    the specified ranking metrics and saves the figure to the specified output
    file. It depends on the create_plot_ranking_metric_heatmaps function and
    any additional dependencies provided.

    Parameters
    ----------
    output_file : Union[Path, str]
        The path to the output file where the heatmap will be saved.
    metrics_df_file : Union[Str, Path]
        The path to the input DataFrame file containing the ranking metrics.
    metrics : List[str], optional
        The ranking metrics to include in the heatmap, by default
        ("kendall_tau", "spearman_r", "dcg").
    figsize : tuple[float, float] | None, optional
        The size of the figure to create, by default None.
    dependencies : Optional[List[Job]], optional
        Any additional dependencies for the job, by default None.

    Returns
    -------
    Job
        The job that generates the heatmap.
    """

    def __dump(
        outfiles,
        output_file=output_file,
        metrics_df_file=metrics_df_file,
        metrics=metrics,
        figsize=figsize,
    ):
        create_plot_ranking_metric_heatmaps(
            output_file=output_file,
            metrics_df_file=metrics_df_file,
            metrics=metrics,
            figsize=figsize,
        )

    if dependencies is None:
        dependencies = []

    return (
        FileGeneratingJob(Path(output_file), __dump)
        .depends_on(
            [
                FunctionInvariant(
                    "create_plot_ranking_metric_heatmaps",
                    create_plot_ranking_metric_heatmaps,
                )
            ]
        )
        .depends_on(dependencies)
    )
