import pandas as pd
from crisprscreens.core.plots import (
    plot_selected_venn,
    volcano_plot,
    plot_ranking_metric_heatmaps,
)
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
from pandas import DataFrame
from .io import save_figure, read_dataframe


def write_venn(
    outdir: Union[Path, str],
    filebasename: str,
    label_to_file: Dict[str, str],
    id_cols: Union[List[str], str] = "id",
    sep: str = "\t",
    figsize: Tuple[float, float] = (6, 6),
    title: str | None = None,
):
    outdir = Path(outdir)
    outdir.parent.mkdir(exist_ok=True, parents=True)
    out_venn = f"{filebasename}_venn"
    out_dataframe = outdir / f"{filebasename}.tsv"
    results = plot_selected_venn(
        label_to_file=label_to_file,
        id_cols=id_cols,
        sep=sep,
        figsize=figsize,
        title=title,
    )
    save_figure(results["figure"], outdir, out_venn)
    results["memberships"].to_csv(out_dataframe, sep="t", index=False)


def write_volcano_plot(
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
    fdr_threshold: float = 0.05,  # FDR threshold (always interpreted on raw FDR scale)
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
):
    if isinstance(df, str) or isinstance(df, Path):
        df = read_dataframe(df)
    elif isinstance(df, DataFrame):
        pass
    else:
        raise TypeError("df must be a DataFrame or a path to a file")
    fig, axes = volcano_plot(
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
    save_figure(fig, Path(folder), str(filename))


def create_plot_ranking_metric_heatmaps(
    output_file: Union[Path, str],
    metrics_df_file: Union[str, Path],
    metrics: List[str] = ("kendall_tau", "spearman_r", "dcg"),
    figsize: tuple[float, float] | None = None,
) -> Path:
    """
    create_plot_ranking_metric_heatmaps creates heatmaps for the specified
    ranking metrics and saves the figure to the specified output file.

    Parameters
    ----------
    output_file : Union[Path, str]
        Path to the output file where the figure will be saved.
    metrics_df : pd.DataFrame
        DataFrame containing the ranking metrics.
    metrics : List[str], optional
        List of ranking metrics to plot, by default ("kendall_tau", "spearman_r", "dcg")
    figsize : tuple[float, float] | None, optional
        Figure size, by default None

    Returns
    -------
    Path
        _description_
    """
    metrics_df = pd.read_csv(metrics_df_file, sep="\t")
    fig = plot_ranking_metric_heatmaps(
        metrics_df=metrics_df,
        metrics=metrics,
        figsize=figsize,
    )
    save_figure(fig, Path(output_file).parent, Path(output_file).stem)
    return Path(output_file)
