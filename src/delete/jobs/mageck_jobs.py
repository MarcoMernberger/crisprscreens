from pypipegraph2 import Job, FileGeneratingJob, MultiFileGeneratingJob
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Literal
from crisprscreens.services.mageck_io import (
    write_filtered_mageck_comparison,
    combine_comparison_output,
)
from crisprscreens.r_integration.mageck_wrapper import run_mageck_scatterview


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
    ),
    select: Optional[
        Literal["positive", "negative", "both", "none"]
    ] = "negative",
    neg_effect_cutoff: float = -0.4,
    pos_effect_cutoff: float = 0.4,
    delta_cutoff_k: float = 2,
    fdr_x_col: Optional[str] = None,  # FDR column for x (optional)
    fdr_y_col: Optional[str] = None,  # FDR column for y (optional)
    fdr_cutoff: float = 0.05,  # FDR cutoff
    # plot parameters
    toplabels: bool = True,
    select_label: bool = False,
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
        fdr_x_col=fdr_x_col,
        fdr_y_col=fdr_y_col,
        fdr_cutoff=fdr_cutoff,
        toplabels=toplabels,
        select_label=select_label,
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
            fdr_x_col=fdr_x_col,
            fdr_y_col=fdr_y_col,
            fdr_cutoff=fdr_cutoff,
            toplabels=toplabels,
            select_label=select_label,
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
