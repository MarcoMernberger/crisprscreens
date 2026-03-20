import rpy2.robjects as ro

from rpy2.robjects.vectors import StrVector, FloatVector
from rpy2.rinterface import NULL
from pathlib import Path
from typing import Tuple, Optional, Union, Literal, Iterable


# path to the mageck R functions
HERE = Path(__file__).resolve().parent  # .../src/r_integration
SRC_ROOT = HERE.parent  # .../src
MAGECK_R_PATH = SRC_ROOT / "r" / "mageck.R"  # .../src/r/mageck.R


def _load_run_mageck_scatterview():
    """
    Source src/r/mageck.R and return RunMageckScatterView
    """
    if not MAGECK_R_PATH.exists():
        raise FileNotFoundError(f"mageck.R not found at: {MAGECK_R_PATH}")

    r_source = ro.r["source"]
    r_source(str(MAGECK_R_PATH))

    return ro.globalenv["RunMageckScatterView"]


# load on import
RunMageckScatterView_R = _load_run_mageck_scatterview()


# Wrapper functions


def run_mageck_scatterview(
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
):
    """
    Python wrapper for R-function RunMageckScatterView from src/r/mageck.R.
    returns R-list object
    """
    if x_col is None or y_col is None:
        raise ValueError("x_col and y_col must be provided.")

    r_groups = StrVector(list(groups))
    r_x_cut = FloatVector(list(x_cut)) if x_cut is not None else NULL
    r_y_cut = FloatVector(list(y_cut)) if y_cut is not None else NULL
    r_xlab = xlab if xlab is not None else NULL
    r_ylab = ylab if ylab is not None else NULL
    input_file = str(input_file)
    output_dir = str(output_dir)
    if isinstance(toplabels, Iterable):
        toplabels = StrVector(toplabels)

    res = RunMageckScatterView_R(
        input_file=input_file,
        x_col=x_col,
        y_col=y_col,
        output_dir=output_dir,
        filebase_name=filebase_name,
        data_type=datatype,
        gene_col=gene_col,
        sep=sep,
        normalize=normalize,
        top=top,
        groups=r_groups,
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
        xlab=r_xlab,
        ylab=r_ylab,
        jpeg_width=jpeg_width,
        jpeg_height=jpeg_height,
        auto_cut_diag=auto_cut_diag,
        auto_cut_x=auto_cut_x,
        auto_cut_y=auto_cut_y,
        x_cut=r_x_cut,
        y_cut=r_y_cut,
    )
    return res
