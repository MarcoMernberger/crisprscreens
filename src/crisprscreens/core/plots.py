import itertools
import warnings
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Axes, Figure  # type: ignore[import]
from pandas import DataFrame
from pypipegraph2 import FileGeneratingJob, Job  # type: ignore[import]
from sklearn.decomposition import PCA

try:
    from matplotlib_venn import venn2, venn3

    _HAS_MPL_VENN = True
except ImportError:
    _HAS_MPL_VENN = False


def plot_selected_venn(
    label_to_file: Dict[str, str],
    id_cols: Union[List[str], str] = "id",
    sep: str = "\t",
    figsize: Tuple[float, float] = (6, 6),
    title: str | None = None,
):
    """
    Plot Venn/UpSet-style diagram from 2–4 selected.tsv files.

    Parameters
    ----------
    label_to_file : dict
        Mapping label -> path to selected.tsv
        (values are read with pandas.read_csv(..., sep=sep)).
    id_col : str
        Column name that contains gene IDs.
    sep : str
        Separator for the TSV files.
    figsize : (float, float)
        Figure size for matplotlib.
    title : str or None
        Optional plot title.

    Returns
    -------
    result : dict
        {
          "sets": dict[label -> set(ids)],
          "memberships": pd.DataFrame with boolean membership per label,
          "figure": matplotlib.figure.Figure
        }
    """
    if not (2 <= len(label_to_file) <= 4):
        raise ValueError("Please provide between 2 and 4 files (labels).")

    sets: Dict[str, set] = {}
    ii = 0
    for label, path in label_to_file.items():
        if isinstance(id_cols, str):
            id_col = id_cols
        else:
            id_col = id_cols[ii]
        df = pd.read_csv(path, sep=sep)
        if id_col not in df.columns:
            raise ValueError(
                f"Column '{id_col}' not found in file '{path}'. "
                f"Available columns: {list(df.columns)}"
            )
        ids = set(df[id_col].dropna().astype(str))
        sets[label] = ids
        ii += 1

    labels = list(sets.keys())
    n = len(labels)

    all_ids = sorted(set().union(*sets.values()))
    membership_data = {"id": all_ids}
    for label in labels:
        membership_data[label] = [gene in sets[label] for gene in all_ids]

    memberships = pd.DataFrame(membership_data).set_index("id")

    fig = plt.figure(figsize=figsize)

    if n in (2, 3) and _HAS_MPL_VENN:
        ax = fig.add_subplot(111)
        set_list = [sets[ll] for ll in labels]

        if n == 2:
            venn2(set_list, set_labels=labels, ax=ax)
        else:
            venn3(set_list, set_labels=labels, ax=ax)

        if title is not None:
            ax.set_title(title)

    else:
        ax = fig.add_subplot(111)

        combos: Iterable[Tuple[str, ...]] = []
        for r in range(1, n + 1):
            combos += itertools.combinations(labels, r)

        combo_counts = []
        combo_labels = []
        for combo in combos:
            mask = memberships[list(combo)].all(axis=1)
            count = mask.sum()
            if count > 0:
                combo_counts.append(count)
                combo_labels.append("&".join(combo))

        order = sorted(
            range(len(combo_counts)),
            key=lambda i: combo_counts[i],
            reverse=True,
        )

        combo_counts = [combo_counts[i] for i in order]
        combo_labels = [combo_labels[i] for i in order]

        ax.bar(range(len(combo_counts)), combo_counts)
        ax.set_xticks(range(len(combo_labels)))
        ax.set_xticklabels(combo_labels, rotation=45, ha="right")
        ax.set_ylabel("Number of genes")
        if title is None:
            ax.set_title("Set intersections (UpSet-style)")
        else:
            ax.set_title(title)

        fig.tight_layout()

    result = {
        "sets": sets,
        "memberships": memberships,
        "figure": fig,
    }
    return result


def volcano_plot(
    df: pd.DataFrame,
    log_fc_column: str,
    y_column: Union[str, Tuple[str, str]],
    fdr_column: Optional[Union[str, Tuple[str, str]]] = None,
    *,
    name_column: Optional[str] = None,
    top_n_labels: int = 0,
    transform_y: bool = True,  # True -> plot -log10(FDR)
    log_threshold: float = 1.0,
    fdr_threshold: float = 0.05,
    point_size: float = 12,
    alpha: float = 0.75,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    y_clip_min: float = 1e-300,
    y_clip_max: Optional[float] = None,
    label_fontsize: int = 9,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Volcano plot supporting either:
      - a single p or FDR column (MLE-style), or
      - separate p or FDR columns for positive/negative effects (RRA-style).

    If fdr_column is a tuple:
        (fdr_pos, fdr_neg)
    then:
        logFC >= 0 -> fdr_pos
        logFC <  0 -> fdr_neg

    Parameters
    ----------
    df : pd.DataFrame
        Data with logFC and FDR columns.
    log_fc_column : str
        Column name for log fold-change.
    y_column : str
        Y column name, may be p-value or fdr (fdr_pos, fdr_neg) for RRA-style.
    fdr_column : str or tuple
        FDR column name, or (fdr_pos, fdr_neg) for RRA-style.
    name_column : str, optional
        Column for gene/sgRNA names (for labeling).
    top_n_labels : int
        Number of top hits to label.
    transform_y : bool
        If True, plot -log10(FDR); if False, plot raw FDR.
    log_threshold : float
        Absolute log fold-change threshold for significance.
    fdr_threshold : float
        FDR threshold for significance (always on raw scale).
    point_size : float
        Size of scatter points.
    alpha : float
        Transparency of points.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple
        Figure size.
    y_clip_min : float
        Minimum y-value for clipping (avoids -log10(0)).
    y_clip_max : float, optional
        Maximum y-value for clipping. Values above this are shown as
        upward-facing triangles at y_clip_max to indicate truncation.
    label_fontsize : int
        Font size for point labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if fdr_column is None:
        fdr_column = y_column
    print(y_column, fdr_column)
    if log_fc_column not in df.columns:
        raise KeyError(f"Missing column: {log_fc_column}")

    if isinstance(fdr_column, tuple):
        if len(fdr_column) != 2:
            raise ValueError("fdr_column tuple must be (pos_fdr, neg_fdr)")
        fdr_pos_col, fdr_neg_col = fdr_column
        for col in (fdr_pos_col, fdr_neg_col):
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")
    else:
        if fdr_column not in df.columns:
            raise KeyError(f"Missing column: {fdr_column}")
        fdr_pos_col = fdr_column
        fdr_neg_col = None

    if isinstance(y_column, tuple):
        if len(y_column) != 2:
            raise ValueError("y_column tuple must be size 2")
        y_pos_col, y_neg_col = fdr_column
        for col in (y_pos_col, y_neg_col):
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")
    else:
        if y_column not in df.columns:
            raise KeyError(f"Missing column: {y_column}")
        y_pos_col = y_column
        y_neg_col = None
    cols = [log_fc_column, fdr_pos_col, y_pos_col]
    if fdr_neg_col is not None:
        cols.append(fdr_neg_col)
    if y_neg_col is not None:
        cols.append(y_neg_col)
    if name_column:
        cols.append(name_column)

    data = df[list(set(cols))].copy()

    # Ensure numeric
    data[log_fc_column] = pd.to_numeric(data[log_fc_column], errors="coerce")
    for col in [fdr_pos_col, fdr_neg_col, y_pos_col, y_neg_col]:
        if col is not None:
            print(col)
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[log_fc_column])
    x = data[log_fc_column].to_numpy()

    # Select correct FDR depending on direction
    if fdr_neg_col is not None:
        fdr_raw = np.where(
            x >= 0,
            data[fdr_pos_col].to_numpy(),
            data[fdr_neg_col].to_numpy(),
        )
        y_raw = np.where(
            x >= 0,
            data[y_pos_col].to_numpy(),
            data[y_neg_col].to_numpy(),
        )
    else:
        fdr_raw = data[fdr_pos_col].to_numpy()
        y_raw = data[y_pos_col].to_numpy()

    y_raw = np.clip(fdr_raw, y_clip_min, 1.0)

    if transform_y:
        y = -np.log10(y_raw)
        fdr_thr_plot = -np.log10(max(fdr_threshold, y_clip_min))
        y_label = ylabel or r"$-\log_{10}(\mathrm{FDR})$"
    else:
        y = y_raw
        fdr_thr_plot = fdr_threshold
        y_label = ylabel or "FDR"

    # Significance logic (RAW FDR scale)
    sig = (fdr_raw <= fdr_threshold) & (np.abs(x) >= log_threshold)
    pos = sig & (x >= log_threshold)
    neg = sig & (x <= -log_threshold)
    nonsig = ~sig

    # Determine which points exceed y_clip_max (if specified)
    if y_clip_max is not None:
        clipped = y > y_clip_max
        y_plot = np.where(clipped, y_clip_max, y)
    else:
        clipped = np.zeros(len(y), dtype=bool)
        y_plot = y

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Non-significant first (circles for normal, triangles for clipped)
    nonsig_normal = nonsig & ~clipped
    nonsig_clipped = nonsig & clipped

    ax.scatter(
        x[nonsig_normal],
        y_plot[nonsig_normal],
        s=point_size,
        alpha=alpha,
        c="grey",
        edgecolors="none",
        marker="o",
    )
    if nonsig_clipped.any():
        ax.scatter(
            x[nonsig_clipped],
            y_plot[nonsig_clipped],
            s=point_size * 1.5,
            alpha=alpha,
            c="grey",
            edgecolors="black",
            linewidths=0.5,
            marker="^",
        )

    # Significant negative (circles for normal, triangles for clipped)
    neg_normal = neg & ~clipped
    neg_clipped = neg & clipped

    ax.scatter(
        x[neg_normal],
        y_plot[neg_normal],
        s=point_size,
        alpha=alpha,
        c="blue",
        edgecolors="none",
        marker="o",
    )
    if neg_clipped.any():
        ax.scatter(
            x[neg_clipped],
            y_plot[neg_clipped],
            s=point_size * 1.5,
            alpha=alpha,
            c="blue",
            edgecolors="black",
            linewidths=0.5,
            marker="^",
        )

    # Significant positive (circles for normal, triangles for clipped)
    pos_normal = pos & ~clipped
    pos_clipped = pos & clipped

    ax.scatter(
        x[pos_normal],
        y_plot[pos_normal],
        s=point_size,
        alpha=alpha,
        c="red",
        edgecolors="none",
        marker="o",
    )
    if pos_clipped.any():
        ax.scatter(
            x[pos_clipped],
            y_plot[pos_clipped],
            s=point_size * 1.5,
            alpha=alpha,
            c="red",
            edgecolors="black",
            linewidths=0.5,
            marker="^",
        )

    # Threshold lines
    ax.axvline(+log_threshold, linestyle="--", linewidth=1, color="grey")
    ax.axvline(-log_threshold, linestyle="--", linewidth=1, color="grey")
    ax.axhline(fdr_thr_plot, linestyle="--", linewidth=1, color="grey")

    ax.set_xlabel(xlabel or log_fc_column)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Optional labeling (top by smallest relevant FDR)
    if top_n_labels > 0:
        if not name_column:
            raise ValueError("top_n_labels requires name_column")

        label_df = data.loc[sig].copy()
        label_df["_fdr_used"] = fdr_raw[sig]
        label_df = pd.concat(
            [
                label_df.sort_values("_fdr_used").head(top_n_labels),
                label_df.sort_values("_fdr_used").tail(top_n_labels),
            ]
        )

        for _, row in label_df.iterrows():
            xi = row[log_fc_column]
            fdr_i = max(min(row["_fdr_used"], 1.0), y_clip_min)
            yi = -np.log10(fdr_i) if transform_y else fdr_i
            # Clip label position if needed
            if y_clip_max is not None and yi > y_clip_max:
                yi = y_clip_max
            ax.text(xi, yi, str(row[name_column]), fontsize=label_fontsize)

    ax.margins(x=0.05, y=0.05)

    # Set y-axis limit if clipping
    if y_clip_max is not None:
        ax.set_ylim(bottom=ax.get_ylim()[0], top=y_clip_max * 1.02)
    return fig, ax


def plot_control_distribution_per_condition(
    qc_results: Dict,
    figsize: Tuple[float, float] = (12, 8),
    bins: int = 50,
    alpha: float = 0.7,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot histograms/density of Δc for controls in each condition vs baseline.

    Creates faceted plots showing distribution with median, IQR, and tail-rate
    annotations.

    Parameters
    ----------
    qc_results : dict
        Output from control_sgrna_qc().
    figsize : tuple
        Figure size.
    bins : int
        Number of histogram bins.
    alpha : float
        Histogram transparency.

    Returns
    -------
    tuple
        (figure, axes array)
    """
    delta_df = qc_results["delta"]
    conditions = qc_results["conditions"]
    baseline_condition = qc_results["baseline_condition"]
    metrics = qc_results["metrics"]

    # Exclude baseline
    plot_conditions = [c for c in conditions.keys() if c != baseline_condition]

    n_conditions = len(plot_conditions)
    if n_conditions == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No non-baseline conditions found",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        return fig, np.array([ax])

    ncols = min(3, n_conditions)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n_conditions / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, cond in enumerate(plot_conditions):
        ax = axes[idx]
        cols = conditions[cond]

        # Flatten all replicates
        delta_values = delta_df[cols].values.flatten()
        delta_values = delta_values[~np.isnan(delta_values)]

        # Histogram
        ax.hist(
            delta_values,
            bins=bins,
            alpha=alpha,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        # Add KDE
        try:
            from scipy.stats import gaussian_kde

            # Require at least some variability to compute KDE
            if len(delta_values) < 2 or np.all(delta_values == delta_values[0]):
                raise TypeError("Insufficient variability to compute KDE")

            kde = gaussian_kde(delta_values)
            x_range = np.linspace(delta_values.min(), delta_values.max(), 200)
            kde_values = kde(x_range)
            # Scale KDE to match histogram counts (for overlay)
            kde_scaled = (
                kde_values
                * len(delta_values)
                * (delta_values.max() - delta_values.min())
                / bins
            )

            # Plot density on secondary axis
            ax2 = ax.twinx()
            ax2.plot(
                x_range,
                kde_values,
                color="r",
                linewidth=2,
                alpha=0.7,
                label="Density",
            )
            ax2.set_ylabel("Density", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

            # Also overlay scaled KDE on the primary axis
            ax.plot(
                x_range,
                kde_scaled,
                color="darkred",
                linewidth=1.5,
                alpha=0.6,
                linestyle="--",
                label="KDE (scaled)",
            )
        except (TypeError, np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(
                f"Could not compute KDE for condition '{cond}': {e}",
                UserWarning,
            )

        # Add vertical line at median
        med = metrics[cond]["median"]
        ax.axvline(
            med,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Median: {med:.3f}",
        )
        ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # Annotations
        textstr = (
            f"Median: {metrics[cond]['median']:.3f}\n"
            f"IQR: {metrics[cond]['iqr']:.3f}\n"
            f"Tail |Δc|>1: {metrics[cond]['tail_rate_1.0']*100:.1f}%\n"
            f"Tail |Δc|>0.5: {metrics[cond]['tail_rate_0.5']*100:.1f}%\n"
            f"n={metrics[cond]['n_values']}"
        )

        ax.text(
            0.98,
            0.97,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )

        ax.set_xlabel(f"Δc = log2(CPM/{baseline_condition})", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(
            f"{cond} vs {baseline_condition}", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_conditions, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Control sgRNA Distribution per Condition",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    return fig, axes


def plot_pairwise_control_shifts(
    qc_results: Dict,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    center: float = 0.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annot: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot heatmap of pairwise median control log2FC between all conditions.

    Parameters
    ----------
    qc_results : dict
        Output from control_sgrna_qc().
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name (diverging recommended).
    center : float
        Value to center colormap at.
    vmin, vmax : float
        Colormap limits.
    annot : bool
        Annotate cells with values.

    Returns
    -------
    tuple
        (figure, axes)
    """
    pairwise_median = qc_results["pairwise_median"]

    fig, ax = plt.subplots(figsize=figsize)

    # Handle empty or all-NaN pairwise_median
    if pairwise_median.size == 0 or np.all(np.isnan(pairwise_median.values)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No pairwise shifts available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        return fig, ax

    # Determine vmin/vmax if not provided
    if vmin is None or vmax is None:
        abs_vals = np.abs(
            pairwise_median.values[~np.isnan(pairwise_median.values)]
        )
        if abs_vals.size == 0:
            abs_max = 0.0
        else:
            abs_max = abs_vals.max()
        if vmin is None:
            vmin = -abs_max
        if vmax is None:
            vmax = abs_max

    sns.heatmap(
        pairwise_median,
        annot=annot,
        fmt=".3f",
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Median log2(FC)"},
        ax=ax,
    )

    ax.set_title(
        "Pairwise Median Control Shifts\n(Should be ~0 everywhere)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_ylabel("Condition", fontsize=11)

    fig.tight_layout()

    return fig, ax


def plot_control_replicate_correlation(
    qc_results: Dict,
    figsize: Tuple[float, float] = (14, 10),
    cmap: str = "RdYlGn",
    annot: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot correlation matrices between replicates for each condition.

    Shows Pearson correlation of log2(CPM+1) values for control sgRNAs.

    Parameters
    ----------
    qc_results : dict
        Output from control_sgrna_qc().
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name.
    annot : bool
        Annotate cells with correlation values.

    Returns
    -------
    tuple
        (figure, axes array)
    """
    replicate_correlations = qc_results["replicate_correlations"]
    if not replicate_correlations:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No replicate correlations available (no condition with >1 replicate)",  # noqa: E501
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        return fig, np.array([ax])

    conditions = list(replicate_correlations.keys())

    n_conditions = len(conditions)
    ncols = min(3, n_conditions)
    nrows = int(np.ceil(n_conditions / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        corr_matrix = replicate_correlations[cond]

        sns.heatmap(
            corr_matrix,
            annot=annot,
            fmt=".3f",
            cmap=cmap,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Pearson r"},
            ax=ax,
        )

        ax.set_title(
            f"{cond} - Replicate Correlation", fontsize=11, fontweight="bold"
        )

    # Hide unused subplots
    for idx in range(n_conditions, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Control sgRNA Replicate Consistency\n(High correlation expected)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    return fig, axes


def plot_ma(
    sample_col: str,
    ma_df: DataFrame,
    ax: Optional[plt.Axes] = None,
    compute_trend: bool = True,
    title: Optional[str] = None,
    alpha: float = 0.3,
    s: float = 1,
) -> Tuple[plt.Axes, Dict]:
    """
    Plot MA (M vs A) scatter plot for one sample vs baseline.

    M = logCPM_sample - logCPM_baseline (fold-change)
    A = 0.5 * (logCPM_sample + logCPM_baseline) (average expression)

    Parameters
    ----------
    sample_col : str
        Sample column name.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    compute_trend : bool
        If True, compute and plot trend line (M vs A correlation).
    title : str, optional
        Plot title.
    alpha : float
        Point transparency.
    s : float
        Point size.

    Returns
    -------
    tuple
        (ax, metrics) where metrics contains:
        - ma_correlation: Pearson correlation between M and A
        - trend_slope: Slope from robust regression (if computed)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Compute M and A
    # baseline_mean = logcpm_df[baseline_cols].mean(axis=1)
    # M = logcpm_df[sample_col] - baseline_mean
    # A = 0.5 * (logcpm_df[sample_col] + baseline_mean)

    # Remove NaN
    # valid_mask = ~(M.isna() | A.isna())
    # M_clean = M[valid_mask]
    # A_clean = A[valid_mask]

    M_clean = ma_df[sample_col + "-M"].values
    A_clean = ma_df[sample_col + "-A"].values
    # Scatter plot
    ax.scatter(A_clean, M_clean, alpha=alpha, s=s, c="gray", rasterized=True)

    # Compute metrics
    metrics = {}

    if len(M_clean) > 10:
        ma_corr = np.corrcoef(A_clean, M_clean)[0, 1]
        metrics["ma_correlation"] = ma_corr

        if compute_trend and not np.isnan(ma_corr):
            # Fit robust regression (simple linear fit)
            from scipy import stats as sp_stats

            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
                A_clean, M_clean
            )
            metrics["trend_slope"] = slope
            metrics["trend_intercept"] = intercept

            # Plot trend line
            A_range = np.array([A_clean.min(), A_clean.max()])
            M_trend = slope * A_range + intercept
            ax.plot(
                A_range,
                M_trend,
                "r--",
                linewidth=2,
                label=f"Trend (slope={slope:.3f})",
            )
            ax.legend(loc="best", fontsize=8)

            # Annotate correlation
            ax.text(
                0.02,
                0.98,
                f"Corr(M, A) = {ma_corr:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
    else:
        metrics["ma_correlation"] = np.nan

    # Horizontal line at M=0
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Labels
    ax.set_xlabel("A = 0.5 * (log2CPM_sample + log2CPM_baseline)", fontsize=10)
    ax.set_ylabel("M = log2CPM_sample - log2CPM_baseline", fontsize=10)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    ax.grid(True, alpha=0.3)

    return ax, metrics


def plot_ma_grid(
    ma_df: DataFrame,
    figsize: Optional[Tuple[float, float]] = None,
    compute_trend: bool = True,
) -> Tuple[plt.Figure, Dict]:
    """
    Plot grid of MA plots for all samples vs baseline.

    Parameters
    ----------
    logcpm_df : DataFrame
        Log2(CPM+1) values.
    conditions_dict : dict
        Mapping condition -> list of sample columns.
    baseline_cols : list
        Baseline columns.
    figsize : tuple, optional
        Figure size. If None, auto-calculated.
    compute_trend : bool
        If True, compute trend metrics.

    Returns
    -------
    tuple
        (fig, metrics_dict) where metrics_dict maps sample -> metrics.
    """
    # Collect all non-baseline samples
    all_samples = [col[:-2] for col in ma_df.columns if col.endswith("-M")]
    # for condition, samples in conditions_dict.items():
    #     for sample in samples:
    #         if sample not in baseline_cols:
    #             all_samples.append(sample)

    n_samples = len(all_samples)
    if n_samples == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No non-baseline samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig, {}

    # Layout
    ncols = min(3, n_samples)
    nrows = int(np.ceil(n_samples / ncols))

    if figsize is None:
        figsize = (ncols * 5, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each sample
    metrics_dict = {}

    for i, sample in enumerate(all_samples):
        ax = axes[i]
        ma_sample = ma_df[[sample + "-M", sample + "-A"]].dropna()
        ax_out, metrics = plot_ma(
            sample,
            ma_df=ma_sample,
            ax=ax,
            compute_trend=compute_trend,
            title=sample,
            alpha=0.2,
            s=0.5,
        )
        metrics_dict[sample] = metrics

    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "MA Plots (Sample vs Baseline)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    return fig, metrics_dict


def plot_library_pca(
    data_df: DataFrame,
    sample_cols: List[str],
    conditions_dict: Optional[Dict[str, List[str]]] = None,
    figsize: Tuple[float, float] = (12, 10),
    color_by: str = "condition",
    title: str = "PCA Analysis",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot PCA biplot with loadings and scree plot for library-wide data.

    Generalized version of plot_control_pca that works with any data.

    Parameters
    ----------
    data_df : DataFrame
        Data matrix (features x samples). Should be log-transformed.
    sample_cols : list
        Sample column names.
    conditions_dict : dict, optional
        Mapping condition -> sample columns (for coloring).
        If None, will be inferred from sample names or all same color.
    figsize : tuple
        Figure size.
    color_by : str
        Color by "condition" or "replicate".
    title : str
        Plot title.

    Returns
    -------
    tuple
        (fig, (ax_biplot, ax_scree))
    """
    # Prepare data
    pca_data = data_df[sample_cols].T  # Transpose: samples x features

    # Remove columns with zero variance
    col_std = pca_data.std()
    valid_cols = col_std[col_std > 0].index
    pca_data = pca_data[valid_cols]

    if pca_data.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "Insufficient variance for PCA\n(need at least 2 features with variance)",  # noqa: E501
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        return fig, (ax, None)

    # Standardize
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)

    # PCA
    n_components = min(10, pca_data_scaled.shape[0], pca_data_scaled.shape[1])
    try:
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(pca_data_scaled)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5, 0.5, f"PCA failed: {e}", ha="center", va="center", fontsize=12
        )
        ax.axis("off")
        return fig, (ax, None)

    # Extract PC1 and PC2
    pc1 = (
        pca_transformed[:, 0]
        if pca_transformed.shape[1] > 0
        else np.zeros(len(sample_cols))
    )
    pc2 = (
        pca_transformed[:, 1]
        if pca_transformed.shape[1] > 1
        else np.zeros(len(sample_cols))
    )

    # Assign colors
    if conditions_dict is not None:
        # Create mapping
        sample_to_condition = {}
        sample_to_replicate = {}
        for condition, samples in conditions_dict.items():
            for sample in samples:
                sample_to_condition[sample] = condition
                # Try to parse replicate
                parts = sample.rsplit("_", 1)
                if len(parts) == 2:
                    sample_to_replicate[sample] = parts[1]
                else:
                    sample_to_replicate[sample] = "Rep1"

        if color_by == "condition":
            labels = [
                sample_to_condition.get(s, "unknown") for s in sample_cols
            ]
        else:
            labels = [
                sample_to_replicate.get(s, "unknown") for s in sample_cols
            ]
    else:
        labels = ["sample"] * len(sample_cols)

    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    # Create figure
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

    ax_biplot = fig.add_subplot(gs[0, 0])
    ax_scree = fig.add_subplot(gs[0, 1])

    # Biplot
    for label in unique_labels:
        mask = np.array(labels) == label
        ax_biplot.scatter(
            pc1[mask],
            pc2[mask],
            c=[label_to_color[label]],
            label=label,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    # Annotate points
    for i, sample in enumerate(sample_cols):
        ax_biplot.annotate(
            sample,
            (pc1[i], pc2[i]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = (
        pca.explained_variance_ratio_[1] * 100
        if len(pca.explained_variance_ratio_) > 1
        else 0
    )

    ax_biplot.set_xlabel(f"PC1 ({var_pc1:.1f}% variance)", fontsize=11)
    ax_biplot.set_ylabel(f"PC2 ({var_pc2:.1f}% variance)", fontsize=11)
    ax_biplot.set_title(f"{title} - PCA Biplot", fontsize=12, fontweight="bold")
    ax_biplot.legend(loc="best", fontsize=9)
    ax_biplot.grid(True, alpha=0.3)
    ax_biplot.axhline(
        0, color="black", linestyle="--", linewidth=0.8, alpha=0.5
    )
    ax_biplot.axvline(
        0, color="black", linestyle="--", linewidth=0.8, alpha=0.5
    )

    # Scree plot
    ax_scree.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_ * 100,
        alpha=0.7,
        color="steelblue",
    )
    ax_scree.set_xlabel("Principal Component", fontsize=10)
    ax_scree.set_ylabel("% Variance Explained", fontsize=10)
    ax_scree.set_title("Scree Plot", fontsize=11, fontweight="bold")
    ax_scree.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    return fig, (ax_biplot, ax_scree)


def plot_sample_correlations(
    logcpm_df: DataFrame,
    sample_cols: List[str],
    conditions_dict: Optional[Dict[str, List[str]]] = None,
    method: str = "spearman",
    figsize: Tuple[float, float] = (10, 9),
    cmap: str = "RdYlBu_r",
) -> plt.Figure:
    """
    Plot sample-vs-sample correlation heatmap for library-wide analysis.

    Parameters
    ----------
    logcpm_df : DataFrame
        Log2(CPM+1) values.
    sample_cols : list
        Sample column names.
    conditions_dict : dict, optional
        Mapping condition -> samples (for ordering/annotation).
    method : str
        Correlation method: 'spearman' or 'pearson'.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name.

    Returns
    -------
    plt.Figure
    """
    # Compute correlation
    if method == "spearman":
        corr_matrix = logcpm_df[sample_cols].corr(method="spearman")
    else:
        corr_matrix = logcpm_df[sample_cols].corr(method="pearson")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap=cmap,
        center=0.7,
        vmin=0,
        vmax=1,
        square=True,
        annot=True if len(sample_cols) <= 10 else False,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": f"{method.capitalize()} Correlation"},
    )

    ax.set_title(
        f"Sample Correlations ({method.capitalize()})",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()

    return fig


def plot_control_pca(
    qc_results: Dict,
    figsize: Tuple[float, float] = (12, 10),
    use_log: bool = True,
    color_by: str = "condition",  # "condition" or "replicate"
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    PCA of control sgRNAs across all samples.

    Shows whether controls cluster by condition (bad - shows treatment effect)
    or by replicate (acceptable).

    Parameters
    ----------
    qc_results : dict
        Output from control_sgrna_qc().
    figsize : tuple
        Figure size.
    use_log : bool
        Use log2(CPM+1) for PCA.
    color_by : str
        Color samples by "condition" or "replicate".

    Returns
    -------
    tuple
        (figure, (scatter_ax, variance_ax))
    """
    cpm_df = qc_results["cpm"]
    sample_cols = qc_results["sample_cols"]

    # Prepare data matrix: samples as rows, sgRNAs as columns
    if use_log:
        data_matrix = np.log2(cpm_df[sample_cols].T + 1)
    else:
        data_matrix = cpm_df[sample_cols].T

    # Remove any columns with NaN or zero variance
    valid_cols = data_matrix.std(axis=0) > 0
    data_matrix = data_matrix.loc[:, valid_cols]

    # PCA
    pca = PCA(n_components=min(len(sample_cols), 10))
    try:
        pca_coords = pca.fit_transform(data_matrix)
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        warnings.warn(f"PCA failed: {e}", UserWarning)
        ax.text(0.5, 0.5, f"PCA failed: {str(e)}", ha="center", va="center")
        ax.axis("off")
        return fig, (ax, ax)

    # Ensure at least two components for plotting
    if pca_coords.shape[1] == 1:
        pca_coords = np.hstack([pca_coords, np.zeros((pca_coords.shape[0], 1))])

    # Parse sample metadata
    sample_metadata = []
    for col in sample_cols:
        from crisprscreens.core.qc import parse_condition_replicate

        cond, rep = parse_condition_replicate(col)
        sample_metadata.append(
            {"sample": col, "condition": cond, "replicate": rep}
        )
    sample_meta_df = pd.DataFrame(sample_metadata)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_variance = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[0, 1])

    # Color mapping
    if color_by == "condition":
        unique_vals = sample_meta_df["condition"].unique()
        color_col = "condition"
    else:
        unique_vals = sample_meta_df["replicate"].unique()
        color_col = "replicate"

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
    color_map = dict(zip(unique_vals, colors))

    # Scatter plot
    for val in unique_vals:
        mask = sample_meta_df[color_col] == val
        ax_scatter.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=[color_map[val]],
            label=val,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

    # Add sample labels
    for idx, row in sample_meta_df.iterrows():
        ax_scatter.annotate(
            row["sample"],
            (pca_coords[idx, 0], pca_coords[idx, 1]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    pc1_var = (
        pca.explained_variance_ratio_[0]
        if len(pca.explained_variance_ratio_) > 0
        else 0.0
    )
    pc2_var = (
        pca.explained_variance_ratio_[1]
        if len(pca.explained_variance_ratio_) > 1
        else 0.0
    )

    ax_scatter.set_xlabel(f"PC1 ({pc1_var*100:.1f}%)", fontsize=11)
    ax_scatter.set_ylabel(f"PC2 ({pc2_var*100:.1f}%)", fontsize=11)
    ax_scatter.set_title(
        f"PCA of Control sgRNAs (colored by {color_by})",
        fontsize=13,
        fontweight="bold",
    )
    ax_scatter.grid(True, alpha=0.3)

    # Legend
    ax_legend.axis("off")
    ax_legend.legend(
        *ax_scatter.get_legend_handles_labels(),
        loc="center",
        frameon=True,
        fontsize=10,
    )

    # Variance explained
    n_components_plot = min(10, len(pca.explained_variance_ratio_))
    ax_variance.bar(
        range(1, n_components_plot + 1),
        pca.explained_variance_ratio_[:n_components_plot] * 100,
        color="steelblue",
        alpha=0.7,
    )
    ax_variance.set_xlabel("Principal Component", fontsize=10)
    ax_variance.set_ylabel("Variance Explained (%)", fontsize=10)
    ax_variance.set_title("Scree Plot", fontsize=11, fontweight="bold")
    ax_variance.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Control sgRNA PCA - Should cluster by replicate, not condition",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    return fig, (ax_scatter, ax_variance)


def plot_effect_size_vs_reproducibility(
    gene_summary_df,
    effect_col="beta",
    fdr_col="fdr",
    sgrna_summary_df=None,
    gene_col="Gene",
    fdr_threshold=0.25,
    figsize=(10, 8),
    title="Effect Size vs. Reproducibility",
    ax=None,
):
    """
    Plot effect size vs. sgRNA consistency for each gene.

    Addresses the question: "Is this a real hit?" by showing whether
    significant genes also have consistent effects across sgRNAs.

    Parameters
    ----------
    gene_summary_df : pd.DataFrame
        MAGeCK gene summary with columns: Gene, beta, fdr
    effect_col : str
        Column name for effect size (beta)
    fdr_col : str
        Column name for FDR
    sgrna_summary_df : pd.DataFrame, optional
        MAGeCK sgRNA summary with columns: Gene, LFC
        If provided, computes sgRNA consistency per gene
    gene_col : str
        Column name for gene identifiers
    fdr_threshold : float
        FDR threshold for coloring (default 0.25)
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: spearman_cor, n_consistent_hits, n_inconsistent_hits
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute sgRNA consistency if data available
    if sgrna_summary_df is not None:
        # Calculate coefficient of variation of LFC per gene
        consistency = (
            sgrna_summary_df.groupby(gene_col)
            .agg(
                {
                    "LFC": lambda x: 1
                    - (x.std() / (abs(x.mean()) + 1e-6) if len(x) > 1 else 0)
                }
            )
            .rename(columns={"LFC": "consistency"})
        )

        plot_df = gene_summary_df.merge(
            consistency, left_on=gene_col, right_index=True, how="left"
        )
        plot_df["consistency"] = plot_df["consistency"].fillna(0.5)

        # Count sgRNAs per gene
        sgrna_counts = (
            sgrna_summary_df.groupby(gene_col).size().rename("n_sgrnas")
        )
        plot_df = plot_df.merge(
            sgrna_counts, left_on=gene_col, right_index=True, how="left"
        )
    else:
        # Fallback: use -log10(FDR) as proxy for reproducibility
        plot_df = gene_summary_df.copy()
        plot_df["consistency"] = np.clip(
            -np.log10(plot_df[fdr_col] + 1e-10) / 10, 0, 1
        )
        plot_df["n_sgrnas"] = 4  # default size

    # Create scatter plot
    significant = plot_df[fdr_col] < fdr_threshold

    # Non-significant genes
    ax.scatter(
        plot_df.loc[~significant, effect_col],
        plot_df.loc[~significant, "consistency"],
        s=plot_df.loc[~significant, "n_sgrnas"] * 10,
        c="lightgray",
        alpha=0.3,
        label=f"FDR ≥ {fdr_threshold}",
    )

    # Significant genes colored by FDR
    scatter = ax.scatter(
        plot_df.loc[significant, effect_col],
        plot_df.loc[significant, "consistency"],
        s=plot_df.loc[significant, "n_sgrnas"] * 10,
        c=-np.log10(plot_df.loc[significant, fdr_col] + 1e-10),
        cmap="coolwarm",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
        label=f"FDR < {fdr_threshold}",
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("-log10(FDR)", fontsize=10)

    # Add quadrant labels
    x_mid = plot_df[effect_col].median()
    y_mid = 0.7

    ax.axhline(y_mid, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x_mid, color="gray", linestyle="--", alpha=0.3)

    ax.text(
        0.95,
        0.95,
        "High Effect\nHigh Consistency",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )
    ax.text(
        0.05,
        0.05,
        "Low Effect\nLow Consistency",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )

    ax.set_xlabel(f"{effect_col} (Effect Size)", fontsize=11)
    ax.set_ylabel("sgRNA Consistency", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    # Calculate metrics
    from scipy.stats import spearmanr

    valid = ~plot_df[effect_col].isna() & ~plot_df["consistency"].isna()
    spearman_cor = (
        spearmanr(
            plot_df.loc[valid, effect_col].abs(),
            plot_df.loc[valid, "consistency"],
        )[0]
        if valid.sum() > 2
        else np.nan
    )

    n_consistent_hits = (
        (plot_df[fdr_col] < fdr_threshold) & (plot_df["consistency"] > y_mid)
    ).sum()
    n_inconsistent_hits = (
        (plot_df[fdr_col] < fdr_threshold) & (plot_df["consistency"] <= y_mid)
    ).sum()

    metrics = {
        "spearman_cor": spearman_cor,
        "n_consistent_hits": n_consistent_hits,
        "n_inconsistent_hits": n_inconsistent_hits,
    }

    return fig, ax, metrics


def plot_rank_stability(
    rra_summary_df,
    mle_summary_df,
    gene_col="Gene",
    rra_rank_col="pos|rank",
    mle_rank_col="Gene|rank",
    fdr_col_rra="pos|fdr",
    fdr_col_mle="Gene|fdr",
    fdr_threshold=0.25,
    top_n=50,
    figsize=(10, 8),
    title="Rank Stability: RRA vs. MLE",
    ax=None,
):
    """
    Compare gene rankings between RRA and MLE methods.

    Addresses: "Are results method-robust?" Shows whether top hits
    are consistent across analysis methods.

    Parameters
    ----------
    rra_summary_df : pd.DataFrame
        MAGeCK RRA gene summary
    mle_summary_df : pd.DataFrame
        MAGeCK MLE gene summary
    gene_col : str
        Column name for gene identifiers
    rra_rank_col : str
        Column with RRA ranks
    mle_rank_col : str
        Column with MLE ranks
    fdr_col_rra : str
        RRA FDR column
    fdr_col_mle : str
        MLE FDR column
    fdr_threshold : float
        FDR threshold for highlighting
    top_n : int
        Number of top genes to highlight
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: spearman_cor, top_n_jaccard, agreement_rate
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Merge RRA and MLE results
    merged = rra_summary_df[[gene_col, rra_rank_col, fdr_col_rra]].merge(
        mle_summary_df[[gene_col, mle_rank_col, fdr_col_mle]],
        on=gene_col,
        how="inner",
    )

    # Identify significant genes
    sig_both = (merged[fdr_col_rra] < fdr_threshold) & (
        merged[fdr_col_mle] < fdr_threshold
    )
    sig_rra_only = (merged[fdr_col_rra] < fdr_threshold) & (
        merged[fdr_col_mle] >= fdr_threshold
    )
    sig_mle_only = (merged[fdr_col_rra] >= fdr_threshold) & (
        merged[fdr_col_mle] < fdr_threshold
    )
    sig_neither = (merged[fdr_col_rra] >= fdr_threshold) & (
        merged[fdr_col_mle] >= fdr_threshold
    )

    # Plot different categories
    ax.scatter(
        merged.loc[sig_neither, rra_rank_col],
        merged.loc[sig_neither, mle_rank_col],
        c="lightgray",
        alpha=0.3,
        s=20,
        label="Non-significant",
    )

    ax.scatter(
        merged.loc[sig_rra_only, rra_rank_col],
        merged.loc[sig_rra_only, mle_rank_col],
        c="blue",
        alpha=0.5,
        s=40,
        label="RRA only",
        marker="^",
    )

    ax.scatter(
        merged.loc[sig_mle_only, rra_rank_col],
        merged.loc[sig_mle_only, mle_rank_col],
        c="orange",
        alpha=0.5,
        s=40,
        label="MLE only",
        marker="s",
    )

    ax.scatter(
        merged.loc[sig_both, rra_rank_col],
        merged.loc[sig_both, mle_rank_col],
        c="red",
        alpha=0.7,
        s=60,
        label="Both methods",
        edgecolors="black",
        linewidths=0.5,
    )

    # Add diagonal line
    max_rank = max(merged[rra_rank_col].max(), merged[mle_rank_col].max())
    ax.plot(
        [0, max_rank],
        [0, max_rank],
        "k--",
        alpha=0.3,
        label="Perfect agreement",
    )

    # Highlight top N region
    ax.axhline(top_n, color="green", linestyle=":", alpha=0.4)
    ax.axvline(top_n, color="green", linestyle=":", alpha=0.4)
    ax.fill_between([0, top_n], 0, top_n, alpha=0.1, color="green")

    ax.set_xlabel("RRA Rank", fontsize=11)
    ax.set_ylabel("MLE Rank", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, max_rank * 1.05)
    ax.set_ylim(0, max_rank * 1.05)

    # Calculate metrics
    from scipy.stats import spearmanr

    spearman_cor = spearmanr(merged[rra_rank_col], merged[mle_rank_col])[0]

    # Jaccard index for top N genes
    top_rra = set(merged.nsmallest(top_n, rra_rank_col)[gene_col])
    top_mle = set(merged.nsmallest(top_n, mle_rank_col)[gene_col])
    jaccard = len(top_rra & top_mle) / len(top_rra | top_mle)

    # Agreement rate (both significant)
    agreement_rate = sig_both.sum() / (
        sig_both.sum() + sig_rra_only.sum() + sig_mle_only.sum()
    )

    # Add text box with metrics
    textstr = f"Spearman ρ: {spearman_cor:.3f}\nTop-{top_n} Jaccard: {jaccard:.3f}\nAgreement: {agreement_rate:.1%}"  # noqa: E501
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    metrics = {
        "spearman_cor": spearman_cor,
        "top_n_jaccard": jaccard,
        "agreement_rate": agreement_rate,
    }

    return fig, ax, metrics


def plot_direction_consistency(
    sgrna_summary_df,
    gene_col="Gene",
    lfc_col="LFC",
    genes=None,
    min_sgrnas=3,
    figsize=(12, 8),
    title="sgRNA Direction Consistency",
    ax=None,
):
    """
    Plot per-gene sgRNA effects to assess directional consistency.

    Addresses: "Are sgRNAs consistent?" Shows whether all sgRNAs for
    a gene point in the same direction.

    Parameters
    ----------
    sgrna_summary_df : pd.DataFrame
        MAGeCK sgRNA summary with columns: Gene, sgRNA, LFC
    gene_col : str
        Column name for gene identifiers
    lfc_col : str
        Column name for log fold change
    genes : list, optional
        List of specific genes to plot. If None, plots top genes by effect
    min_sgrnas : int
        Minimum number of sgRNAs per gene to include
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: consistency_scores (dict gene -> score)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Filter genes with sufficient sgRNAs
    sgrna_counts = sgrna_summary_df.groupby(gene_col).size()
    valid_genes = sgrna_counts[sgrna_counts >= min_sgrnas].index

    plot_df = sgrna_summary_df[
        sgrna_summary_df[gene_col].isin(valid_genes)
    ].copy()

    # Select genes to plot
    if genes is None:
        # Select genes with largest mean absolute effect
        gene_effects = plot_df.groupby(gene_col)[lfc_col].apply(
            lambda x: abs(x.mean())
        )
        genes = gene_effects.nlargest(20).index.tolist()

    plot_df = plot_df[plot_df[gene_col].isin(genes)]

    # Calculate consistency score per gene
    consistency_scores = {}
    for gene in genes:
        gene_data = plot_df[plot_df[gene_col] == gene][lfc_col]
        if len(gene_data) > 0:
            # Fraction of sgRNAs with same sign as majority
            pos = (gene_data > 0).sum()
            neg = (gene_data < 0).sum()
            consistency_scores[gene] = max(pos, neg) / len(gene_data)
        else:
            consistency_scores[gene] = 0

    # Sort genes by consistency
    sorted_genes = sorted(
        genes, key=lambda g: consistency_scores.get(g, 0), reverse=True
    )

    # Create stripplot with jitter
    positions = {gene: i for i, gene in enumerate(sorted_genes)}

    for gene in sorted_genes:
        gene_data = plot_df[plot_df[gene_col] == gene]
        y_pos = positions[gene]

        # Color by sign
        pos_data = gene_data[gene_data[lfc_col] >= 0]
        neg_data = gene_data[gene_data[lfc_col] < 0]

        if len(pos_data) > 0:
            ax.scatter(
                pos_data[lfc_col],
                [y_pos] * len(pos_data),
                color="red",
                alpha=0.6,
                s=80,
                edgecolors="black",
                linewidths=0.5,
            )
        if len(neg_data) > 0:
            ax.scatter(
                neg_data[lfc_col],
                [y_pos] * len(neg_data),
                color="blue",
                alpha=0.6,
                s=80,
                edgecolors="black",
                linewidths=0.5,
            )

        # Add mean marker
        mean_lfc = gene_data[lfc_col].mean()
        ax.scatter(
            mean_lfc,
            y_pos,
            color="gold",
            s=200,
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
        )

        # Add consistency percentage
        consistency_pct = consistency_scores[gene] * 100
        ax.text(
            ax.get_xlim()[1] * 0.95,
            y_pos,
            f"{consistency_pct:.0f}%",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_yticks(range(len(sorted_genes)))
    ax.set_yticklabels(sorted_genes, fontsize=9)
    ax.set_xlabel("sgRNA Log Fold Change", fontsize=11)
    ax.set_ylabel("Gene", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.6, label="Positive LFC"),
        Patch(facecolor="blue", alpha=0.6, label="Negative LFC"),
        plt.scatter(
            [],
            [],
            color="gold",
            s=200,
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            label="Mean",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    metrics = {"consistency_scores": consistency_scores}

    return fig, ax, metrics


def plot_replicate_effect_heatmap(
    data_matrix,
    row_labels=None,
    col_labels=None,
    cmap="RdBu_r",
    center=0,
    figsize=(10, 12),
    title="Replicate Effect Heatmap",
    ax=None,
):
    """
    Heatmap of gene effects across replicates/conditions.

    Addresses: "Is this experimentally stable?" Shows consistency
    across replicates and conditions.

    Parameters
    ----------
    data_matrix : pd.DataFrame or np.ndarray
        Matrix of effects (genes × replicates/conditions)
    row_labels : list, optional
        Gene names for rows
    col_labels : list, optional
        Replicate/condition names for columns
    cmap : str
        Colormap name
    center : float
        Value to center colormap on
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: mean_correlation, replicate_cv
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if isinstance(data_matrix, pd.DataFrame):
        data = data_matrix.values
        row_labels = row_labels or data_matrix.index.tolist()
        col_labels = col_labels or data_matrix.columns.tolist()
    else:
        data = data_matrix

    # Create heatmap
    sns.heatmap(
        data,
        cmap=cmap,
        center=center,
        cbar_kws={"label": "Effect Size"},
        xticklabels=col_labels if col_labels else True,
        yticklabels=row_labels if row_labels else True,
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray",
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.set_xlabel("Replicates / Conditions", fontsize=11)
    ax.set_ylabel("Genes", fontsize=11)

    # Calculate metrics
    # Mean pairwise correlation between replicates
    from scipy.stats import spearmanr

    if data.shape[1] > 1:
        correlations = []
        for i in range(data.shape[1]):
            for j in range(i + 1, data.shape[1]):
                valid = ~np.isnan(data[:, i]) & ~np.isnan(data[:, j])
                if valid.sum() > 2:
                    cor = spearmanr(data[valid, i], data[valid, j])[0]
                    correlations.append(cor)
        mean_correlation = np.mean(correlations) if correlations else np.nan
    else:
        mean_correlation = np.nan

    # Coefficient of variation across replicates
    replicate_cv = np.nanstd(data, axis=1) / (
        np.abs(np.nanmean(data, axis=1)) + 1e-6
    )
    median_cv = np.nanmedian(replicate_cv)

    metrics = {"mean_correlation": mean_correlation, "replicate_cv": median_cv}

    return fig, ax, metrics


def plot_effect_decomposition(
    mle_summary_df,
    effect_cols,
    gene_col="Gene",
    top_n=20,
    stacked=False,
    figsize=(12, 8),
    title="Effect Decomposition Across Conditions",
    ax=None,
):
    """
    Decompose MLE effects into individual condition contributions.

    For complex designs (e.g., Time + Treatment + Interaction), shows
    which factors drive the overall effect.

    Parameters
    ----------
    mle_summary_df : pd.DataFrame
        MAGeCK MLE gene summary with multiple beta columns
    effect_cols : list
        List of column names for different effects (e.g., ['beta_Time',
        'beta_Met', 'beta_High'])
    gene_col : str
        Column name for gene identifiers
    top_n : int
        Number of top genes to show
    stacked : bool
        Whether to use stacked bars
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: dominant_factor_counts
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Calculate total effect magnitude
    plot_df = mle_summary_df.copy()
    plot_df["total_effect"] = plot_df[effect_cols].abs().sum(axis=1)

    # Select top genes
    top_genes = plot_df.nlargest(top_n, "total_effect")

    # Prepare data for plotting
    plot_data = top_genes[[gene_col] + effect_cols].set_index(gene_col)

    # Sort by total effect
    plot_data = plot_data.loc[
        plot_data.abs().sum(axis=1).sort_values(ascending=True).index
    ]

    # Create bar plot
    if stacked:
        plot_data.plot(kind="barh", stacked=True, ax=ax, width=0.8)
    else:
        plot_data.plot(kind="barh", ax=ax, width=0.8)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Effect Size (Beta)", fontsize=11)
    ax.set_ylabel("Gene", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(title="Condition", fontsize=9, title_fontsize=10)
    ax.grid(axis="x", alpha=0.2)

    # Calculate which factor is dominant per gene
    dominant_factors = plot_data.abs().idxmax(axis=1).value_counts().to_dict()

    metrics = {"dominant_factor_counts": dominant_factors}

    return fig, ax, metrics


def plot_contrast(
    mle_summary_df,
    effect_col_x,
    effect_col_y,
    gene_col="Gene",
    fdr_col=None,
    fdr_threshold=0.25,
    figsize=(10, 8),
    title="Contrast Plot",
    ax=None,
):
    """
    Compare effects between two conditions (e.g., Low vs. High dose).

    Shows genes with condition-specific effects vs. shared effects.

    Parameters
    ----------
    mle_summary_df : pd.DataFrame
        MAGeCK MLE gene summary
    effect_col_x : str
        Column name for first condition effect
    effect_col_y : str
        Column name for second condition effect
    gene_col : str
        Column name for gene identifiers
    fdr_col : str, optional
        Column for FDR to color points
    fdr_threshold : float
        FDR threshold for highlighting
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: shared_effect_genes, specific_x_genes, specific_y_genes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    plot_df = mle_summary_df[[gene_col, effect_col_x, effect_col_y]].copy()

    if fdr_col and fdr_col in mle_summary_df.columns:
        plot_df["fdr"] = mle_summary_df[fdr_col]
        significant = plot_df["fdr"] < fdr_threshold

        ax.scatter(
            plot_df.loc[~significant, effect_col_x],
            plot_df.loc[~significant, effect_col_y],
            c="lightgray",
            alpha=0.3,
            s=20,
            label="Non-significant",
        )

        ax.scatter(
            plot_df.loc[significant, effect_col_x],
            plot_df.loc[significant, effect_col_y],
            c="red",
            alpha=0.6,
            s=60,
            edgecolors="black",
            linewidths=0.5,
            label=f"FDR < {fdr_threshold}",
        )
    else:
        ax.scatter(
            plot_df[effect_col_x],
            plot_df[effect_col_y],
            c="steelblue",
            alpha=0.5,
            s=40,
        )

    # Add diagonal and axes
    lims = [
        min(plot_df[effect_col_x].min(), plot_df[effect_col_y].min()),
        max(plot_df[effect_col_x].max(), plot_df[effect_col_y].max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, label="Equal effect")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.3)

    # Add quadrant labels
    ax.text(
        0.95,
        0.95,
        f"{effect_col_x} & {effect_col_y}\nBoth +",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )
    ax.text(
        0.05,
        0.05,
        f"{effect_col_x} & {effect_col_y}\nBoth -",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )
    ax.text(
        0.05,
        0.95,
        f"{effect_col_y} only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )
    ax.text(
        0.95,
        0.05,
        f"{effect_col_x} only",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        alpha=0.6,
    )

    ax.set_xlabel(f"{effect_col_x}", fontsize=11)
    ax.set_ylabel(f"{effect_col_y}", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Calculate metrics
    threshold = 0.5  # Effect size threshold
    shared = (plot_df[effect_col_x].abs() > threshold) & (
        plot_df[effect_col_y].abs() > threshold
    )
    specific_x = (plot_df[effect_col_x].abs() > threshold) & (
        plot_df[effect_col_y].abs() <= threshold
    )
    specific_y = (plot_df[effect_col_x].abs() <= threshold) & (
        plot_df[effect_col_y].abs() > threshold
    )

    metrics = {
        "shared_effect_genes": shared.sum(),
        "specific_x_genes": specific_x.sum(),
        "specific_y_genes": specific_y.sum(),
    }

    return fig, ax, metrics


def plot_pathway_enrichment_summary(
    enrichment_df,
    pathway_col="pathway",
    pvalue_col="pvalue",
    gene_ratio_col="gene_ratio",
    top_n=15,
    figsize=(10, 8),
    title="Pathway Enrichment Summary",
    ax=None,
):
    """
    Visualize pathway enrichment results as barplot or dotplot.

    Shows which functional categories are enriched among hits.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Pathway enrichment results with columns: pathway, pvalue, gene_ratio
    pathway_col : str
        Column name for pathway names
    pvalue_col : str
        Column name for p-values
    gene_ratio_col : str
        Column name for gene ratio (e.g., "10/150")
    top_n : int
        Number of top pathways to show
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Prepare data
    plot_df = enrichment_df.copy()
    plot_df["-log10(p)"] = -np.log10(plot_df[pvalue_col] + 1e-10)

    # Extract numeric gene ratio
    if isinstance(plot_df[gene_ratio_col].iloc[0], str):
        plot_df["ratio_numeric"] = plot_df[gene_ratio_col].apply(
            lambda x: eval(x.replace("/", "/")) if "/" in str(x) else float(x)
        )
    else:
        plot_df["ratio_numeric"] = plot_df[gene_ratio_col]

    # Select top pathways
    top_pathways = plot_df.nsmallest(top_n, pvalue_col)
    top_pathways = top_pathways.sort_values("-log10(p)")

    # Create dotplot
    scatter = ax.scatter(
        top_pathways["-log10(p)"],
        range(len(top_pathways)),
        s=top_pathways["ratio_numeric"] * 1000,
        c=top_pathways["-log10(p)"],
        cmap="Reds",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.set_yticks(range(len(top_pathways)))
    ax.set_yticklabels(top_pathways[pathway_col], fontsize=9)
    ax.set_xlabel("-log10(p-value)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Significance", fontsize=10)

    return fig, ax


def plot_gene_set_score_distribution(
    gene_summary_df,
    gene_sets,
    effect_col="beta",
    gene_col="Gene",
    figsize=(10, 6),
    title="Gene Set Score Distribution",
    ax=None,
):
    """
    Compare effect size distributions across predefined gene sets.

    Shows whether specific gene sets (e.g., kinases, transcription factors)
    are enriched for strong effects.

    Parameters
    ----------
    gene_summary_df : pd.DataFrame
        MAGeCK gene summary with effect sizes
    gene_sets : dict
        Dictionary mapping set names to lists of genes
    effect_col : str
        Column name for effect size
    gene_col : str
        Column name for gene identifiers
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: set_medians, p_values
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Prepare data
    data_by_set = {}
    for set_name, genes in gene_sets.items():
        set_data = gene_summary_df[gene_summary_df[gene_col].isin(genes)][
            effect_col
        ]
        data_by_set[set_name] = set_data.values

    # Add background distribution
    data_by_set["Background"] = gene_summary_df[effect_col].values

    # Create violin plots
    positions = range(len(data_by_set))
    parts = ax.violinplot(
        list(data_by_set.values()),
        positions=positions,
        showmeans=True,
        showmedians=True,
    )

    # Color by set
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_by_set)))
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(data_by_set.keys(), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"{effect_col} Distribution", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.2)

    # Calculate metrics
    set_medians = {name: np.median(data) for name, data in data_by_set.items()}

    # Statistical tests vs background
    from scipy.stats import mannwhitneyu

    p_values = {}
    background = data_by_set["Background"]
    for set_name in gene_sets.keys():
        if len(data_by_set[set_name]) > 0:
            _, pval = mannwhitneyu(
                data_by_set[set_name], background, alternative="two-sided"
            )
            p_values[set_name] = pval

    metrics = {"set_medians": set_medians, "p_values": p_values}

    return fig, ax, metrics


def plot_beta_vs_standard_error(
    mle_summary_df,
    beta_col,
    se_col,
    fdr_col=None,
    fdr_threshold=0.25,
    gene_col="Gene",
    figsize=(10, 8),
    title="Beta vs. Standard Error",
    ax=None,
):
    """
    Plot effect size vs. its standard error.

    Reveals poorly estimated large effects (high beta, high SE).

    Parameters
    ----------
    mle_summary_df : pd.DataFrame
        MAGeCK MLE gene summary
    beta_col : str
        Column name for beta values
    se_col : str
        Column name for standard errors
    fdr_col : str, optional
        Column for FDR coloring
    fdr_threshold : float
        FDR threshold for highlighting
    gene_col : str
        Column name for gene identifiers
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: high_uncertainty_genes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    plot_df = mle_summary_df[[gene_col, beta_col, se_col]].copy()

    if fdr_col and fdr_col in mle_summary_df.columns:
        plot_df["fdr"] = mle_summary_df[fdr_col]
        significant = plot_df["fdr"] < fdr_threshold

        ax.scatter(
            plot_df.loc[~significant, beta_col],
            plot_df.loc[~significant, se_col],
            c="lightgray",
            alpha=0.3,
            s=20,
            label="Non-significant",
        )

        ax.scatter(
            plot_df.loc[significant, beta_col],
            plot_df.loc[significant, se_col],
            c="red",
            alpha=0.6,
            s=60,
            edgecolors="black",
            linewidths=0.5,
            label=f"FDR < {fdr_threshold}",
        )
    else:
        ax.scatter(
            plot_df[beta_col], plot_df[se_col], c="steelblue", alpha=0.5, s=40
        )

    # Add warning zone (high beta, high SE)
    ax.axhspan(
        plot_df[se_col].quantile(0.75),
        plot_df[se_col].max(),
        alpha=0.1,
        color="orange",
        label="High uncertainty",
    )

    ax.set_xlabel(f"{beta_col}", fontsize=11)
    ax.set_ylabel("Standard Error", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)

    # Identify high-uncertainty genes
    se_threshold = plot_df[se_col].quantile(0.75)
    high_uncertainty = (plot_df[se_col] > se_threshold) & (
        plot_df[beta_col].abs() > 1
    )

    metrics = {"high_uncertainty_genes": high_uncertainty.sum()}

    return fig, ax, metrics


def plot_wald_z_distribution(
    mle_summary_df,
    z_col,
    figsize=(10, 6),
    title="Wald Z-statistic Distribution",
    ax=None,
):
    """
    Plot distribution of Wald Z-statistics with N(0,1) overlay.

    Detects model overdispersion or underdispersion.

    Parameters
    ----------
    mle_summary_df : pd.DataFrame
        MAGeCK MLE gene summary
    z_col : str
        Column name for Wald Z-statistics
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: kurtosis, excess_tails
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    z_values = mle_summary_df[z_col].dropna()

    # Plot histogram
    ax.hist(
        z_values,
        bins=50,
        density=True,
        alpha=0.6,
        color="steelblue",
        label="Observed",
    )

    # Overlay N(0,1)
    x = np.linspace(z_values.min(), z_values.max(), 200)
    from scipy.stats import norm

    ax.plot(x, norm.pdf(x, 0, 1), "r-", linewidth=2, label="N(0,1)")

    ax.set_xlabel("Wald Z-statistic", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Calculate metrics
    from scipy.stats import kurtosis

    kurt = kurtosis(z_values)

    # Check for excess tails (|Z| > 3)
    excess_tails = (z_values.abs() > 3).sum() / len(z_values)
    expected_tails = 0.0027  # for N(0,1)

    textstr = f"Kurtosis: {kurt:.2f}\nTails |Z|>3: {excess_tails:.2%}\n(Expected: {expected_tails:.2%})"  # noqa: E501
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    metrics = {"kurtosis": kurt, "excess_tails": excess_tails - expected_tails}

    return fig, ax, metrics


def plot_qq(pvalues, figsize=(8, 8), title="QQ-Plot of p-values", ax=None):
    """
    QQ-plot of observed vs. expected -log10(p-values).

    Assesses overall p-value calibration and genomic inflation.

    Parameters
    ----------
    pvalues : array-like
        P-values to plot
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    metrics : dict
        Contains: lambda_gc (genomic inflation factor)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Remove NaN and clip to valid range
    pvalues = np.array(pvalues)
    pvalues = pvalues[~np.isnan(pvalues)]
    pvalues = np.clip(pvalues, 1e-300, 1.0)

    # Sort observed p-values
    observed = -np.log10(np.sort(pvalues))

    # Expected under null
    n = len(pvalues)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))

    # Plot
    ax.scatter(expected, observed, alpha=0.5, s=20, c="steelblue")

    # Add diagonal
    max_val = max(expected.max(), observed.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Expected")

    # Calculate genomic inflation factor (lambda)
    from scipy.stats import chi2

    # Convert p-values to chi-squared statistics
    chisq = chi2.ppf(1 - pvalues, df=1)
    lambda_gc = np.median(chisq) / chi2.ppf(0.5, df=1)

    ax.set_xlabel("Expected -log10(p)", fontsize=11)
    ax.set_ylabel("Observed -log10(p)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Add lambda annotation
    textstr = f"λ_GC = {lambda_gc:.3f}"
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    metrics = {"lambda_gc": lambda_gc}

    return fig, ax, metrics


def plot_ranking_metric_heatmaps(
    metrics_df: pd.DataFrame,
    metrics: List[str] = ("kendall_tau", "spearman_r", "dcg"),
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Create a combined figure with one heatmap per ranking similarity metric.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of calculate_ranking_metrics with columns:
        ranking_a, ranking_b, <metric columns>
    metrics : Sequence[str]
        Metric columns to plot as heatmaps.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rankings = sorted(
        set(metrics_df["ranking_a"]).union(metrics_df["ranking_b"])
    )
    n = len(metrics)
    if figsize is None:
        figsize = (5 * n, 4.5)

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        mat = pd.DataFrame(
            np.nan,
            index=rankings,
            columns=rankings,
            dtype=float,
        )

        # fill symmetric matrix
        for _, row in metrics_df.iterrows():
            a = row["ranking_a"]
            b = row["ranking_b"]
            val = row[metric]
            mat.loc[a, b] = val
            mat.loc[b, a] = val

        np.fill_diagonal(mat.values, 1.0)

        im = ax.imshow(
            mat.values, vmin=np.nanmin(mat.values), vmax=np.nanmax(mat.values)
        )

        ax.set_title(metric)
        ax.set_xticks(range(len(rankings)))
        ax.set_yticks(range(len(rankings)))
        ax.set_xticklabels(rankings, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(rankings, fontsize=8)

        # annotate cells
        for i in range(len(rankings)):
            for j in range(len(rankings)):
                val = mat.iat[i, j]
                if np.isfinite(val):
                    ax.text(
                        j, i, f"{val:.2f}", ha="center", va="center", fontsize=8
                    )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig


def _save(fig: Figure, outfile: Path | str):
    base = Path(outfile).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(base.with_suffix(ext), bbox_inches="tight")


def get_dfs(
    substitution_frequencies: Mapping[str, Path],
) -> tuple[Mapping[str, pd.DataFrame], list[str]]:
    dfs = {}
    for name, path in substitution_frequencies.items():
        if not path or not Path(path).exists():
            # allow placeholders (e.g. empty Path) - skip them
            continue
        df = pd.read_csv(path, sep="\t", index_col=0)
        dfs[name] = df

    if not dfs:
        raise ValueError("No valid substitution frequency files found.")

    # all tables must have the same columns (same reference sequence positions)
    reference_columns = list(next(iter(dfs.values())).columns)
    for name, df in dfs.items():
        if list(df.columns) != reference_columns:
            raise ValueError(
                f"Substitution frequency table for {name} has different columns than others: "  # noqa: E501
                f"{list(df.columns)} != {reference_columns}"
            )
    return dfs, reference_columns


def _prepare_df(
    df: pd.DataFrame,
    base_order: list[str],
    input_type: Literal["counts", "percentages"],
    display_type: Literal["counts", "percentages"],
    omit_reference: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Reindex rows, apply conversion, return (df_ready, y_label)."""
    df = df.reindex(base_order).fillna(0.0)
    label = "Count"
    if input_type == "percentages":
        # percentages are always displayed as percentages
        df = df * 100
        label = "Frequency (%)"
    else:
        # input is counts
        if display_type == "percentages":
            col_sums = df.sum(axis=0).replace(0, 1)
            df = df.divide(col_sums, axis=1) * 100.0
            label = "Frequency (%)"

        # else display raw counts as is
    if omit_reference:
        for col in df.columns:
            ref_base = col.split(".")[0]
            df.loc[ref_base, col] = 0.0
    return df, label


def _strip_ref_label(col: str) -> str:
    """Return just the nucleotide character from a column name like 'C' or 'C.1'."""
    return col.split(".")[0]


def __plot_separate(
    outfile: Path | str,
    dfs_plot,
    ref_seq,
    positions,
    base_order,
    colors,
    y_label: str,
    display_type: Literal["counts", "percentages"],
    title: str | None,
    ylim: float | None = None,
):
    # One subfigure per sample, shared X-axis.
    names = sorted(dfs_plot.keys(), key=lambda k: (k == "base", k))
    n_pos = len(positions)
    fig, axes = plt.subplots(
        len(names),
        1,
        figsize=(max(8, n_pos * 0.35), 2.5 * len(names)),
        sharex=True,
        dpi=150,
    )
    if len(names) == 1:
        axes = [axes]

    x_labels = [_strip_ref_label(c) for c in ref_seq]

    for ax, name in zip(axes, names):
        df = dfs_plot[name]
        bottom = np.zeros(n_pos, dtype=float)
        for base in base_order:
            values = df.loc[base].to_numpy(dtype=float)
            ax.bar(
                positions,
                values,
                bottom=bottom,
                label=base,
                color=colors.get(base, "gray"),
                edgecolor="black",
            )
            bottom += values
        # if display_type == "percentages" and ylim is None:
        #    ax.set_ylim(0, 100)
        if ylim is not None:
            ax.set_ylim(0, ylim)
        ax.set_ylabel(f"{name}\n{y_label}")
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, rotation=0, fontsize=8)

    axes[-1].set_xlabel("Reference position")
    axes[0].legend(
        ncol=len(base_order),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        frameon=False,
        fontsize=9,
        title="Base",
    )
    if title:
        fig.suptitle(title, y=1.04)
    fig.tight_layout()
    _save(fig, outfile)
    plt.close(fig)
    return


HATCHES = ["", "///", "...", "xxx", "+++", "\\\\\\"]


def __plot_grouped(
    outfile: Path | str,
    dfs_plot,
    ref_seq,
    positions,
    base_order,
    colors,
    y_label: str,
    display_type: Literal["counts", "percentages"],
    title: str | None,
    ylim: float | None = None,
):
    n = len(dfs_plot)
    n_pos = len(positions)
    width = min(0.8, 0.8 / max(1, n))
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n)
    x_labels = [_strip_ref_label(c) for c in ref_seq]
    min_bar_width_inch = 0.3
    fig_width = np.clip(n_pos * n * min_bar_width_inch, 12, 200)

    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=150)

    legend_base_added: set[str] = set()
    sample_handles = []  # für zweite Legende

    for i, (offset, (name, df)) in enumerate(zip(offsets, dfs_plot.items())):
        hatch = HATCHES[i % len(HATCHES)]
        bottom = np.zeros(n_pos, dtype=float)

        for base in base_order:
            values = df.loc[base].to_numpy(dtype=float)
            label = base if base not in legend_base_added else "_nolegend_"
            ax.bar(
                positions + offset,
                values,
                width=width,
                bottom=bottom,
                label=label,
                color=colors.get(base, "gray"),
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )
            legend_base_added.add(base)
            bottom += values

        # Proxy-Artist für Sample-Legende
        sample_handles.append(
            mpatches.Patch(
                facecolor="white", edgecolor="black", hatch=hatch, label=name
            )
        )

    # Legende 1: Nukleotide (Farben) – oben zentriert
    legend1 = ax.legend(
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        fontsize=9,
        title="Base",
    )
    ax.add_artist(legend1)  # wichtig! sonst wird sie von legend2 überschrieben

    # Legende 2: Samples (Hatch) – daneben
    ax.legend(
        handles=sample_handles,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.55),
        frameon=True,
        fontsize=9,
        title="Sample",
    )

    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=8)
    ax.set_xlabel("Reference position")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, pad=50)

    fig.tight_layout(rect=[0, 0, 1, 0.92])  # space for legends above
    fig.subplots_adjust(top=0.82)  # adjust for both legends
    _save(fig, outfile)
    plt.close(fig)


def plot_substitution_frequency(
    substitution_frequencies: Mapping[str, Path],
    outfile: Path,
    dependencies: list[Job] = [],
    plottype: Literal["grouped", "separate"] = "separate",
    input_type: Literal["counts", "percentages"] = "counts",
    display_type: Literal["counts", "percentages"] = "percentages",
    omit_reference: bool = False,
    window: tuple[int, int] | None = None,
    title: str | None = None,
    # colors: dict = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728", "N": "#7f7f7f"},
    colors: dict = {
        "A": "#1b9e77",
        "C": "#377eb8",
        "G": "#d95f02",
        "T": "#e7298a",
        "N": "#999999",
    },
    ylim: float | None = None,
) -> Job:
    """
    plot_substitution_frequency collects the substitution frequencies per nucleotide
    position from the different samples and plots them as stacked bar charts (ACGTN).
    The X-axis labels show the reference nucleotide at each position.

    The plot is saved as a pdf, png and svg file.

    The function returns a Job that can be added to the pipeline. The Job depends on
    the jobs that generate the substitution frequency tables.


    Parameters
    ----------
    substitution_frequencies : Mapping[str, Path]
        A mapping of sample names to their corresponding substitution frequency table
        file paths.
    outfile : Path
        The path to the output file where the plot will be saved.
    dependencies : list[Job], optional
        A list of jobs that this job depends on, by default []
    plottype : Literal["grouped", "separate"], optional
        The type of bar chart to plot, by default "separate". Options are:
        - "grouped": Grouped bar chart – samples sit side-by-side at each position,
          each bar stacked ACGTN.
        - "separate": One subfigure per sample with a shared X-axis; each bar is a
          stacked ACGTN bar.
    input_type : Literal["counts", "percentages"], optional
        Whether the values in the input files are raw read counts or already
        percentages/frequencies, by default "counts".
    display_type : Literal["counts", "percentages"], optional
        Whether to display raw counts or convert to percentages, by default
        "percentages".  Ignored when input_type is "percentages" (always shown as %).
    window : tuple[int, int] | None, optional
        A (start, end) slice (0-based, end exclusive) into the full reference sequence
        to restrict plotting to a sub-region, by default None (plot all positions).
    title : str | None, optional
        An optional title to add to the figure, by default None.
    colors : dict, optional
        A mapping of nucleotide bases to colors for the plot, by default
        {"A": "green", "C": "blue", "G": "orange", "T": "red", "N": "gray"}.

    Returns
    -------
    Job
        A Job that can be added to the pipeline.

    """

    def __plot(outfile):
        # reference sequence is encoded in the column labels (one base per position)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        dfs, reference_columns = get_dfs(substitution_frequencies)
        base_order = ["A", "C", "G", "T", "N"]

        # apply window slice if requested
        if window is not None:
            start, end = window
            reference_columns = reference_columns[start:end]
            dfs = {name: df[reference_columns] for name, df in dfs.items()}

        ref_seq = reference_columns
        positions = np.arange(len(ref_seq))

        # prepare (convert / normalise) each dataframe
        dfs_plot = {}
        y_label = "Frequency (%)"
        for name, df in dfs.items():
            df_ready, y_label = _prepare_df(
                df, base_order, input_type, display_type, omit_reference
            )
            dfs_plot[name] = df_ready
        df_out = pd.concat(dfs_plot.values())
        df_out.to_csv(outfile.with_suffix(".prepared_data.csv"))
        if plottype == "separate":
            return __plot_separate(
                outfile,
                dfs_plot,
                ref_seq,
                positions,
                base_order,
                colors,
                y_label,
                display_type,
                title,
                ylim,
            )
        elif plottype == "grouped":
            return __plot_grouped(
                outfile,
                dfs_plot,
                ref_seq,
                positions,
                base_order,
                colors,
                y_label,
                display_type,
                title,
                ylim,
            )
        else:
            raise NotImplementedError(
                "Parameter plottype must be one of ['grouped', 'separate']"
            )

    return FileGeneratingJob(outfile, __plot).depends_on(dependencies)


def _scale(
    values: np.ndarray,
    scale: Literal["log"] | Callable | None,
    offset: float = 0,
) -> np.ndarray:
    if not scale:
        return values
    arr = np.array(values, dtype=np.float64)
    if isinstance(scale, str) and scale == "log":
        return -np.log10(
            offset + np.where(arr > 0, arr, np.nan)
        )  # -log10, NaN für <=0
    elif callable(scale):
        return scale(arr)
    else:
        raise ValueError(f"Invalid scale: {scale}")


def plot_effect_size_with_labels_zoom(
    df: DataFrame,
    effect_col: str,
    p_value_col: str,
    label_col: str | None = None,
    center_y: float = 0.0,
    zoom_on_ranks: tuple[int, int] | None = (0, 10),
    zoom_side: Literal["top", "bottom"] = "top",
    select: list[str] | None | Callable = None,
    scale_y: Literal["log"] | Callable | None = None,
    ylabel: str = "Gene",
    ascending: bool = True,
) -> tuple[Figure, Axes]:
    """
    Waterfall plot: rank (x) vs effect size (y) with optional inset zoom
    on top or bottom ranked selected genes.

    Parameters
    ----------
    ascending : bool
        If True, most negative effect is rank 0 (left). If False, most
        positive effect is rank 0 (left).
    zoom_on_ranks : tuple[int, int] | None
        Rank range (0-based, end exclusive) among selected genes to zoom in on.
    zoom_side : "top" | "bottom"
        "top" zooms on largest absolute effect, "bottom" on smallest.
    """

    # --- Daten vorbereiten ---
    df = df.copy()
    if effect_col not in df.columns:
        raise KeyError(f"Effect column '{effect_col}' not found in DataFrame")

    df[effect_col] = pd.to_numeric(df[effect_col], errors="coerce")

    if label_col is not None:
        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in DataFrame")
        df["label"] = df[label_col].astype(str)
    else:
        df["label"] = df.index.astype(str)

    df = df.dropna(subset=[effect_col])

    # Wasserfall: sortiert nach tatsächlichem Effektwert
    df = df.sort_values(effect_col, ascending=ascending).reset_index(drop=True)
    df["rank"] = df.index.astype(int)
    df = df.assign(_abs_effect=df[effect_col].abs())

    if df.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=12
        )
        ax.axis("off")
        return fig, ax

    # --- Selected Gene bestimmen ---
    if select is not None:
        if callable(select):
            try:
                sel_mask = select(df)
            except Exception as e:
                raise ValueError(f"select callable raised an error: {e}")
            sel_labels = set(df.loc[sel_mask, "label"])
        else:
            sel_labels = set(select)
    else:
        sel_labels = set()

    # --- Skalierung (nur y = effect sinnvoll) ---
    df["_y"] = _scale(df[effect_col].to_numpy(dtype=np.float64), scale_y)
    df["_x"] = df["rank"].astype(float)

    y_label = (
        f"-log10({effect_col})"
        if isinstance(scale_y, str) and scale_y == "log"
        else effect_col
    )

    def _color_by_sign(vals):
        return ["red" if v >= 0 else "blue" for v in vals]

    # --- Zoom-Bereich bestimmen (nach absolutem Effekt) ---
    df_zoom = None
    if zoom_on_ranks is not None and sel_labels:
        start_rank, end_rank = zoom_on_ranks
        start_rank = max(0, int(start_rank))
        end_rank = max(start_rank + 1, int(end_rank))

        df_sel = df[df["label"].isin(sel_labels)].copy()
        df_sel = df_sel.sort_values(
            "_abs_effect", ascending=(zoom_side == "bottom")
        ).reset_index(drop=True)
        df_sel["zoom_rank"] = df_sel.index.astype(int)
        df_zoom = df_sel[
            (df_sel["zoom_rank"] >= start_rank)
            & (df_sel["zoom_rank"] < end_rank)
        ].copy()

    has_zoom = df_zoom is not None and df_zoom.shape[0] > 0

    # --- Figure aufbauen ---
    fig, ax_main = plt.subplots(figsize=(12, 6))

    # --- Hauptplot: Wasserfall ---
    mask_sel = df["label"].isin(sel_labels)
    df_rest = df[~mask_sel]
    df_highlighted = df[mask_sel]

    # Alle Gene als kleine Punkte
    ax_main.scatter(
        df_rest["_x"],
        df_rest["_y"],
        c=_color_by_sign(df_rest[effect_col].values),
        s=10,
        alpha=0.4,
    )

    # Selected Gene hervorgehoben mit Labels
    if not df_highlighted.empty:
        ax_main.scatter(
            df_highlighted["_x"],
            df_highlighted["_y"],
            c=_color_by_sign(df_highlighted[effect_col].values),
            s=50,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        for _, row in df_highlighted.iterrows():
            ax_main.annotate(
                row["label"],
                xy=(row["_x"], row["_y"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.85,
            )

    ax_main.axhline(center_y, color="gray", linestyle="--", alpha=0.6)
    ax_main.set_xlabel("Rank")
    ax_main.set_ylabel(y_label)
    ax_main.set_title(f"Waterfall — {effect_col} by Rank ({ylabel} Labels)")
    ax_main.grid(alpha=0.2)

    # --- Zoom-Inset ---
    if has_zoom:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        x_zoom = df_zoom["_x"].values
        y_zoom = df_zoom["_y"].values
        x_pad = max((x_zoom.max() - x_zoom.min()) * 0.3, 2.0)
        y_pad = max((y_zoom.max() - y_zoom.min()) * 0.3, 0.05)
        x1, x2 = x_zoom.min() - x_pad, x_zoom.max() + x_pad
        y1, y2 = y_zoom.min() - y_pad, y_zoom.max() + y_pad

        # Inset-Position: oben links bei ascending (negative Effekte links),
        # oben rechts bei descending
        loc = "lower right" if ascending else "lower left"
        ax_inset = inset_axes(ax_main, width="35%", height="40%", loc=loc)

        # Alle Punkte im Fenster plotten
        df_in_window = df[
            (df["_x"] >= x1)
            & (df["_x"] <= x2)
            & (df["_y"] >= y1)
            & (df["_y"] <= y2)
        ]
        df_in_rest = df_in_window[~df_in_window["label"].isin(sel_labels)]
        df_in_sel = df_in_window[df_in_window["label"].isin(sel_labels)]

        ax_inset.scatter(
            df_in_rest["_x"],
            df_in_rest["_y"],
            c=_color_by_sign(df_in_rest[effect_col].values),
            s=15,
            alpha=0.4,
        )
        ax_inset.scatter(
            df_in_sel["_x"],
            df_in_sel["_y"],
            c=_color_by_sign(df_in_sel[effect_col].values),
            s=40,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        for _, row in df_zoom.iterrows():
            ax_inset.annotate(
                row["label"],
                xy=(row["_x"], row["_y"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

        ax_inset.set_xlim(x1, x2)
        ax_inset.set_ylim(y1, y2)
        ax_inset.axhline(
            center_y, color="gray", linestyle="--", alpha=0.5, lw=0.8
        )
        ax_inset.set_title(
            f"{'top' if zoom_side == 'top' else 'bottom'} "
            f"{len(df_zoom)} selected (ranks {start_rank}–{end_rank - 1})",
            fontsize=8,
        )
        ax_inset.tick_params(labelsize=7)

        mark_inset(
            ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="gray", lw=0.8
        )

    fig.tight_layout()
    return fig, ax_main
