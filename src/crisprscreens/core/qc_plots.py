"""
Plotting functions for pairing QC analysis.
"""

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from typing import Tuple, Optional, Dict, List
from pandas import DataFrame  # type: ignore
from matplotlib.figure import Figure  # type: ignore


def plot_replicate_correlation_heatmap(
    correlation_matrix: DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".3f",
) -> Figure:
    """
    Plot heatmap of replicate rank correlations.

    Parameters
    ----------
    correlation_matrix : DataFrame
        Correlation matrix from replicate_gene_ranking_consistency
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    vmin : float
        Minimum value for colormap
    vmax : float
        Maximum value for colormap
    annot : bool
        Whether to annotate cells with values
    fmt : str
        Format string for annotations

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    print(correlation_matrix)
    sns.heatmap(
        correlation_matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        square=True,
        cbar_kws={"label": "Spearman Correlation"},
        linewidths=0.5,
    )

    if title is None:
        title = "Replicate Gene Rank Correlations"
    ax.set_title(title, fontsize=14, pad=10)

    plt.tight_layout()
    return fig


def plot_top_n_overlap_heatmap(
    overlap_matrix: DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".3f",
) -> Figure:
    """
    Plot heatmap of top-N gene overlap (Jaccard indices).

    Parameters
    ----------
    overlap_matrix : DataFrame
        Overlap matrix from replicate_gene_ranking_consistency
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    vmin : float
        Minimum value for colormap
    vmax : float
        Maximum value for colormap
    annot : bool
        Whether to annotate cells with values
    fmt : str
        Format string for annotations

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        overlap_matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        square=True,
        cbar_kws={"label": "Jaccard Index"},
        linewidths=0.5,
    )

    if title is None:
        title = "Top-100 Gene Overlap Between Replicates"
    ax.set_title(title, fontsize=14, pad=10)

    plt.tight_layout()
    return fig


def plot_paired_vs_unpaired_scatter(
    paired_results: DataFrame,
    unpaired_results: DataFrame,
    metric: str = "neg|rank",
    gene_col: str = "id",
    highlight_genes: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    point_size: float = 20,
    alpha: float = 0.5,
    highlight_color: str = "red",
) -> Figure:
    """
    Scatter plot comparing paired vs unpaired gene rankings.

    Parameters
    ----------
    paired_results : DataFrame
        MAGeCK paired gene summary
    unpaired_results : DataFrame
        MAGeCK unpaired gene summary
    metric : str
        Column name to compare (e.g., "neg|rank", "neg|score")
    gene_col : str
        Gene column name
    highlight_genes : list, optional
        List of genes to highlight
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    point_size : float
        Marker size
    alpha : float
        Point transparency
    highlight_color : str
        Color for highlighted genes

    Returns
    -------
    Figure
        Matplotlib figure
    """
    # Detect actual column names (MAGeCK may vary)
    if "Gene" in paired_results.columns:
        gene_col_actual = "Gene"
    elif "id" in paired_results.columns:
        gene_col_actual = "id"
    else:
        gene_col_actual = gene_col

    # Merge
    merged = unpaired_results[[gene_col_actual, metric]].merge(
        paired_results[[gene_col_actual, metric]],
        on=gene_col_actual,
        how="inner",
        suffixes=("_unpaired", "_paired"),
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Main scatter
    ax.scatter(
        merged[f"{metric}_unpaired"],
        merged[f"{metric}_paired"],
        s=point_size,
        alpha=alpha,
        c="grey",
        edgecolors="none",
    )

    # Highlight genes if provided
    if highlight_genes is not None:
        highlight_mask = merged[gene_col_actual].isin(highlight_genes)
        highlight_data = merged[highlight_mask]

        ax.scatter(
            highlight_data[f"{metric}_unpaired"],
            highlight_data[f"{metric}_paired"],
            s=point_size * 2,
            alpha=1.0,
            c=highlight_color,
            edgecolors="black",
            linewidths=0.5,
            label=f"Highlighted ({len(highlight_data)} genes)",
        )

        # Label highlighted genes
        for _, row in highlight_data.iterrows():
            ax.annotate(
                row[gene_col_actual],
                xy=(row[f"{metric}_unpaired"], row[f"{metric}_paired"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        ax.legend()

    # Diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, zorder=0, linewidth=1)

    # Compute correlation
    from scipy.stats import spearmanr

    corr, pval = spearmanr(
        merged[f"{metric}_unpaired"], merged[f"{metric}_paired"]
    )

    ax.text(
        0.05,
        0.95,
        f"Spearman r = {corr:.3f}\np = {pval:.2e}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if title is None:
        title = f"Paired vs. Unpaired Gene Rankings"
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(f"Unpaired {metric}", fontsize=12)
    ax.set_ylabel(f"Paired {metric}", fontsize=12)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_downsampling_stability(
    stability_df: DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    color: str = "steelblue",
) -> Figure:
    """
    Plot downsampling stability across fractions.

    Parameters
    ----------
    stability_df : DataFrame
        Stability summary from downsampling_stability_qc
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    color : str
        Color for bars/points

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Bar plot with error bars
    x = stability_df["fraction"].values
    y = stability_df["mean"].values
    yerr = stability_df["std"].values

    ax.bar(x, y, width=0.15, color=color, alpha=0.7, yerr=yerr, capsize=5)
    ax.plot(x, y, "o-", color=color, markersize=8, linewidth=2)

    # Reference line at 1.0 (full stability)
    ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5, linewidth=1)

    # Annotate values
    for xi, yi, yerri in zip(x, y, yerr):
        ax.text(
            xi,
            yi + yerri + 0.02,
            f"{yi:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if title is None:
        title = "Top-100 Gene Stability Under Downsampling"
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Fraction of Original Read Depth", fontsize=12)
    ax.set_ylabel("Mean Jaccard Index\n(vs. Full Depth)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(xi*100)}%" for xi in x])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_positive_control_ranks(
    gene_summary: DataFrame,
    positive_controls: List[str],
    rank_col: str = "neg|rank",
    gene_col: str = "id",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    highlight_color: str = "red",
) -> Figure:
    """
    Plot rank distribution with positive controls highlighted.

    Parameters
    ----------
    gene_summary : DataFrame
        MAGeCK gene summary
    positive_controls : list
        List of positive control gene names
    rank_col : str
        Rank column name
    gene_col : str
        Gene column name
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    highlight_color : str
        Color for positive controls

    Returns
    -------
    Figure
        Matplotlib figure
    """
    # Detect actual column names
    if "Gene" in gene_summary.columns:
        gene_col_actual = "Gene"
    elif "id" in gene_summary.columns:
        gene_col_actual = "id"
    else:
        gene_col_actual = gene_col

    fig, ax = plt.subplots(figsize=figsize)

    # All genes
    ranks = gene_summary[rank_col].values
    ax.hist(ranks, bins=50, color="grey", alpha=0.5, label="All genes")

    # Positive controls
    pos_mask = gene_summary[gene_col_actual].isin(positive_controls)
    pos_ranks = gene_summary[pos_mask][rank_col].values

    if len(pos_ranks) > 0:
        ax.hist(
            pos_ranks,
            bins=50,
            color=highlight_color,
            alpha=0.7,
            label=f"Positive controls (n={len(pos_ranks)})",
        )

        # Add vertical lines at median/mean
        median_pos = np.median(pos_ranks)
        mean_pos = np.mean(pos_ranks)

        ax.axvline(
            median_pos,
            color=highlight_color,
            linestyle="--",
            linewidth=2,
            label=f"Median pos. control rank: {median_pos:.0f}",
        )

        # Add annotation
        n_in_top100 = (pos_ranks <= 100).sum()
        frac_in_top100 = n_in_top100 / len(pos_ranks)

        ax.text(
            0.98,
            0.98,
            f"Positive controls in top-100:\n{n_in_top100}/{len(pos_ranks)} ({frac_in_top100:.1%})",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    if title is None:
        title = "Gene Rank Distribution with Positive Controls"
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Gene Rank", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pairing_decision_summary(
    recommendation_dict: Dict,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """
    Summary plot showing score components for pairing decision.

    Parameters
    ----------
    recommendation_dict : dict
        Recommendation dictionary from comprehensive_pairing_qc
    title : str, optional
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract score components
    components = recommendation_dict["score_components"]
    names = [c["component"] for c in components]
    scores = [c["score"] for c in components]
    weights = [c["weight"] for c in components]

    # Create color map
    colors = []
    for score in scores:
        if score >= 0.7:
            colors.append("green")
        elif score >= 0.5:
            colors.append("orange")
        else:
            colors.append("red")

    # Horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.7)

    # Add weight labels
    for i, (score, weight) in enumerate(zip(scores, weights)):
        ax.text(
            score + 0.02,
            i,
            f"weight: {weight:.1f}",
            va="center",
            fontsize=9,
            alpha=0.7,
        )

    # Reference lines
    ax.axvline(x=0.5, color="grey", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0.7, color="grey", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_xlim(0, 1.0)

    # Add overall recommendation
    overall_score = recommendation_dict["recommendation_score"]
    recommendation = recommendation_dict["recommendation"].upper()

    rec_color = (
        "green"
        if overall_score >= 0.6
        else "red" if overall_score < 0.4 else "orange"
    )

    ax.text(
        0.98,
        0.02,
        f"Overall Score: {overall_score:.3f}\nRecommendation: {recommendation}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor=rec_color,
            alpha=0.3,
            edgecolor=rec_color,
            linewidth=2,
        ),
    )

    if title is None:
        title = "Pairing Decision: Score Components"
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig
