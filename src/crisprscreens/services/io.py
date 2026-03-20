import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from pandas import DataFrame  # type: ignore

from crisprscreens.core.qc import (
    control_sgrna_qc,
    export_control_counts_and_cpm,
    generate_standard_qc_report as _generate_standard_qc_report,
    calculate_ranking_metrics,
)
from crisprscreens.core.mageck_report import (
    generate_mageck_report as _generate_mageck_report,
)
from crisprscreens.core.plots import (
    plot_control_distribution_per_condition,
    plot_control_pca,
    plot_control_replicate_correlation,
    plot_pairwise_control_shifts,
)
from crisprscreens.core.mageck import (
    mageck_pathway as _mageck_pathway,
    mageck_plot as _mageck_plot,
)


def save_figure(f, folder, name, bbox_inches="tight"):
    folder.mkdir(exist_ok=True, parents=True)
    for suffix in [".png", ".svg", ".pdf"]:
        f.savefig(folder / (name + suffix), bbox_inches=bbox_inches)


# def control_qc_report(
#     count_table: Union[Path, str],
#     control_sgrnas: Union[Path, str],
#     baseline_condition: str,
#     output_dir: Union[Path, str],
#     prefix: str = "control_qc",
#     sgrna_col: str = "sgRNA",
#     gene_col: str = "Gene",
#     delimiter: str = "_",
#     save_formats: List[str] = ["png", "pdf"],
# ):
#     """
#     Generate comprehensive control sgRNA QC report.

#     Wrapper around core.qc.generate_control_qc_report for service layer.

#     Parameters
#     ----------
#     count_table : Path or str
#         Path to MAGeCK count table.
#     control_sgrnas : Path or str
#         Path to control sgRNA file.
#     baseline_condition : str
#         Baseline condition name (e.g., "Total", "T0").
#     output_dir : Path or str
#         Output directory for QC reports.
#     prefix : str
#         Filename prefix for output files.
#     sgrna_col : str
#         sgRNA column name in count table.
#     gene_col : str
#         Gene column name in count table.
#     delimiter : str
#         Delimiter for parsing condition_replicate.
#     save_formats : list
#         List of formats to save ("png", "pdf", "svg").

#     Returns
#     -------
#     dict
#         QC results with file paths.
#     """
#     from crisprscreens.core.qc import generate_control_qc_report

#     return generate_control_qc_report(
#         count_table=count_table,
#         control_sgrnas=control_sgrnas,
#         baseline_condition=baseline_condition,
#         output_dir=output_dir,
#         prefix=prefix,
#         sgrna_col=sgrna_col,
#         gene_col=gene_col,
#         delimiter=delimiter,
#         save_formats=save_formats,
#     )


def generate_control_qc_report(
    count_table: Union[Path, str, DataFrame],
    control_sgrnas: Union[Path, str, set],
    baseline_condition: str,
    output_dir: Union[Path, str],
    prefix: str = "control_qc",
    sgrna_col: str = "sgRNA",
    gene_col: str = "Gene",
    delimiter: str = "_",
    save_formats: List[str] = ["png", "pdf"],
) -> Dict:
    """
    Generate comprehensive control sgRNA QC report.

    Creates all QC plots and saves metrics to file.

    Parameters
    ----------
    count_table : Path, str, or DataFrame
        MAGeCK count table.
    control_sgrnas : Path, str, or set
        Control sgRNA IDs.
    baseline_condition : str
        Baseline condition name.
    output_dir : Path or str
        Output directory.
    prefix : str
        Filename prefix.
    sgrna_col : str
        sgRNA column name.
    gene_col : str
        Gene column name.
    delimiter : str
        Condition/replicate delimiter.
    save_formats : list
        List of formats to save ("png", "pdf", "svg").

    Returns
    -------
    dict
        QC results with file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Run QC analysis
    print("Running control sgRNA QC analysis...")
    qc_results = control_sgrna_qc(
        count_table=count_table,
        control_sgrnas=control_sgrnas,
        baseline_condition=baseline_condition,
        sgrna_col=sgrna_col,
        gene_col=gene_col,
        delimiter=delimiter,
    )

    # Save metrics
    metrics_file = output_dir / f"{prefix}_metrics.tsv"
    metrics_df = pd.DataFrame(qc_results["metrics"]).T
    metrics_df.to_csv(metrics_file, sep="\t")
    print(f"Saved metrics to {metrics_file}")

    # Save pairwise median shifts
    pairwise_file = output_dir / f"{prefix}_pairwise_shifts.tsv"
    qc_results["pairwise_median"].to_csv(pairwise_file, sep="\t")
    print(f"Saved pairwise shifts to {pairwise_file}")
    cpm_files = export_control_counts_and_cpm(
        count_table=qc_results["raw_counts"],
        control_sgrnas=control_sgrnas,
        output_dir=output_dir,
        prefix=f"{prefix}",
        sgrna_col=sgrna_col,
        gene_col=gene_col,
    )
    saved_files = {
        "metrics": metrics_file,
        "pairwise_shifts": pairwise_file,
    }
    saved_files.update(cpm_files)

    # Generate and save plots
    plots_to_generate = [
        ("distribution", plot_control_distribution_per_condition),
        ("pairwise_heatmap", plot_pairwise_control_shifts),
        ("replicate_correlation", plot_control_replicate_correlation),
        (
            "pca_condition",
            lambda qc: plot_control_pca(qc, color_by="condition"),
        ),
        (
            "pca_replicate",
            lambda qc: plot_control_pca(qc, color_by="replicate"),
        ),
    ]

    for plot_name, plot_func in plots_to_generate:
        print(f"Generating {plot_name} plot...")
        try:
            fig_result = plot_func(qc_results)
            if isinstance(fig_result, tuple):
                fig = fig_result[0]
            else:
                fig = fig_result

            for fmt in save_formats:
                outfile = output_dir / f"{prefix}_{plot_name}.{fmt}"
                fig.savefig(outfile, dpi=300, bbox_inches="tight")
                saved_files[f"{plot_name}_{fmt}"] = outfile

            plt.close(fig)
            print(f"  Saved {plot_name}")
        except Exception as e:
            print(f"  Warning: Failed to generate {plot_name}: {e}")

    print(f"\nControl QC report complete. Files saved to {output_dir}")

    return {
        "qc_results": qc_results,
        "files": saved_files,
    }


def generate_spike_evaluation_report(
    eval_table: Union[Path, str],
    output_dir: Union[Path, str],
    prefix: str = "spike_evaluation",
    top_n: int = 5,
    pdf_report: bool = True,
) -> Dict:
    """
    Generate a human-readable spike-in evaluation report (markdown + optional PDF).

    Parameters
    ----------
    eval_table : Path or str
        TSV file produced by `evaluate_spike_in_performance_job` (ranked methods)
    output_dir : Path or str
        Directory to save report files
    prefix : str
        Filename prefix for outputs (default: 'spike_evaluation')
    top_n : int
        Number of top methods to include in the summary table
    pdf_report : bool
        Whether to attempt generating a PDF (requires optional dependencies)

    Returns
    -------
    dict
        Dict of saved file paths and a short summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read table
    df = read_dataframe(eval_table, dtype={})
    print(df.head())

    # Ensure composite score and rank exist (compute if necessary)
    if "composite_score" not in df.columns or "rank" not in df.columns:
        try:
            from crisprscreens.services.spike_evaluation import (
                rank_mageck_methods,
            )

            df = rank_mageck_methods(df)
        except Exception:
            # If ranking fails, proceed but warn
            print(
                "Warning: Could not compute composite_score/rank; report will use available columns"
            )

    # Prepare files
    md_file = output_dir / f"{prefix}.md"
    html_file = output_dir / f"{prefix}.html"
    png_file = output_dir / f"{prefix}_scores.png"
    pdf_file = output_dir / f"{prefix}.pdf"

    # Summary lines
    lines = []
    lines.append(f"# Spike-in Evaluation Report: {prefix}\n")
    lines.append(
        "This report summarizes performance metrics for different MAGeCK analysis methods using spike-in controls. It includes an executive summary, a top-methods table, simple interpretations and recommendations.\n"
    )
    # Top methods
    if "rank" in df.columns:
        top = df.sort_values("rank").head(top_n)
    elif "final_score" in df.columns:
        top = df.sort_values("final_score", ascending=False).head(top_n)
    elif "composite_score" in df.columns:
        top = df.sort_values("composite_score", ascending=False).head(top_n)
    else:
        top = df.head(top_n)

    lines.append("## Top methods\n")
    # Create small table
    table_cols = [
        c
        for c in [
            "rank",
            "comparison",
            "final_score",
            "composite_score",
            "f1",
            "precision",
            "recall",
            "auc_roc",
            "aucc",
            "n_detected_hits",
            "n_expected_hits",
        ]
        if c in top.columns
    ]
    if len(table_cols) > 0:
        # Header
        header = " | ".join([c.capitalize() for c in table_cols])
        lines.append(header)
        lines.append(" | ".join(["---"] * len(table_cols)))
        for _, row in top.iterrows():
            vals = [
                (
                    f"{row[c]:.3f}"
                    if isinstance(row[c], (float, int)) and not pd.isna(row[c])
                    else str(row.get(c, ""))
                )
                for c in table_cols
            ]
            lines.append(" | ".join(vals))
        lines.append("\n")
    else:
        lines.append(
            "(No suitable columns found to display a top-methods table)\n"
        )

    # Add interpretation heuristics for the top method
    if len(top) > 0:
        best = top.iloc[0]
        lines.append("## Interpretation & Recommendations\n")
        comp = best.get("composite_score", None)
        final = best.get("final_score", None)
        f1 = best.get("f1", None)
        prec = best.get("precision", None)
        rec = best.get("recall", None)
        auc = best.get("auc_roc", None)
        detected = best.get("n_detected_hits", None)
        expected = best.get("n_expected_hits", None)

        lines.append(
            f"**Top method:** {best.get('comparison', 'unknown')} (rank {int(best.get('rank')) if not pd.isna(best.get('rank')) else 'NA'})\n"
        )

        # Heuristic messages
        if pd.notna(f1) and f1 >= 0.8:
            lines.append(
                f"- **Very good detection (F1 = {f1:.2f})** — good balance between precision and recall.\n"
            )
        elif pd.notna(f1) and f1 >= 0.5:
            lines.append(
                f"- **Moderate detection (F1 = {f1:.2f})** — consider manual inspection of top hits.\n"
            )
        elif pd.notna(f1):
            lines.append(
                f"- **Low detection (F1 = {f1:.2f})** — results may not be reliable for calling hits.\n"
            )

        if pd.notna(auc):
            if auc >= 0.9:
                lines.append(
                    f"- **Excellent ranking power (AUC-ROC = {auc:.2f})** — top-ranked genes are well separated.\n"
                )
            elif auc >= 0.75:
                lines.append(
                    f"- **Reasonable ranking power (AUC-ROC = {auc:.2f})** — inspect top-k enrichment.\n"
                )
            else:
                lines.append(
                    f"- **Poor ranking power (AUC-ROC = {auc:.2f})** — top-ranked genes may include many false positives.\n"
                )

        if pd.notna(detected) and pd.notna(expected):
            lines.append(
                f"- **Detected hits:** {int(detected)}/{int(expected)} expected spike-in hits detected.\n"
            )
            if expected > 0 and detected / max(1, expected) < 0.2:
                lines.append(
                    "  - Low detection fraction — investigate normalization, sample depth, or effect-size thresholds.\n"
                )

        # Composite score interpretation
        if pd.notna(comp):
            lines.append(
                f"- **Composite score:** {comp:.3f} — this aggregates multiple metrics (higher is better).\n"
            )
            lines.append(
                "  - Recommendation: prioritize methods with high composite scores and good F1/AUC values.\n"
            )
        # Composite score interpretation
        if pd.notna(final):
            lines.append(
                f"- **Final score:** {final:.3f} — this aggregates quality (Cohen's D, neg_CV) and detection scores (recall) metrics (higher is better).\n"
            )
            lines.append(
                "  - Recommendation: prioritize methods with high F1, recall and quality values.\n"
            )

        # Generic recommendations
        lines.append("\n### Actionable recommendations\n")
        if pd.notna(f1) and f1 >= 0.8 and pd.notna(auc) and auc >= 0.8:
            lines.append(
                f"- Use **{best.get('comparison')}** as primary analysis method for downstream hit-calling.\n"
            )
        else:
            lines.append(
                "- No single method shows unambiguous high performance — consider combining results or manually inspecting top-ranked genes.\n"
            )
        lines.append(
            "- Visualize top hits and check sgRNA-level consistency before validation.\n"
        )
        lines.append(
            "- If detection is low, re-check sample group contrasts and consider increasing LFC threshold or FDR cutoff.\n"
        )

    else:
        lines.append(
            "No methods available to interpret. Check the input evaluation table.\n"
        )

    # Save markdown
    md_text = "\n".join(lines)
    md_file.write_text(md_text)
    print(f"Saved markdown report to {md_file}")

    # Create simple bar plot of composite scores
    try:
        plot_df = (
            df.dropna(subset=["final_score"])
            if "final_score" in df.columns
            else df
        )
        plot_df = plot_df.sort_values("final_score", ascending=False).head(
            max(top_n, len(plot_df))
        )
        import matplotlib.pyplot as _plt

        fig, ax = _plt.subplots(figsize=(8, 4))
        ax.barh(
            plot_df.get("comparison", plot_df.index).astype(str),
            plot_df.get("final_score", 0),
        )
        ax.invert_yaxis()
        ax.set_xlabel("Final score")
        ax.set_title("Method final scores")
        fig.tight_layout()
        fig.savefig(png_file, dpi=300)
        _plt.close(fig)
        print(f"Saved final score plot to {png_file}")
    except Exception as e:
        print(f"Warning: failed to generate final score plot: {e}")

    saved_files = {"md": md_file, "png": png_file}

    # Convert markdown to HTML and optionally to PDF
    try:
        from markdown import markdown as _md_to_html

        html = _md_to_html(md_text, extensions=["tables", "fenced_code"])
        html_file.write_text(html)
        saved_files["html"] = html_file
        print(f"Saved HTML report to {html_file}")

        if pdf_report:
            try:
                from weasyprint import HTML as _WeasyHTML

                _WeasyHTML(string=html).write_pdf(str(pdf_file))
                saved_files["pdf"] = pdf_file
                print(f"Saved PDF report to {pdf_file}")
            except Exception as e:
                print(
                    f"PDF generation failed (weasyprint not available or error): {e}"
                )
    except Exception as e:
        print(f"HTML conversion skipped: {e}")

    return {"files": saved_files, "summary": md_text}


def mageck_pathway(
    gene_ranking: Union[Path, str],
    gmt_file: Union[Path, str],
    output_dir: Union[Path, str],
    prefix: str = "pathway",
    method: str = "gsea",
    single_ranking: bool = False,
    sort_criteria: str = "neg",
    keep_tmp: bool = False,
    ranking_column: Optional[Union[str, int]] = None,
    ranking_column_2: Optional[Union[str, int]] = None,
    pathway_alpha: Optional[float] = None,
    permutation: Optional[int] = None,
    other_parameter: Dict[str, str] = [],
) -> Dict:
    """
    Service wrapper for `mageck pathway`.

    Returns a dict from the underlying core function containing stdout/stderr
    and output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return _mageck_pathway(
        gene_ranking=gene_ranking,
        gmt_file=gmt_file,
        out_dir=output_dir,
        prefix=prefix,
        method=method,
        single_ranking=single_ranking,
        sort_criteria=sort_criteria,
        keep_tmp=keep_tmp,
        ranking_column=ranking_column,
        ranking_column_2=ranking_column_2,
        pathway_alpha=pathway_alpha,
        permutation=permutation,
        other_parameter=other_parameter,
    )


def mageck_plot(
    gene_summary: Optional[Union[Path, str]] = None,
    sgrna_summary: Optional[Union[Path, str]] = None,
    output_dir: Union[Path, str] = ".",
    prefix: str = "plot",
    other_parameter: Dict[str, str] = [],
) -> Dict:
    """
    Service wrapper for `mageck plot`.

    Returns a dict from the underlying core function containing stdout/stderr
    and output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return _mageck_plot(
        gene_summary=gene_summary,
        sgrna_summary=sgrna_summary,
        out_dir=output_dir,
        prefix=prefix,
        other_parameter=other_parameter,
    )


def read_dataframe(path: Union[str, Path], **kwargs) -> DataFrame:
    """
    Read a tabular file into a pandas DataFrame based on file extension.

    Rules:
    - .csv        -> read as CSV
    - .tsv        -> read as TSV
    - .txt        -> treated as TSV
    - .xls/.xlsx  -> read as Excel
    - other       -> try TSV, raise error if that fails

    Additional keyword arguments are forwarded to the pandas reader.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(path, **kwargs)

        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t", **kwargs)

        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path, **kwargs)

        # Fallback: try TSV for unknown extensions
        try:
            return pd.read_csv(path, sep="\t", **kwargs)
        except Exception as exc:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                "Tried to read as TSV but failed."
            ) from exc

    except Exception as exc:
        raise RuntimeError(f"Failed to read file '{path}': {exc}") from exc


def standard_qc_report(
    count_tsv: Union[Path, str],
    output_dir: Union[Path, str],
    metadata_tsv: Union[Path, str, None] = None,
    control_sgrna_txt: Union[Path, str, None] = None,
    baseline_condition: str = "total",
    samples_to_select: Optional[List[str]] = None,
    **kwargs,
) -> Dict:
    """
    Generate standard QC report for CRISPR screen.

    Service wrapper for core.qc.generate_standard_qc_report.

    Parameters
    ----------
    count_tsv : Path or str
        Path to MAGeCK count table.
    output_dir : Path or str
        Output directory.
    metadata_tsv : Path or str, optional
        Path to metadata file.
    control_sgrna_txt : Path or str, optional
        Path to control sgRNA file.
    baseline_condition : str
        Baseline condition name.
    samples_to_select : list of str, optional
        If provided, only analyze these samples from count table.
    **kwargs
        Additional arguments passed to generate_standard_qc_report.

    Returns
    -------
    dict
        QC results.
    """
    return _generate_standard_qc_report(
        count_tsv=count_tsv,
        output_dir=output_dir,
        metadata_tsv=metadata_tsv,
        control_sgrna_txt=control_sgrna_txt,
        baseline_condition=baseline_condition,
        samples_to_select=samples_to_select,
        **kwargs,
    )


def mageck_report(
    output_dir,
    gene_summary_path=None,
    sgrna_summary_path=None,
    rra_summary_path=None,
    mle_summary_path=None,
    readout="full",
    effect_cols=None,
    gene_col="Gene",
    fdr_threshold=0.25,
    top_n=50,
    gene_sets=None,
    pathway_enrichment_path=None,
    **kwargs,
):
    """
    Generate comprehensive MAGeCK reporting with specialized plots.

    Service layer wrapper around core.mageck_report.generate_mageck_report.

    Creates publication-ready visualizations addressing key questions:
    - "Is this a real hit?" → Effect-size vs reproducibility, rank stability
    - "Is this experimentally stable?" → Replicate heatmaps, direction
       consistency
    - "What is functionally connected?" → Pathway enrichment, gene-set
      distributions
    - "How good is the model?" → Beta vs SE, Wald-Z, QQ-plots

    Parameters
    ----------
    output_dir : str or Path
        Directory to save report outputs
    gene_summary_path : str or Path, optional
        Path to MAGeCK gene summary (for MLE or single RRA)
    sgrna_summary_path : str or Path, optional
        Path to MAGeCK sgRNA summary
    rra_summary_path : str or Path, optional
        Path to MAGeCK RRA gene summary (for comparison with MLE)
    mle_summary_path : str or Path, optional
        Path to MAGeCK MLE gene summary (for comparison with RRA)
    readout : str
        Report mode: "minimal" (5 key plots) or "full" (all 11+ plots)
        - minimal: volcano, effect-vs-reproducibility, rank stability,
                   direction consistency, pathway enrichment
        - full: adds replicate heatmap, effect decomposition, contrast,
                gene-set distributions, beta-vs-SE, Wald-Z, QQ-plot,
                plus executive summary metrics
    effect_cols : list, optional
        For MLE: list of beta column names for effect decomposition
        E.g., ['beta|Time', 'beta|Met', 'beta|High']
    gene_col : str
        Column name for gene identifiers (default "Gene")
    fdr_threshold : float
        FDR threshold for significance (default 0.25)
    top_n : int
        Number of top genes to highlight in plots (default 50)
    gene_sets : dict, optional
        Dictionary mapping gene set names to lists of genes
        For gene-set score distribution analysis
        Example: {'Kinases': ['BRAF', 'RAF1', ...], 'TFs': ['TP53', ...]}
    pathway_enrichment_path : str or Path, optional
        Path to pathway enrichment results (CSV/TSV)
        Expected columns: pathway, pvalue, gene_ratio
    **kwargs : dict
        Additional parameters passed to plotting functions

    Returns
    -------
    dict
        Summary metrics and paths to generated plots containing:
        - 'output_dir': Path to output directory
        - 'readout': Mode used ('minimal' or 'full')
        - 'plots': Dictionary mapping plot types to file paths
        - 'metrics': Dictionary of computed metrics per plot
        - 'executive_summary': Overall summary metrics (full mode only)

    Examples
    --------
    Minimal mode (quick check):
    >>> results = mageck_report(
    ...     output_dir="results/mageck_report",
    ...     gene_summary_path="results/mageck_mle/gene_summary.txt",
    ...     sgrna_summary_path="results/mageck_mle/sgrna_summary.txt",
    ...     readout="minimal"
    ... )

    Full mode with method comparison:
    >>> results = mageck_report(
    ...     output_dir="results/mageck_report_full",
    ...     rra_summary_path="results/mageck_test/gene_summary.txt",
    ...     mle_summary_path="results/mageck_mle/gene_summary.txt",
    ...     sgrna_summary_path="results/mageck_mle/sgrna_summary.txt",
    ...     readout="full",
    ...     effect_cols=['beta|Treatment', 'beta|Time'],
    ...     gene_sets={'Kinases': kinase_genes, 'TFs': tf_genes},
    ...     fdr_threshold=0.25
    ... )
    """
    return _generate_mageck_report(
        output_dir=output_dir,
        gene_summary_path=gene_summary_path,
        sgrna_summary_path=sgrna_summary_path,
        rra_summary_path=rra_summary_path,
        mle_summary_path=mle_summary_path,
        readout=readout,
        effect_cols=effect_cols,
        gene_col=gene_col,
        fdr_threshold=fdr_threshold,
        top_n=top_n,
        gene_sets=gene_sets,
        pathway_enrichment_path=pathway_enrichment_path,
        **kwargs,
    )


# ...existing code...


def write_replicate_correlation_heatmap(
    filename: str,
    folder: Union[Path, str],
    correlation_matrix: Union[Path, str, DataFrame],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".3f",
):
    """Write replicate correlation heatmap to file."""
    from crisprscreens.core.qc_plots import plot_replicate_correlation_heatmap

    if isinstance(correlation_matrix, (str, Path)):
        correlation_matrix = read_dataframe(correlation_matrix)
        correlation_matrix.set_index(
            correlation_matrix.columns[0], inplace=True
        )
    print(correlation_matrix)
    fig = plot_replicate_correlation_heatmap(
        correlation_matrix=correlation_matrix,
        title=title,
        figsize=figsize,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
    )

    save_figure(fig, Path(folder), filename)


def write_top_n_overlap_heatmap(
    filename: str,
    folder: Union[Path, str],
    overlap_matrix: Union[Path, str, DataFrame],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".3f",
):
    """Write top-N overlap heatmap to file."""
    from crisprscreens.core.qc_plots import plot_top_n_overlap_heatmap

    if isinstance(overlap_matrix, (str, Path)):
        overlap_matrix = read_dataframe(overlap_matrix)
        overlap_matrix.set_index(overlap_matrix.columns[0], inplace=True)

    fig = plot_top_n_overlap_heatmap(
        overlap_matrix=overlap_matrix,
        title=title,
        figsize=figsize,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
    )

    save_figure(fig, Path(folder), filename)


def write_paired_vs_unpaired_scatter(
    filename: str,
    folder: Union[Path, str],
    paired_results: Union[Path, str, DataFrame],
    unpaired_results: Union[Path, str, DataFrame],
    metric: str = "neg|rank",
    gene_col: str = "id",
    highlight_genes: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    point_size: float = 20,
    alpha: float = 0.5,
    highlight_color: str = "red",
):
    """Write paired vs unpaired scatter plot to file."""
    from crisprscreens.core.qc_plots import plot_paired_vs_unpaired_scatter

    if isinstance(paired_results, (str, Path)):
        paired_results = read_dataframe(paired_results)
    if isinstance(unpaired_results, (str, Path)):
        unpaired_results = read_dataframe(unpaired_results)

    fig = plot_paired_vs_unpaired_scatter(
        paired_results=paired_results,
        unpaired_results=unpaired_results,
        metric=metric,
        gene_col=gene_col,
        highlight_genes=highlight_genes,
        title=title,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        highlight_color=highlight_color,
    )

    save_figure(fig, Path(folder), filename)


def write_downsampling_stability_plot(
    filename: str,
    folder: Union[Path, str],
    stability_df: Union[Path, str, DataFrame],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    color: str = "steelblue",
):
    """Write downsampling stability plot to file."""
    from crisprscreens.core.qc_plots import plot_downsampling_stability

    if isinstance(stability_df, (str, Path)):
        stability_df = read_dataframe(stability_df)

    fig = plot_downsampling_stability(
        stability_df=stability_df,
        title=title,
        figsize=figsize,
        color=color,
    )

    save_figure(fig, Path(folder), filename)


def write_positive_control_ranks_plot(
    filename: str,
    folder: Union[Path, str],
    gene_summary: Union[Path, str, DataFrame],
    positive_controls: Union[Path, str, List[str]],
    rank_col: str = "neg|rank",
    gene_col: str = "id",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    highlight_color: str = "red",
):
    """Write positive control ranks plot to file."""
    from crisprscreens.core.qc_plots import plot_positive_control_ranks

    if isinstance(gene_summary, (str, Path)):
        gene_summary = read_dataframe(gene_summary)

    # Load positive controls if file path
    if isinstance(positive_controls, (str, Path)):
        with open(positive_controls) as f:
            positive_controls = [line.strip() for line in f if line.strip()]

    fig = plot_positive_control_ranks(
        gene_summary=gene_summary,
        positive_controls=positive_controls,
        rank_col=rank_col,
        gene_col=gene_col,
        title=title,
        figsize=figsize,
        highlight_color=highlight_color,
    )

    save_figure(fig, Path(folder), filename)


def write_pairing_decision_summary_plot(
    filename: str,
    folder: Union[Path, str],
    recommendation_dict: Union[Path, str, Dict],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
):
    """Write pairing decision summary plot to file."""
    from crisprscreens.core.qc_plots import plot_pairing_decision_summary
    import json

    if isinstance(recommendation_dict, (str, Path)):
        with open(recommendation_dict) as f:
            recommendation_dict = json.load(f)

    fig = plot_pairing_decision_summary(
        recommendation_dict=recommendation_dict,
        title=title,
        figsize=figsize,
    )

    save_figure(fig, Path(folder), filename)


def write_rankings(
    output_file: Union[Path, str],
    gene_ranking_files_dict: Dict[str, Union[Path, str]],
    gene_id_columns: Dict[str, str],
    ranking_columns: Dict[str, str],
    ascending: Dict[str, bool],
) -> Path:
    """
    write_rankings writes gene rankings to disk.

    Parameters
    ----------
    gene_ranking_files_dict : Dict[str, Union[Path, str]]
        Dictionary of names to gene ranking files.
    gene_id_columns : Dict[str, str]
        ID column names for each ranking file.
    ranking_columns : Dict[str, str]
        Ranking column names for each ranking file.

    Returns
    -------
    Path
        Path to the output file.
    """
    df_out = calculate_ranking_metrics(
        gene_ranking_files_dict=gene_ranking_files_dict,
        gene_id_columns=gene_id_columns,
        ranking_columns=ranking_columns,
        ascending=ascending,
    )
    df_out.to_csv(output_file, sep="\t", index=False)
    return Path(output_file)
