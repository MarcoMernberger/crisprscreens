"""
MAGeCK Result Reporting Module

Comprehensive reporting for MAGeCK RRA and MLE results with publication-ready
plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

from .plots import (
    volcano_plot,
    plot_effect_size_vs_reproducibility,
    plot_rank_stability,
    plot_direction_consistency,
    plot_replicate_effect_heatmap,
    plot_effect_decomposition,
    plot_contrast,
    plot_pathway_enrichment_summary,
    plot_gene_set_score_distribution,
    plot_beta_vs_standard_error,
    plot_wald_z_distribution,
    plot_qq,
)


def generate_mageck_report(
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
        Report mode: "minimal" (5 key plots) or "full" (all 11 plots + metrics)
    effect_cols : list, optional
        For MLE: list of beta column names for effect decomposition
        E.g., ['beta|Time', 'beta|Met', 'beta|High']
    gene_col : str
        Column name for gene identifiers
    fdr_threshold : float
        FDR threshold for significance (default 0.25)
    top_n : int
        Number of top genes to highlight in plots
    gene_sets : dict, optional
        Dictionary mapping gene set names to lists of genes
        For gene-set score distribution analysis
    pathway_enrichment_path : str or Path, optional
        Path to pathway enrichment results (CSV/TSV)
    **kwargs : dict
        Additional parameters passed to plotting functions

    Returns
    -------
    dict
        Summary metrics and paths to generated plots

    Notes
    -----
    Minimal mode includes:
    1. Volcano plot
    2. Effect-size vs reproducibility
    3. Rank stability (if both RRA and MLE available)
    4. Direction consistency
    5. Pathway enrichment (if available)

    Full mode adds:
    6. Replicate effect heatmap
    7. Effect decomposition (for multi-factor MLE)
    8. Contrast plot (for 2-condition comparisons)
    9. Gene-set score distributions
    10. Beta vs standard error
    11. Wald-Z distribution
    12. QQ-plot of p-values
    Plus executive summary metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "output_dir": str(output_dir),
        "readout": readout,
        "plots": {},
        "metrics": {},
    }

    # Load data
    gene_summary = (
        pd.read_csv(gene_summary_path, sep="\t") if gene_summary_path else None
    )
    sgrna_summary = (
        pd.read_csv(sgrna_summary_path, sep="\t")
        if sgrna_summary_path
        else None
    )
    rra_summary = (
        pd.read_csv(rra_summary_path, sep="\t") if rra_summary_path else None
    )
    mle_summary = (
        pd.read_csv(mle_summary_path, sep="\t") if mle_summary_path else None
    )
    pathway_enrichment = (
        pd.read_csv(pathway_enrichment_path, sep="\t")
        if pathway_enrichment_path
        else None
    )

    # Use appropriate summary for main analysis
    main_summary = (
        mle_summary
        if mle_summary is not None
        else (rra_summary if rra_summary is not None else gene_summary)
    )

    if main_summary is None:
        raise ValueError(
            "At least one of gene_summary_path, rra_summary_path, or mle_summary_path must be provided"  # NOQA
        )

    # Detect column names automatically
    beta_col = _detect_column(main_summary, ["beta", "lfc", "score"], "beta")
    fdr_col = _detect_column(main_summary, ["fdr", "q-value", "q_value"], "fdr")
    pvalue_col = _detect_column(
        main_summary, ["p-value", "pvalue", "p_value"], None
    )

    # =========================================================================
    # MINIMAL MODE PLOTS (always generated)
    # =========================================================================

    # 1. Volcano Plot
    print("Generating volcano plot...")
    if beta_col and fdr_col:
        fig, ax = volcano_plot(
            main_summary,
            beta_col=beta_col,
            fdr_col=fdr_col,
            fdr_threshold=fdr_threshold,
            title=f"MAGeCK Volcano Plot (FDR < {fdr_threshold})",
        )
        volcano_path = output_dir / "01_volcano_plot.png"
        fig.savefig(volcano_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        results["plots"]["volcano"] = str(volcano_path)

    # 2. Effect-size vs Reproducibility
    print("Generating effect-size vs reproducibility plot...")
    fig, ax, metrics = plot_effect_size_vs_reproducibility(
        gene_summary_df=main_summary,
        effect_col=beta_col,
        fdr_col=fdr_col,
        sgrna_summary_df=sgrna_summary,
        gene_col=gene_col,
        fdr_threshold=fdr_threshold,
        title="Effect Size vs. sgRNA Consistency",
    )
    effect_repro_path = output_dir / "02_effect_vs_reproducibility.png"
    fig.savefig(effect_repro_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    results["plots"]["effect_reproducibility"] = str(effect_repro_path)
    results["metrics"]["effect_reproducibility"] = metrics

    # 3. Rank Stability (if both RRA and MLE available)
    if rra_summary is not None and mle_summary is not None:
        print("Generating rank stability plot...")
        rra_rank_col = _detect_column(
            rra_summary, ["pos|rank", "rank", "pos_rank"], "pos|rank"
        )
        mle_rank_col = _detect_column(
            mle_summary, ["Gene|rank", "rank", "gene_rank"], "Gene|rank"
        )
        rra_fdr_col = _detect_column(
            rra_summary, ["pos|fdr", "fdr", "pos_fdr"], "pos|fdr"
        )
        mle_fdr_col = _detect_column(
            mle_summary, ["Gene|fdr", "fdr", "gene_fdr"], "Gene|fdr"
        )

        fig, ax, metrics = plot_rank_stability(
            rra_summary_df=rra_summary,
            mle_summary_df=mle_summary,
            gene_col=gene_col,
            rra_rank_col=rra_rank_col,
            mle_rank_col=mle_rank_col,
            fdr_col_rra=rra_fdr_col,
            fdr_col_mle=mle_fdr_col,
            fdr_threshold=fdr_threshold,
            top_n=top_n,
            title="Method Robustness: RRA vs. MLE Rankings",
        )
        rank_stability_path = output_dir / "03_rank_stability.png"
        fig.savefig(rank_stability_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        results["plots"]["rank_stability"] = str(rank_stability_path)
        results["metrics"]["rank_stability"] = metrics

    # 4. Direction Consistency (sgRNA level)
    if sgrna_summary is not None:
        print("Generating direction consistency plot...")
        lfc_col = _detect_column(sgrna_summary, ["LFC", "lfc", "log2fc"], "LFC")

        fig, ax, metrics = plot_direction_consistency(
            sgrna_summary_df=sgrna_summary,
            gene_col=gene_col,
            lfc_col=lfc_col,
            genes=None,  # Auto-select top genes
            min_sgrnas=3,
            title="sgRNA Direction Consistency per Gene",
        )
        direction_path = output_dir / "04_direction_consistency.png"
        fig.savefig(direction_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        results["plots"]["direction_consistency"] = str(direction_path)
        results["metrics"]["direction_consistency"] = metrics

    # 5. Pathway Enrichment (if available)
    if pathway_enrichment is not None:
        print("Generating pathway enrichment summary...")
        pathway_col = _detect_column(
            pathway_enrichment, ["pathway", "term", "name"], "pathway"
        )
        pvalue_col_pathway = _detect_column(
            pathway_enrichment, ["pvalue", "p-value", "p_value"], "pvalue"
        )
        ratio_col = _detect_column(
            pathway_enrichment,
            ["gene_ratio", "ratio", "GeneRatio"],
            "gene_ratio",
        )

        fig, ax = plot_pathway_enrichment_summary(
            enrichment_df=pathway_enrichment,
            pathway_col=pathway_col,
            pvalue_col=pvalue_col_pathway,
            gene_ratio_col=ratio_col,
            top_n=15,
            title="Top Enriched Pathways",
        )
        pathway_path = output_dir / "05_pathway_enrichment.png"
        fig.savefig(pathway_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        results["plots"]["pathway_enrichment"] = str(pathway_path)

    # =========================================================================
    # FULL MODE PLOTS (additional analyses)
    # =========================================================================

    if readout == "full":

        # 6. Replicate Effect Heatmap
        if effect_cols and len(effect_cols) > 1 and mle_summary is not None:
            print("Generating replicate effect heatmap...")
            # Extract effect columns
            effect_data = mle_summary[[gene_col] + effect_cols].copy()

            # Select top genes by total effect
            effect_data["total_effect"] = (
                effect_data[effect_cols].abs().sum(axis=1)
            )
            top_genes = effect_data.nlargest(30, "total_effect")

            heatmap_data = top_genes.set_index(gene_col)[effect_cols]

            fig, ax, metrics = plot_replicate_effect_heatmap(
                data_matrix=heatmap_data,
                cmap="RdBu_r",
                center=0,
                figsize=(10, 12),
                title="Gene Effects Across Conditions",
            )
            heatmap_path = output_dir / "06_replicate_heatmap.png"
            fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            results["plots"]["replicate_heatmap"] = str(heatmap_path)
            results["metrics"]["replicate_heatmap"] = metrics

        # 7. Effect Decomposition (for multi-factor MLE)
        if effect_cols and len(effect_cols) > 1 and mle_summary is not None:
            print("Generating effect decomposition plot...")
            fig, ax, metrics = plot_effect_decomposition(
                mle_summary_df=mle_summary,
                effect_cols=effect_cols,
                gene_col=gene_col,
                top_n=20,
                stacked=False,
                title="Effect Decomposition Across Factors",
            )
            decomp_path = output_dir / "07_effect_decomposition.png"
            fig.savefig(decomp_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            results["plots"]["effect_decomposition"] = str(decomp_path)
            results["metrics"]["effect_decomposition"] = metrics

        # 8. Contrast Plot (for 2-condition comparisons)
        if effect_cols and len(effect_cols) == 2 and mle_summary is not None:
            print("Generating contrast plot...")
            fig, ax, metrics = plot_contrast(
                mle_summary_df=mle_summary,
                effect_col_x=effect_cols[0],
                effect_col_y=effect_cols[1],
                gene_col=gene_col,
                fdr_col=fdr_col,
                fdr_threshold=fdr_threshold,
                title=f"Contrast: {effect_cols[0]} vs {effect_cols[1]}",
            )
            contrast_path = output_dir / "08_contrast_plot.png"
            fig.savefig(contrast_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            results["plots"]["contrast"] = str(contrast_path)
            results["metrics"]["contrast"] = metrics

        # 9. Gene-set Score Distribution
        if gene_sets:
            print("Generating gene-set score distributions...")
            fig, ax, metrics = plot_gene_set_score_distribution(
                gene_summary_df=main_summary,
                gene_sets=gene_sets,
                effect_col=beta_col,
                gene_col=gene_col,
                title="Effect Size Distribution by Gene Set",
            )
            geneset_path = output_dir / "09_gene_set_distributions.png"
            fig.savefig(geneset_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            results["plots"]["gene_set_distribution"] = str(geneset_path)
            results["metrics"]["gene_set_distribution"] = metrics

        # 10. Beta vs Standard Error (MLE only)
        if mle_summary is not None:
            print("Generating beta vs standard error plot...")
            se_col = _detect_column(
                mle_summary, ["beta|z", "stderr", "se", "std_err"], None
            )
            if se_col is None:
                # Calculate SE from Z-score if available
                z_col = _detect_column(
                    mle_summary, ["beta|z", "wald-z", "z"], None
                )
                if z_col and beta_col:
                    mle_summary["calculated_se"] = mle_summary[
                        beta_col
                    ].abs() / (mle_summary[z_col].abs() + 1e-10)
                    se_col = "calculated_se"

            if se_col:
                fig, ax, metrics = plot_beta_vs_standard_error(
                    mle_summary_df=mle_summary,
                    beta_col=beta_col,
                    se_col=se_col,
                    fdr_col=fdr_col,
                    fdr_threshold=fdr_threshold,
                    gene_col=gene_col,
                    title="Effect Size vs. Estimation Uncertainty",
                )
                beta_se_path = output_dir / "10_beta_vs_se.png"
                fig.savefig(beta_se_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                results["plots"]["beta_vs_se"] = str(beta_se_path)
                results["metrics"]["beta_vs_se"] = metrics

        # 11. Wald-Z Distribution (MLE only)
        if mle_summary is not None:
            print("Generating Wald-Z distribution plot...")
            z_col = _detect_column(
                mle_summary, ["beta|z", "wald-z", "z", "wald_z"], None
            )
            if z_col:
                fig, ax, metrics = plot_wald_z_distribution(
                    mle_summary_df=mle_summary,
                    z_col=z_col,
                    title="Wald Z-statistic Distribution (Model QC)",
                )
                waldz_path = output_dir / "11_wald_z_distribution.png"
                fig.savefig(waldz_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                results["plots"]["wald_z"] = str(waldz_path)
                results["metrics"]["wald_z"] = metrics

        # 12. QQ-Plot
        if pvalue_col:
            print("Generating QQ-plot...")
            pvalues = main_summary[pvalue_col].dropna().values
            fig, ax, metrics = plot_qq(
                pvalues=pvalues, title="QQ-Plot: P-value Calibration"
            )
            qq_path = output_dir / "12_qq_plot.png"
            fig.savefig(qq_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            results["plots"]["qq_plot"] = str(qq_path)
            results["metrics"]["qq_plot"] = metrics

        # =====================================================================
        # EXECUTIVE SUMMARY METRICS
        # =====================================================================
        print("Computing executive summary metrics...")
        summary_metrics = _compute_executive_summary(
            main_summary=main_summary,
            sgrna_summary=sgrna_summary,
            rra_summary=rra_summary,
            mle_summary=mle_summary,
            beta_col=beta_col,
            fdr_col=fdr_col,
            gene_col=gene_col,
            fdr_threshold=fdr_threshold,
        )
        results["executive_summary"] = summary_metrics

    # Save results to JSON
    results_path = output_dir / "mageck_report_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✓ MAGeCK report generated successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Mode: {readout}")
    print(f"  Plots generated: {len(results['plots'])}")

    return results


def _detect_column(df, candidates, default):
    """Helper to detect column names from candidate list."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return default


def _compute_executive_summary(
    main_summary,
    sgrna_summary,
    rra_summary,
    mle_summary,
    beta_col,
    fdr_col,
    gene_col,
    fdr_threshold,
):
    """Compute executive summary metrics."""
    summary = {}

    # Total genes analyzed
    summary["n_genes_total"] = len(main_summary)

    # Significant hits
    if fdr_col:
        summary["n_genes_fdr_0.1"] = (main_summary[fdr_col] < 0.1).sum()
        summary["n_genes_fdr_0.25"] = (
            main_summary[fdr_col] < fdr_threshold
        ).sum()

    # Effect size statistics
    if beta_col:
        summary["median_abs_beta"] = main_summary[beta_col].abs().median()
        summary["n_genes_large_effect"] = (
            main_summary[beta_col].abs() > 1
        ).sum()

    # sgRNA statistics
    if sgrna_summary is not None:
        summary["n_sgrnas_total"] = len(sgrna_summary)
        sgrna_per_gene = sgrna_summary.groupby(gene_col).size()
        summary["median_sgrnas_per_gene"] = sgrna_per_gene.median()

        # sgRNA consistency
        lfc_col = _detect_column(sgrna_summary, ["LFC", "lfc", "log2fc"], "LFC")
        if lfc_col:
            consistency = sgrna_summary.groupby(gene_col).apply(
                lambda x: (x[lfc_col] > 0).sum() / len(x) if len(x) > 0 else 0
            )
            # Genes with >70% sgRNAs in same direction
            summary["n_genes_consistent_direction"] = (
                (consistency > 0.7) | (consistency < 0.3)
            ).sum()

    # RRA-MLE agreement
    if rra_summary is not None and mle_summary is not None:
        rra_rank_col = _detect_column(
            rra_summary, ["pos|rank", "rank"], "pos|rank"
        )
        mle_rank_col = _detect_column(
            mle_summary, ["Gene|rank", "rank"], "Gene|rank"
        )

        merged = rra_summary[[gene_col, rra_rank_col]].merge(
            mle_summary[[gene_col, mle_rank_col]], on=gene_col, how="inner"
        )

        from scipy.stats import spearmanr

        if len(merged) > 2:
            summary["rra_mle_rank_correlation"] = spearmanr(
                merged[rra_rank_col], merged[mle_rank_col]
            )[0]

    return summary
