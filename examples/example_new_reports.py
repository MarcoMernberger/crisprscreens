"""
Example: Using the new QCReport and ResultReport classes

This script demonstrates how to use the refactored report classes
for comprehensive CRISPR screen analysis.
"""

from crisprscreens.models import QCReport, QCConfig, ResultReport, ReportConfig


def run_comprehensive_qc_analysis(
    count_table: str,
    control_sgrnas: str,
    baseline_condition: str = "Total",
    output_dir: str = "qc_results",
):
    """
    Run comprehensive QC analysis with the new QCReport class.

    This replaces the old control_qc_job and standard_qc_job approach.
    """
    print("Running Comprehensive QC Analysis")
    print("=" * 60)

    # Configure QC analysis
    config = QCConfig(
        project_name="CRISPR Screen QC",
        out_dir=output_dir,
        baseline_condition=baseline_condition,
        norm_methods=[
            "total",
            "median",
            "stable_set",
        ],  # Test multiple normalizations
        sgrna_col="sgRNA",
        gene_col="Gene",
    )

    # Create QC report
    qc = QCReport(
        config=config,
        count_table_path=count_table,
        control_sgrnas_path=control_sgrnas,
        metadata_path=None,  # Will be inferred from column names
    )

    # Run analysis
    qc.build(
        run_control_qc=True,  # Validate control sgRNAs
        run_library_qc=True,  # Compare normalization methods
        generate_pdf=True,  # Generate PDF report
    )

    print(f"\nQC analysis complete! Results saved to: {output_dir}")
    print("Files generated:")
    print("  - qc_summary.md (Markdown report)")
    print("  - qc_summary.json (Machine-readable summary)")
    print("  - qc_report.pdf (PDF report)")
    print("  - qc_assets/plots/*.png (QC plots)")
    print("  - qc_assets/tables/*.tsv (QC metrics)")


def generate_mageck_result_report(
    gene_summary: str,
    sgrna_summary: str = None,
    qc_json: str = None,
    output_dir: str = "report_results",
):
    """
    Generate comprehensive result report from MAGeCK analysis.

    This uses the new ResultReport class with PDF export capability.
    """
    print("\nGenerating MAGeCK Result Report")
    print("=" * 60)

    # Configure report
    config = ReportConfig(
        project_name="CRISPR Screen Results",
        out_dir=output_dir,
        fdr_threshold=0.1,
        effect_threshold=1.0,
        top_n_labels=15,
        top_n_hits_table=50,
    )

    # Create result report
    report = ResultReport(
        config=config,
        gene_summary_path=gene_summary,
        sgrna_summary_path=sgrna_summary,
        qc_json_path=qc_json,
    )

    # Generate reports
    report.build(generate_pdf=True)

    # Export ranked gene list for GSEA (optional)
    if report.effects:
        for effect in report.effects:
            ranklist_path = report.export_ranklist(eff=effect)
            print(f"  Exported ranklist for {effect}: {ranklist_path}")

    print(f"\nResult report complete! Results saved to: {output_dir}")
    print("Files generated:")
    print("  - report.md (Markdown report)")
    print("  - report.html (HTML report)")
    print("  - result_report.pdf (PDF report)")
    print("  - report_assets/plots/*.png (Volcano, waterfall plots)")
    print("  - report_assets/tables/*.tsv (Top hits)")


def main():
    """
    Example workflow: QC → MAGeCK analysis → Result report
    """
    print("CRISPR Screen Analysis Workflow")
    print("=" * 60)

    # Step 1: QC Analysis
    run_comprehensive_qc_analysis(
        count_table="data/brunello.count.tsv",
        control_sgrnas="data/control_sgRNAs.txt",
        baseline_condition="Total",
        output_dir="results/qc",
    )

    # Step 2: Run MAGeCK (not shown - would use mageck_mle_job or mageck_rra_job)
    # ...

    # Step 3: Generate Result Report
    generate_mageck_result_report(
        gene_summary="results/mageck/gene_summary.txt",
        sgrna_summary="results/mageck/sgrna_summary.txt",
        qc_json="results/qc/qc_summary.json",
        output_dir="results/report",
    )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    main()

    # Or use individually:
    # run_comprehensive_qc_analysis("counts.tsv", "controls.txt")
    # generate_mageck_result_report("gene_summary.txt")
