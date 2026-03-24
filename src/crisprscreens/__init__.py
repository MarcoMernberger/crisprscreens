"""
CRISPR screens – demultiplexing and read processing tools.

This package provides:
- core
- models
- services
- jobs
- api
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crispr-screens")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"


from .core.interactive import (
    plot_effect_size,
    select_top_n,
)
from .core.mageck import mageck_mle, mageck_test
from .core.plots import (
    _save,
    plot_effect_size_with_labels_zoom,
    plot_substitution_frequency,
)
from .jobs.mageck_jobs import (
    combine_comparison_output_job,
    create_combine_gene_info_with_mageck_output_job,
    create_query_control_sgrna_frames_job,
    create_spiked_count_table_job,
    evaluate_spike_in_performance_job,
    mageck_count_job,
    mageck_count_job2,
    mageck_filter_count_job,
    mageck_mle_job,
    mageck_rra_job,
    run_mageck_scatterview_job,
    spike_evaluation_report_job,
    write_count_table_with_MA_CPM_job,
    write_filtered_mageck_comparison_job,
    write_significant_genes_job,
    write_significant_genes_rra_job,
)
from .jobs.method_comparison_jobs import (
    compare_rankings_simple_job,
    control_false_positive_job,
    leave_one_replicate_out_job,
    mageck_method_comparison_job,
    permutation_test_job,
    sgrna_coherence_job,
)
from .jobs.plot_jobs import (
    plot_ranking_metric_heatmaps_job,
    write_venn_job,
    write_volcano_plot_job,
)
from .jobs.qc_jobs import (
    calculate_ranking_metrics_job,
    # comprehensive_qc_job,
    control_qc_job,
    mageck_report_job,
    pairing_qc_job,
    pairing_qc_plots_job,
    standard_qc_job,
)

# from .models import ReportConfig, ResultReport, QCConfig, QCReport

__all__ = [
    "combine_comparison_output_job",
    "write_filtered_mageck_comparison_job",
    "run_mageck_scatterview_job",
    "write_venn_job",
    "mageck_mle_job",
    "mageck_count_job",
    "mageck_rra_job",
    "create_query_control_sgrna_frames_job",
    "create_combine_gene_info_with_mageck_output_job",
    "write_volcano_plot_job",
    "control_qc_job",
    "standard_qc_job",
    "mageck_report_job",
    "plot_substitution_frequency",
    "pairing_qc_job",
    "pairing_qc_plots_job",
    "create_spiked_count_table_job",
    "evaluate_spike_in_performance_job",
    "spike_evaluation_report_job",
    "leave_one_replicate_out_job",
    "sgrna_coherence_job",
    "control_false_positive_job",
    "permutation_test_job",
    "mageck_method_comparison_job",
    "mageck_test",
    "mageck_mle",
    "compare_rankings_simple_job",
    "mageck_filter_count_job",
    "write_count_table_with_MA_CPM_job",
    "write_significant_genes_job",
    "calculate_ranking_metrics_job",
    "plot_ranking_metric_heatmaps_job",
    "plot_effect_size",
    "select_top_n",
    "plot_effect_size_with_labels_zoom",
    "_save",
]
