# """
# QC Report Module

# Comprehensive quality control reporting for CRISPR screens.
# Unifies control sgRNA QC and library QC analysis with recommendations.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, Dict, List, Union, Set
# from datetime import datetime

# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Import from core QC functions
# from ..core.qc import (
#     load_control_sgrnas,
#     read_counts,
#     parse_metadata_from_columns,
#     read_metadata,
#     compute_library_stats,
#     calculate_cpm,
#     calculate_delta_logfc,
#     compute_size_factors_total,
#     compute_size_factors_median_ratio,
#     compute_size_factors_stable_set,
#     compute_size_factors_control,
#     select_stable_guides,
#     apply_size_factors,
#     compare_size_factors,
#     qc_logfc_distribution,
#     qc_replicate_consistency,
#     qc_controls_neutrality,
#     choose_best_normalization,
#     recommend_analysis_method,
# )

# try:
#     import markdown as md
# except Exception:
#     md = None

# try:
#     from reportlab.lib.pagesizes import A4
#     from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
#     from reportlab.lib.units import inch
#     from reportlab.lib import colors
#     from reportlab.platypus import (
#         SimpleDocTemplate,
#         Paragraph,
#         Spacer,
#         Image,
#         PageBreak,
#     )

#     REPORTLAB_AVAILABLE = True
# except ImportError:
#     REPORTLAB_AVAILABLE = False


# @dataclass
# class QCConfig:
#     """
#     Configuration for QC report generation.

#     Holds all user-configurable parameters that control how the QC analysis
#     is performed and where outputs are written. 
#     Attributes
#     ----------
#     project_name : str
#         Name for the project (appears in reports).
#     out_dir : Union[str, Path]
#         Root directory where QC outputs will be placed. Defaults to "qc_out".
#     assets_dirname : str
#         Directory name inside "out_dir" where plots and tables are stored.
#     plots_dirname : str
#         Directory name (inside "assets_dirname") used to keep generated plots.
#     tables_dirname : str
#         Directory name (inside "assets_dirname") used to keep generated tables.
#     sgrna_col : str
#         Column name in the count table that contains sgRNA identifiers.
#     gene_col : str
#         Column name in the count table that contains gene identifiers.
#     delimiter : str
#         Delimiter used to split sample column names into condition and
#         replicate (e.g. "Total_Rep1" using "_").
#     baseline_condition : str
#         Condition name used as reference/baseline for log fold-change
#         calculations (matching is case-insensitive where possible).
#     pseudocount : float
#         Small constant added to counts before log-transformation to avoid
#         log(0) (default 1.0).
#     norm_methods : List[str]
#         List of normalization methods to evaluate. If None, defaults to
#         ["total", "median", "stable_set"]. Possible methods include
#         "total", "median", "stable_set", and "control" (requires
#         controls to be provided).
#     control_median_threshold : float
#         Threshold for the allowed median shift in control sgRNAs.
#     control_iqr_threshold : float
#         Threshold for acceptable IQR of control sgRNA log fold-changes.
#     control_tail_threshold : float
#         Threshold for the fraction of control guides in extreme tails.
#     replicate_corr_threshold : float
#         Minimum Pearson correlation expected between replicates.

#     """

#     project_name: str
#     out_dir: Union[str, Path] = "qc_out"
#     assets_dirname: str = "qc_assets"
#     plots_dirname: str = "plots"
#     tables_dirname: str = "tables"

#     # Column names
#     sgrna_col: str = "sgRNA"
#     gene_col: str = "Gene"
#     delimiter: str = "_"

#     # Analysis parameters
#     baseline_condition: str = "total"
#     pseudocount: float = 1.0

#     # Normalization methods to test
#     norm_methods: List[str] = None

#     # QC thresholds
#     control_median_threshold: float = 0.5
#     control_iqr_threshold: float = 1.0
#     control_tail_threshold: float = 0.15
#     replicate_corr_threshold: float = 0.8

#     def __post_init__(self):
#         if self.norm_methods is None:
#             self.norm_methods = ["total", "median", "stable_set"]


# class QCReport:
#     """
#     Comprehensive QC report generator for CRISPR screens.

#     This class orchestrates the end-to-end QC pipeline for a CRISPR screen
#     experiment. It is responsible for loading the count table and metadata,
#     optionally validating control (non-targeting) sgRNAs, computing and
#     comparing normalization strategies, assessing replicate consistency,
#     generating actionable recommendations, and writing human- and
#     machine-readable reports (Markdown, JSON, optionally PDF).

#     Important attributes are described inline in the initializer; the class
#     stores intermediate and final results so callers can inspect data at
#     each stage (for example `size_factors_dict`, `qc_by_norm`, and
#     `control_qc_results`).
#     """

#     def __init__(
#         self,
#         config: QCConfig,
#         count_table_path: Union[str, Path],
#         control_sgrnas_path: Optional[Union[str, Path]] = None,
#         metadata_path: Optional[Union[str, Path]] = None,
#     ):
#         """
#         Initialize QCReport object.
#         Create output folders and initalize.

#         Parameter
#         ---------
#         config : QCConfig
#             Config with parameters and paths.
#         count_table_path : Union[str, Path]
#             Path to count table (MAGeCK).
#         control_sgrnas_path : Optional[Union[str, Path]]
#             Pfad to control sgRNA ids (one per line).
#         metadata_path : Optional[Union[str, Path]]
#             Path to metadata table (sample, condition, replicate).

#         """
#         self.cfg = config
#         self.count_table_path = Path(count_table_path)
#         self.control_sgrnas_path = (
#             Path(control_sgrnas_path) if control_sgrnas_path else None
#         )
#         self.metadata_path = Path(metadata_path) if metadata_path else None

#         # Setup output directories
#         self.out_dir = Path(self.cfg.out_dir)
#         self.assets_dir = self.out_dir / self.cfg.assets_dirname
#         self.plots_dir = self.assets_dir / self.cfg.plots_dirname
#         self.tables_dir = self.assets_dir / self.cfg.tables_dirname

#         self.out_dir.mkdir(parents=True, exist_ok=True)
#         self.plots_dir.mkdir(parents=True, exist_ok=True)
#         self.tables_dir.mkdir(parents=True, exist_ok=True)

#         # Full count table loaded from `count_table_path` (rows: guides, cols: samples)
#         self.count_df: Optional[pd.DataFrame] = None
#         # List of sample column names extracted from the count table
#         self.sample_cols: Optional[List[str]] = None
#         # Mapping: condition_name -> list of sample column names for that condition
#         self.conditions_dict: Optional[Dict[str, List[str]]] = None
#         # List of sample columns that belong to the chosen baseline condition
#         self.baseline_cols: Optional[List[str]] = None
#         # Set of control sgRNA IDs loaded from `control_sgrnas_path` (if provided)
#         self.control_ids: Optional[Set[str]] = None
#         # Optional metadata DataFrame with columns: sample, condition, replicate
#         self.metadata_df: Optional[pd.DataFrame] = None

#         # Per-sample summary statistics (library sizes, zero counts, etc.)
#         self.library_stats: Optional[pd.DataFrame] = None
#         # Result dict returned by `qc_controls_neutrality` (if controls used)
#         self.control_qc_results: Optional[Dict] = None
#         # Computed size factors for each normalization method (method -> pd.Series)
#         self.size_factors_dict: Dict[str, pd.Series] = {}
#         # Per-normalization QC outputs (logfc distributions, replicate consistency)
#         self.qc_by_norm: Dict[str, Dict] = {}
#         # Best normalization decision (populated by `_make_recommendations`)
#         self.best_norm: Optional[Dict] = None
#         # Analysis recommendation (RRA vs MLE and supporting reasons)
#         self.recommendation: Optional[Dict] = None

#             # Summary for reports
#         self._summary: Optional[Dict] = None

#     def build(
#         self,
#         run_control_qc: bool = True,
#         run_library_qc: bool = True,
#         generate_pdf: bool = False,
#     ) -> None:
#         """
#         Run qc-pipeline and create report.

#         Parameter
#         ---------
#         run_control_qc : bool
#             Whether to run control sgRNA QC.
#         run_library_qc : bool
#             Whether to run library-wide QC (normalization comparison, etc.).
#         generate_pdf : bool
#             Whether to generate a PDF report (requires reportlab).
#         """
#         print("=" * 60)
#         print(f"CRISPR Screen Quality Control: {self.cfg.project_name}")
#         print("=" * 60)

#         # Step 1: Load data
#         self._load_data()

#         # Step 2: Control QC (if requested and controls available)
#         if run_control_qc and self.control_sgrnas_path is not None:
#             self._run_control_qc()

#         # Step 3: Library QC
#         if run_library_qc:
#             self._run_library_qc()

#         # Step 4: Generate recommendations
#         self._make_recommendations()

#         # Step 5: Create summary
#         self._make_summary()

#         # Step 6: Generate reports
#         self._write_markdown_report()

#         if generate_pdf:
#             self.generate_pdf_report()

#         print("\n" + "=" * 60)
#         print("QC Analysis Complete!")
#         print(f"Reports saved to: {self.out_dir}")
#         print("=" * 60)

#     def _load_data(self) -> None:
#         """
#         Load count table and metadata.

#         Load count table via `read_counts`, parse metadata either
#         from `metadata_path` or from the column names, and write basic
#         library statistics as TSV in the tables directory.
#         """
#         print("\n[1/6] Loading count data...")
#         self.count_df, self.sample_cols = read_counts(
#             self.count_table_path, self.cfg.sgrna_col, self.cfg.gene_col
#         )
#         print(
#             f"  Loaded {len(self.count_df)} sgRNAs, {len(self.sample_cols)} samples"  # noqa: E501
#         )

#         # Load metadata
#         print("\n[2/6] Parsing metadata...")
#         if self.metadata_path is not None:
#             self.metadata_df = read_metadata(self.metadata_path)
#             self.conditions_dict = {}
#             for _, row in self.metadata_df.iterrows():
#                 condition = row["condition"]
#                 sample = row["sample"]
#                 if condition not in self.conditions_dict:
#                     self.conditions_dict[condition] = []
#                 self.conditions_dict[condition].append(sample)
#         else:
#             self.conditions_dict, self.metadata_df = (
#                 parse_metadata_from_columns(
#                     self.sample_cols, self.cfg.delimiter
#                 )
#             )

#         print(
#             f"  Found {len(self.conditions_dict)} conditions: {list(self.conditions_dict.keys())}"  # noqa: E501
#         )

#         # Validate baseline (allow case-insensitive match to common naming differences)
#         if self.cfg.baseline_condition not in self.conditions_dict:
#             # try a case-insensitive match
#             matches = [k for k in self.conditions_dict.keys() if k.lower() == self.cfg.baseline_condition.lower()]
#             if matches:
#                 # adopt the canonical condition name from parsed metadata
#                 self.cfg.baseline_condition = matches[0]
#             else:
#                 raise ValueError(
#                     f"Baseline condition '{self.cfg.baseline_condition}' not found. "  # noqa: E501
#                     f"Available: {list(self.conditions_dict.keys())}"
#                 )
#         self.baseline_cols = self.conditions_dict[self.cfg.baseline_condition]

#         # Load controls if available
#         if self.control_sgrnas_path is not None:
#             self.control_ids = load_control_sgrnas(self.control_sgrnas_path)
#             print(f"  Loaded {len(self.control_ids)} control sgRNAs")

#         # Compute library stats
#         self.library_stats = compute_library_stats(
#             self.count_df, self.sample_cols
#         )
#         lib_stats_file = self.tables_dir / "library_stats.tsv"
#         self.library_stats.to_csv(lib_stats_file, sep="\t", index=False)
#         print(f"  Saved library stats to {lib_stats_file.name}")


#     def _run_control_qc(self) -> None:
#         """
#         Checks the neutrality of control sgRNAs.

#         Calls `qc_controls_neutrality`, evaluates the result, and saves
#         the results (JSON) as well as diagnostic plots (if metrics are available).
#         If controls show systematic shifts, the method is
#         logged and control-based normalization is not recommended.
#         """
#         print("\n[3/6] Running control sgRNA QC...")

#         self.control_qc_results = qc_controls_neutrality(
#             self.count_df,
#             self.sample_cols,
#             self.control_ids,
#             self.conditions_dict,
#             self.baseline_cols,
#             self.cfg.sgrna_col,
#             self.cfg.pseudocount,
#             baseline_condition=self.cfg.baseline_condition,
#         )
#         print(self.control_qc_results)

#         if self.control_qc_results["controls_good"]:
#             print("  ✓ Controls are neutral - suitable for normalization")
#             if "control" not in self.cfg.norm_methods:
#                 self.cfg.norm_methods.append("control")
#         else:
#             print("  ✗ Controls show systematic biases:")
#             for reason in self.control_qc_results["reasons"]:
#                 print(f"    - {reason}")
#             print("  → Control-based normalization NOT recommended")

#         # Save control QC results
#         control_qc_file = self.tables_dir / "control_neutrality_qc.json"
#         # Be defensive: qc_controls_neutrality may return limited keys when no controls
#         omc = self.control_qc_results.get("overall_median_corr")
#         metrics = self.control_qc_results.get("metrics", {})
#         json_data = {
#             "controls_good": self.control_qc_results.get("controls_good", False),
#             "metrics": metrics,
#             "overall_median_corr": (
#                 float(omc) if omc is not None and not (isinstance(omc, float) and np.isnan(omc)) else None
#             ),
#             "reasons": self.control_qc_results.get("reasons", []),
#         }
#         with open(control_qc_file, "w") as f:
#             json.dump(json_data, f, indent=2)
#         print(f"  Saved control QC to {control_qc_file.name}")

#         # Generate control QC plots
#         self._plot_control_qc()

#     def _plot_control_qc(self) -> None:
#         """
#         Create diagnostic plots for control sgRNAs.

#         Generates bar charts for median shift and IQR per condition. If
#         no metrics are available (e.g., no controls in the count table),
#         the plots are skipped.
#         """
#         if self.control_qc_results is None:
#             return

#         metrics = self.control_qc_results.get("metrics", {})

#         if not metrics:
#             print("  No control metrics available, skipping control plots.")
#             return

#         # Plot 1: Median Δc per condition
#         conditions = [k for k in metrics.keys() if k != "baseline"]
#         medians = [metrics[cond].get("median_delta", np.nan) for cond in conditions]

#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.bar(range(len(conditions)), medians, color="steelblue", alpha=0.7)
#         ax.axhline(
#             self.cfg.control_median_threshold,
#             color="red",
#             linestyle="--",
#             label=f"Threshold: ±{self.cfg.control_median_threshold}",
#         )
#         ax.axhline(
#             -self.cfg.control_median_threshold, color="red", linestyle="--"
#         )
#         ax.axhline(0, color="black", linewidth=0.5)
#         ax.set_xticks(range(len(conditions)))
#         ax.set_xticklabels(conditions, rotation=45, ha="right")
#         ax.set_ylabel("Median Δc (log2FC)")
#         ax.set_title("Control sgRNA Median Shift per Condition")
#         ax.legend()
#         fig.tight_layout()
#         fig.savefig(self.plots_dir / "control_median_shift.png", dpi=200)
#         plt.close(fig)

#         # Plot 2: IQR per condition
#         iqrs = [metrics[cond]["iqr_delta"] for cond in conditions]

#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.bar(range(len(conditions)), iqrs, color="coral", alpha=0.7)
#         ax.axhline(
#             self.cfg.control_iqr_threshold,
#             color="red",
#             linestyle="--",
#             label=f"Threshold: {self.cfg.control_iqr_threshold}",
#         )
#         ax.set_xticks(range(len(conditions)))
#         ax.set_xticklabels(conditions, rotation=45, ha="right")
#         ax.set_ylabel("IQR of Δc")
#         ax.set_title("Control sgRNA Variability (IQR) per Condition")
#         ax.legend()
#         fig.tight_layout()
#         fig.savefig(self.plots_dir / "control_iqr.png", dpi=200)
#         plt.close(fig)

#     def _run_library_qc(self) -> None:
#         """
#         Runs library-wide QC across different normalization methods.

#         FFor each method listed in `cfg.norm_methods`, size factors are computed,
#         compared, counts normalized, and subsequently log2FC distributions and replicate consistency evaluated.
#         The results are stored in `self.qc_by_norm`.
#         """
#         print("\n[4/6] Running library-wide QC...")

#         # Compute size factors for all methods
#         print(f"  Testing normalization methods: {self.cfg.norm_methods}")

#         for method in self.cfg.norm_methods:
#             if method == "total":
#                 sf = compute_size_factors_total(self.count_df, self.sample_cols)
#             elif method == "median":
#                 sf = compute_size_factors_median_ratio(
#                     self.count_df, self.sample_cols
#                 )
#             elif method == "stable_set":
#                 simple_cpm = calculate_cpm(self.count_df, self.sample_cols)
#                 logcpm_simple = np.log2(simple_cpm[self.sample_cols] + 1)
#                 stable_guides = select_stable_guides(
#                     logcpm_simple, self.sample_cols
#                 )
#                 print(
#                     f"    Selected {len(stable_guides)} stable guides for '{method}'"  # noqa: E501
#                 )
#                 sf = compute_size_factors_stable_set(
#                     self.count_df, self.sample_cols, stable_guides
#                 )
#             elif method == "control":
#                 if self.control_ids is None:
#                     print(f"    Skipping '{method}' - no controls available")
#                     continue
#                 sf = compute_size_factors_control(
#                     self.count_df,
#                     self.sample_cols,
#                     self.control_ids,
#                     self.cfg.sgrna_col,
#                 )
#             else:
#                 print(f"    Unknown method: {method}")
#                 continue

#             self.size_factors_dict[method] = sf
#             print(f"    {method}: median SF = {sf.median():.3f}")

#         # Save size factors
#         sf_df = pd.DataFrame(self.size_factors_dict)
#         sf_file = self.tables_dir / "size_factors.tsv"
#         sf_df.to_csv(sf_file, sep="\t")

#         # Compare size factors
#         sf_comparison = compare_size_factors(self.size_factors_dict)
#         sf_comp_file = self.tables_dir / "size_factors_comparison.tsv"
#         sf_comparison.to_csv(sf_comp_file, sep="\t", index=False)

#         # Run QC for each normalization
#         for method in self.size_factors_dict.keys():
#             norm_counts = apply_size_factors(
#                 self.count_df, self.sample_cols, self.size_factors_dict[method]
#             )
#             norm_cpm = calculate_cpm(norm_counts, self.sample_cols)

#             non_baseline_cols = [
#                 c for c in self.sample_cols if c not in self.baseline_cols
#             ]
#             delta_df = calculate_delta_logfc(
#                 norm_cpm,
#                 non_baseline_cols,
#                 self.baseline_cols,
#                 self.cfg.pseudocount,
#             )

#             logfc_dist = qc_logfc_distribution(
#                 delta_df,
#                 self.conditions_dict,
#                 baseline_condition=self.cfg.baseline_condition,
#             )
#             rep_consistency = qc_replicate_consistency(
#                 delta_df,
#                 self.conditions_dict,
#                 baseline_condition=self.cfg.baseline_condition,
#             )

#             self.qc_by_norm[method] = {
#                 "logfc_dist": logfc_dist,
#                 "replicate_consistency": rep_consistency,
#                 "delta": delta_df,
#             }

#         print(f"  Analyzed {len(self.qc_by_norm)} normalization methods")

#         # Generate library QC plots
#         self._plot_library_qc()

#     def _plot_library_qc(self) -> None:
#         """
#         Create comparison plots for the tested normalizations.

#         This includes histograms of replicate correlations and the median
#         of log2FC distributions per method.
#         """
#         # Plot 1: Replicate correlations comparison
#         fig, axes = plt.subplots(
#             1, len(self.qc_by_norm), figsize=(5 * len(self.qc_by_norm), 4)
#         )
#         if len(self.qc_by_norm) == 1:
#             axes = [axes]

#         for idx, (method, qc) in enumerate(self.qc_by_norm.items()):
#             rep_cons = qc["replicate_consistency"]
#             all_corrs = []
#             for cond in rep_cons.keys():
#                 if cond != self.cfg.baseline_condition:
#                     corr_mat = rep_cons[cond]["correlation_matrix"]
#                     # Extract upper triangle
#                     n = corr_mat.shape[0]
#                     for i in range(n):
#                         for j in range(i + 1, n):
#                             all_corrs.append(corr_mat.iloc[i, j])

#             axes[idx].hist(
#                 all_corrs,
#                 bins=20,
#                 color="steelblue",
#                 alpha=0.7,
#                 edgecolor="black",
#             )
#             axes[idx].axvline(
#                 self.cfg.replicate_corr_threshold,
#                 color="red",
#                 linestyle="--",
#                 label=f"Threshold: {self.cfg.replicate_corr_threshold}",
#             )
#             axes[idx].set_xlabel("Pearson Correlation")
#             axes[idx].set_ylabel("Count")
#             axes[idx].set_title(
#                 f"{method}\n(median: {np.median(all_corrs):.3f})"
#             )
#             axes[idx].legend()

#         fig.suptitle("Replicate Consistency Across Normalization Methods")
#         fig.tight_layout()
#         fig.savefig(
#             self.plots_dir / "replicate_correlations_comparison.png", dpi=200
#         )
#         plt.close(fig)

#         # Plot 2: LogFC distribution comparison
#         fig, axes = plt.subplots(
#             1, len(self.qc_by_norm), figsize=(5 * len(self.qc_by_norm), 4)
#         )
#         if len(self.qc_by_norm) == 1:
#             axes = [axes]

#         for idx, (method, qc) in enumerate(self.qc_by_norm.items()):
#             logfc_dist = qc["logfc_dist"]
#             medians = [
#                 logfc_dist[cond]["median"]
#                 for cond in logfc_dist.keys()
#                 if cond != self.cfg.baseline_condition
#             ]

#             axes[idx].hist(
#                 medians, bins=15, color="coral", alpha=0.7, edgecolor="black"
#             )
#             axes[idx].axvline(0, color="black", linewidth=0.5)
#             axes[idx].set_xlabel("Median log2FC")
#             axes[idx].set_ylabel("Count")
#             axes[idx].set_title(
#                 f"{method}\n(overall: {np.mean(np.abs(medians)):.3f})"
#             )

#         fig.suptitle("LogFC Distribution Across Normalization Methods")
#         fig.tight_layout()
#         fig.savefig(
#             self.plots_dir / "logfc_distribution_comparison.png", dpi=200
#         )
#         plt.close(fig)

#     def _make_recommendations(self) -> None:
#         """
#         Generates recommendations for normalization and analysis methods.

#         Chooses the best normalization via `choose_best_normalization` and
#         then recommends an analysis method (RRA vs. MLE) based on replicate consistency and log2FC distributions.
#         """
#         print("\n[5/6] Generating recommendations...")

#         # Choose best normalization
#         if self.qc_by_norm:
#             self.best_norm = choose_best_normalization(
#                 self.qc_by_norm,
#                 self.conditions_dict,
#                 baseline_condition=self.cfg.baseline_condition,
#             )
#             print(f"  Best normalization: {self.best_norm['best_method']}")
#             print(f"  Reason: {self.best_norm['reason']}")

#             # Save recommendation
#             best_norm_file = self.tables_dir / "best_normalization.json"
#             with open(best_norm_file, "w") as f:
#                 json.dump(self.best_norm, f, indent=2)

#         # Recommend analysis method (RRA vs MLE)
#         if self.qc_by_norm and self.best_norm:
#             best_method = self.best_norm["best_method"]
#             best_qc = self.qc_by_norm[best_method]

#             self.recommendation = recommend_analysis_method(
#                 best_qc["replicate_consistency"],
#                 best_qc["logfc_dist"],
#                 self.conditions_dict,
#                 baseline_condition=self.cfg.baseline_condition,
#             )
#             print(
#                 f"  Recommended analysis: {self.recommendation['preferred_method']}"  # noqa: E501
#             )
#             for reason in self.recommendation["reasons"]:
#                 print(f"    - {reason}")

#             # Save recommendation
#             rec_file = self.tables_dir / "analysis_recommendation.json"
#             with open(rec_file, "w") as f:
#                 json.dump(self.recommendation, f, indent=2)

#     def _make_summary(self) -> Dict:
#         """
#         Creates a summary dictionary of all QC results.

#         Saves the summary as `qc_summary.json` and returns the
#         dictionary. Used by the reporting functions.
#         """
#         print("\n[6/6] Creating summary...")

#         self._summary = {
#             "project_name": self.cfg.project_name,
#             "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "n_sgrnas": len(self.count_df),
#             "n_samples": len(self.sample_cols),
#             "n_conditions": len(self.conditions_dict),
#             "baseline_condition": self.cfg.baseline_condition,
#             "normalization_methods_tested": list(self.size_factors_dict.keys()),
#         }

#         # Control QC summary
#         if self.control_qc_results:
#             self._summary["control_qc"] = {
#                 "n_controls": len(self.control_ids),
#                 "controls_neutral": self.control_qc_results["controls_good"],
#                 "reasons": self.control_qc_results["reasons"],
#             }

#         # Best normalization
#         if self.best_norm:
#             self._summary["best_normalization"] = self.best_norm["best_method"]
#             self._summary["normalization_reason"] = self.best_norm["reason"]

#         # Analysis recommendation
#         if self.recommendation:
#             self._summary["recommended_analysis"] = self.recommendation[
#                 "preferred_method"
#             ]
#             self._summary["analysis_reasons"] = self.recommendation["reasons"]

#         # Save summary
#         summary_file = self.out_dir / "qc_summary.json"
#         with open(summary_file, "w") as f:
#             json.dump(self._summary, f, indent=2)

#         return self._summary

#     def _write_markdown_report(self) -> None:
#         """
#         Writes the Markdown report `qc_summary.md` (and optionally HTML).

#         The report includes Executive Summary, Control QC, Library QC,
#         recommendations, and a listing of generated tables and JSO files.
#         """
#         report_text = "# CRISPR Screen Quality Control Report\n\n"
#         report_text += f"**Project:** {self.cfg.project_name}\n\n"
#         report_text += f"**Generated:** {self._summary['generated_at']}\n\n"
#         report_text += "---\n\n"

#         # Executive Summary
#         report_text += "## Executive Summary\n\n"
#         report_text += f"- **sgRNAs:** {self._summary['n_sgrnas']:,}\n"
#         report_text += f"- **Samples:** {self._summary['n_samples']}\n"
#         report_text += f"- **Conditions:** {self._summary['n_conditions']}\n"
#         report_text += (
#             f"- **Baseline:** {self._summary['baseline_condition']}\n"
#         )

#         if "best_normalization" in self._summary:
#             report_text += f"- **Best Normalization:** {self._summary['best_normalization']}\n"  # noqa: E501
#         if "recommended_analysis" in self._summary:
#             report_text += f"- **Recommended Analysis:** {self._summary['recommended_analysis']}\n"  # noqa: E501

#         report_text += "\n---\n\n"

#         # Control QC Section
#         if "control_qc" in self._summary:
#             report_text += "## Control sgRNA QC\n\n"
#             cqc = self._summary["control_qc"]
#             report_text += f"**Number of controls:** {cqc['n_controls']}\n\n"

#             if cqc["controls_neutral"]:
#                 report_text += "✓ **Controls are neutral** - suitable for normalization\n\n"  # noqa: E501
#             else:
#                 report_text += "✗ **Controls show systematic biases:**\n\n"
#                 for reason in cqc["reasons"]:
#                     report_text += f"- {reason}\n"
#                 report_text += "\n"

#             # Add control plots
#             if (self.plots_dir / "control_median_shift.png").exists():
#                 report_text += "### Control Median Shift\n\n"
#                 report_text += f"![Control Median Shift]({self.cfg.assets_dirname}/{self.cfg.plots_dirname}/control_median_shift.png)\n\n"  # noqa: E501

#             if (self.plots_dir / "control_iqr.png").exists():
#                 report_text += "### Control Variability\n\n"
#                 report_text += f"![Control IQR]({self.cfg.assets_dirname}/{self.cfg.plots_dirname}/control_iqr.png)\n\n"  # noqa: E501

#         # Library QC Section
#         if self.qc_by_norm:
#             report_text += "## Library-Wide QC\n\n"
#             report_text += f"**Normalization methods tested:** {', '.join(self._summary['normalization_methods_tested'])}\n\n"  # noqa: E501

#             if (
#                 self.plots_dir / "replicate_correlations_comparison.png"
#             ).exists():
#                 report_text += "### Replicate Consistency\n\n"
#                 report_text += f"![Replicate Correlations]({self.cfg.assets_dirname}/{self.cfg.plots_dirname}/replicate_correlations_comparison.png)\n\n"  # noqa: E501

#             if (self.plots_dir / "logfc_distribution_comparison.png").exists():
#                 report_text += "### LogFC Distribution\n\n"
#                 report_text += f"![LogFC Distribution]({self.cfg.assets_dirname}/{self.cfg.plots_dirname}/logfc_distribution_comparison.png)\n\n"  # noqa: E501

#         # Recommendations
#         report_text += "## Recommendations\n\n"

#         if "normalization_reason" in self._summary:
#             report_text += (
#                 f"### Normalization: {self._summary['best_normalization']}\n\n"
#             )
#             report_text += f"{self._summary['normalization_reason']}\n\n"

#         if "analysis_reasons" in self._summary:
#             report_text += f"### Analysis Method: {self._summary['recommended_analysis']}\n\n"  # noqa: E501
#             for reason in self._summary["analysis_reasons"]:
#                 report_text += f"- {reason}\n"
#             report_text += "\n"

#         # Data Tables
#         report_text += "## Data Tables\n\n"
#         for table_file in sorted(self.tables_dir.glob("*.tsv")):
#             rel_path = table_file.relative_to(self.out_dir)
#             report_text += f"- `{rel_path}`\n"
#         for table_file in sorted(self.tables_dir.glob("*.json")):
#             rel_path = table_file.relative_to(self.out_dir)
#             report_text += f"- `{rel_path}`\n"

#         # Write markdown
#         report_md = self.out_dir / "qc_summary.md"
#         report_md.write_text(report_text, encoding="utf-8")
#         print(f"  Saved markdown report to {report_md.name}")

#         # Try HTML rendering
#         if md is not None:
#             out_html = self.out_dir / "qc_summary.html"
#             out_html.write_text(md.markdown(report_text), encoding="utf-8")
#             print(f"  Saved HTML report to {out_html.name}")

#     def generate_pdf_report(self) -> Optional[Path]:
#         """
#         Generates (optional) a PDF report with reportlab.

#         Returns the path to the generated PDF file or `None` if
#         reportlab is not installed or an error occurs.
#         """
#         if not REPORTLAB_AVAILABLE:
#             print(
#                 "Warning: reportlab not available. Cannot generate PDF report."
#             )
#             print("Install with: pip install reportlab")
#             return None

#         pdf_path = self.out_dir / "qc_report.pdf"

#         try:
#             self._build_pdf(pdf_path)
#             return pdf_path
#         except Exception as e:
#             print(f"Error generating PDF report: {e}")
#             return None

#     def _build_pdf(self, pdf_path: Path) -> None:
#         """
#         Builds the PDF using reportlab.

#         Uses the internal summary (`self._summary`) and the
#         available plots/tables to create a multi-page PDF.
#         """
#         doc = SimpleDocTemplate(
#             str(pdf_path),
#             pagesize=A4,
#             topMargin=0.75 * inch,
#             bottomMargin=0.75 * inch,
#             leftMargin=0.75 * inch,
#             rightMargin=0.75 * inch,
#         )

#         story = []
#         styles = getSampleStyleSheet()

#         # Custom styles
#         title_style = ParagraphStyle(
#             "CustomTitle",
#             parent=styles["Heading1"],
#             fontSize=24,
#             textColor=colors.HexColor("#2c3e50"),
#             spaceAfter=30,
#             alignment=1,
#         )

#         subtitle_style = ParagraphStyle(
#             "Subtitle",
#             parent=styles["Heading2"],
#             fontSize=16,
#             textColor=colors.HexColor("#34495e"),
#             spaceAfter=20,
#             alignment=1,
#         )

#         # Title page
#         story.append(Spacer(1, 1.5 * inch))
#         story.append(Paragraph(self.cfg.project_name, title_style))
#         story.append(Paragraph("CRISPR Screen Quality Control", subtitle_style))
#         story.append(Spacer(1, 0.3 * inch))

#         if self._summary:
#             story.append(
#                 Paragraph(
#                     f"Generated: {self._summary.get('generated_at', 'N/A')}",
#                     styles["Normal"],
#                 )
#             )

#         story.append(PageBreak())

#         # Executive Summary
#         story.append(Paragraph("Executive Summary", styles["Heading1"]))
#         story.append(Spacer(1, 0.2 * inch))

#         if self._summary:
#             summary_items = [
#                 f"<b>sgRNAs:</b> {self._summary.get('n_sgrnas', 'N/A'):,}",
#                 f"<b>Samples:</b> {self._summary.get('n_samples', 'N/A')}",
#                 f"<b>Conditions:</b> {self._summary.get('n_conditions', 'N/A')}",  # noqa: E501
#                 f"<b>Baseline:</b> {self._summary.get('baseline_condition', 'N/A')}",  # noqa: E501
#             ]

#             if "best_normalization" in self._summary:
#                 summary_items.append(
#                     f"<b>Best Normalization:</b> {self._summary['best_normalization']}"  # noqa: E501
#                 )
#             if "recommended_analysis" in self._summary:
#                 summary_items.append(
#                     f"<b>Recommended Analysis:</b> {self._summary['recommended_analysis']}"  # noqa: E501
#                 )

#             for item in summary_items:
#                 story.append(Paragraph(f"• {item}", styles["BodyText"]))
#                 story.append(Spacer(1, 0.1 * inch))

#         story.append(PageBreak())

#         # Control QC Section
#         if "control_qc" in self._summary:
#             story.append(Paragraph("Control sgRNA QC", styles["Heading1"]))
#             story.append(Spacer(1, 0.2 * inch))

#             cqc = self._summary["control_qc"]
#             story.append(
#                 Paragraph(
#                     f"<b>Number of controls:</b> {cqc['n_controls']}",
#                     styles["BodyText"],
#                 )
#             )
#             story.append(Spacer(1, 0.15 * inch))

#             if cqc["controls_neutral"]:
#                 story.append(
#                     Paragraph(
#                         "<b>✓ Controls are neutral</b> - suitable for normalization",  # noqa: E501
#                         styles["BodyText"],
#                     )
#                 )
#             else:
#                 story.append(
#                     Paragraph(
#                         "<b>✗ Controls show systematic biases:</b>",
#                         styles["BodyText"],
#                     )
#                 )
#                 for reason in cqc["reasons"]:
#                     story.append(Paragraph(f"• {reason}", styles["BodyText"]))

#             story.append(Spacer(1, 0.3 * inch))

#             # Add control plots
#             for plot_name in ["control_median_shift.png", "control_iqr.png"]:
#                 plot_path = self.plots_dir / plot_name
#                 if plot_path.exists():
#                     try:
#                         img = Image(
#                             str(plot_path), width=5 * inch, height=3 * inch
#                         )
#                         img.hAlign = "CENTER"
#                         story.append(img)
#                         story.append(Spacer(1, 0.3 * inch))
#                     except Exception:
#                         pass

#             story.append(PageBreak())

#         # Library QC Section
#         if self.qc_by_norm:
#             story.append(Paragraph("Library-Wide QC", styles["Heading1"]))
#             story.append(Spacer(1, 0.2 * inch))

#             story.append(
#                 Paragraph(
#                     f"<b>Normalization methods tested:</b> {', '.join(self._summary['normalization_methods_tested'])}",  # noqa: E501
#                     styles["BodyText"],
#                 )
#             )
#             story.append(Spacer(1, 0.3 * inch))

#             # Add library plots
#             for plot_name in [
#                 "replicate_correlations_comparison.png",
#                 "logfc_distribution_comparison.png",
#             ]:
#                 plot_path = self.plots_dir / plot_name
#                 if plot_path.exists():
#                     plot_title = (
#                         plot_name.replace("_", " ").replace(".png", "").title()
#                     )
#                     story.append(Paragraph(plot_title, styles["Heading2"]))
#                     story.append(Spacer(1, 0.1 * inch))
#                     try:
#                         img = Image(
#                             str(plot_path), width=6 * inch, height=3 * inch
#                         )
#                         img.hAlign = "CENTER"
#                         story.append(img)
#                         story.append(Spacer(1, 0.3 * inch))
#                     except Exception:
#                         pass

#             story.append(PageBreak())

#         # Recommendations
#         story.append(Paragraph("Recommendations", styles["Heading1"]))
#         story.append(Spacer(1, 0.2 * inch))

#         if "normalization_reason" in self._summary:
#             story.append(
#                 Paragraph(
#                     f"<b>Normalization:</b> {self._summary['best_normalization']}",  # noqa: E501
#                     styles["Heading2"],
#                 )
#             )
#             story.append(
#                 Paragraph(
#                     self._summary["normalization_reason"], styles["BodyText"]
#                 )
#             )
#             story.append(Spacer(1, 0.2 * inch))

#         if "analysis_reasons" in self._summary:
#             story.append(
#                 Paragraph(
#                     f"<b>Analysis Method:</b> {self._summary['recommended_analysis']}",  # noqa: E501
#                     styles["Heading2"],
#                 )
#             )
#             for reason in self._summary["analysis_reasons"]:
#                 story.append(Paragraph(f"• {reason}", styles["BodyText"]))
#                 story.append(Spacer(1, 0.05 * inch))

#         # Build PDF
#         doc.build(story)
#         print(f"  Saved PDF report to {pdf_path.name}")
