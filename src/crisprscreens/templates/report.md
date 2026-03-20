# CRISPR Screen Report — <PROJECT> — <DATE>

## 1. Executive Summary
- **Experiment**: <short description>
- **Analysis method**: <RRA / MLE / RRA+MLE>
- **Normalization**: <median / total / stable_set / control>
- **Primary hit criterion**: <e.g., wald-fdr + |beta| or RRA rank>
- **Top finding**: <one-liner>

### Key metrics
| Metric                                  | Value |
| --------------------------------------- | ----: |
| # sgRNAs                                |   ... |
| # genes                                 |   ... |
| Replicate consistency (median Spearman) |   ... |
| RRA–MLE rank correlation (Spearman)     |   ... |
| # genes with wald-fdr < 0.1             |   ... |
| # genes with                            |  beta | > 1 | ... |

## 2. QC Overview (Counts & Samples)
### 2.1 Library stats
- Table: `tables/qc_library_stats.tsv`
- Plot: library sizes, zeros, top1% fraction

### 2.2 Normalization choice
- Compared methods: median vs total vs stable_set (control if available)
- Chosen: <method> because:
  1) <reason>
  2) <reason>
  3) <reason>

Include:
- `plots/size_factors_comparison.png`
- `plots/ma_plots/*.png`
- `plots/replicate_correlation_heatmap.png`
- `plots/pca_samples.png`

## 3. Main Results
### 3.1 Volcano plot
- `plots/volcano_<effect>.png`

### 3.2 Waterfall plot (ranked effects)
- `plots/waterfall_<effect>.png`

### 3.3 Effect-size vs reproducibility (recommended)
- `plots/effect_vs_reproducibility_<effect>.png`

### 3.4 RRA vs MLE concordance (if both available)
- `plots/rra_vs_mle_rank_scatter.png`

## 4. Top Hits
### 4.1 High-confidence hits
- criteria: <your rule>
- Table: `tables/top_hits_<effect>.tsv`

### 4.2 sgRNA-level consistency for selected hits
- For top N genes:
  - `plots/gene_<GENE>_sgRNA_effects.png`

## 5. Multi-factor interpretation (MLE designs only)
### 5.1 Effect decomposition
- `plots/beta_decomposition.png` (Time / Treatment / Dose)

### 5.2 Dose-response (if applicable)
- `plots/low_vs_high_scatter.png`

## 6. Pathway/Signature Enrichment (rank-based)
- `plots/pathway_barplot_<effect>.png`
- `tables/pathways_<effect>.tsv`

## 7. Appendix
- Commands, versions, parameters
- Input files checksum
