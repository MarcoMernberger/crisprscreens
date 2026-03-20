"""
Quick Integration Guide: MAGeCK Method Comparison

Füge diesen Code zu deinem run.py hinzu um RRA und MLE zu vergleichen.
"""

import pypipegraph2 as ppg
from pathlib import Path
from crisprscreens import mageck_method_comparison_job
from crisprscreens.core.mageck import mageck_test, mageck_mle

# Paths (passe diese an!)
count_table = "results/mageck_count/brunello.count.txt"
control_sgrnas = "cache/input/consolidated/brunello.genes.sgrnas.controls.txt"
incoming_dir = Path("incoming")

# Sample IDs (passe diese an!)
unsorted_ids = ["Total_Rep1", "Total_Rep2", "Total_Rep3"]
sorted_ids = ["Sort_Rep1", "Sort_Rep2", "Sort_Rep3"]

# Output directory
comparison_dir = Path("results/method_comparison")

###############################################################################
# Definiere zu vergleichende Methoden
###############################################################################

methods = {
    # RRA: Original MAGeCK Methode
    "RRA_paired_median": {
        "run_func": mageck_test,
        "params": {
            "paired": True,
            "norm_method": "median",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "id",  # RRA verwendet 'id' als Gene-Spalte
    },
    "RRA_paired_control_norm": {
        "run_func": mageck_test,
        "params": {
            "paired": True,
            "norm_method": "control",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "id",
    },
    # MLE: Besser bei Batch-Effekten
    "MLE_median_norm": {
        "run_func": mageck_mle,
        "params": {
            "design_matrix": str(incoming_dir / "design_matrix.tsv"),
            "norm_method": "median",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "Gene",  # MLE verwendet 'Gene' als Gene-Spalte
    },
    "MLE_batch_median": {
        "run_func": mageck_mle,
        "params": {
            "design_matrix": str(incoming_dir / "design_matrix_batch.tsv"),
            "norm_method": "median",
            "control_sgrnas": str(control_sgrnas),
        },
        "gene_col": "Gene",
    },
}

###############################################################################
# Run Comprehensive Comparison
###############################################################################

# Dependencies: Stelle sicher dass count_job bereits läuft
# dependencies = [count_job]  # uncomment wenn du count_job hast

comparison_job = mageck_method_comparison_job(
    count_table=str(count_table),
    control_ids=unsorted_ids,
    treatment_ids=sorted_ids,
    output_dir=str(comparison_dir),
    control_sgrnas=str(control_sgrnas),
    methods=methods,
    top_n_list=[50, 100, 200],  # Top-N Listen für Overlap-Analyse
    run_leave_one_out=True,  # Replicate consistency
    run_coherence=True,  # sgRNA coherence per gene
    run_control_fp=True,  # Control false-positives
    run_permutation=True,  # Permutation tests
    n_permutations=5,  # Anzahl Permutationen (mehr = genauer)
    dependencies=[],  # Füge [count_job] hinzu wenn vorhanden
)

###############################################################################
# Interpretation nach dem Run
###############################################################################

print(
    """
Nach dem Run:

1. Schau dir an: results/method_comparison/method_comparison_summary.tsv

2. Wichtigste Metriken:
   - mean_spearman: Höher = stabilere Rankings zwischen Replikaten
   - mean_jaccard_top_100: Höher = mehr Überlapp in Top-Hits
   - mean_direction_consistency: Höher = sgRNAs zeigen in gleiche Richtung
   - controls_in_top_100: Niedriger = weniger False-Positives
   - mean_perm_sig_genes: Niedriger = besser kalibriert

3. Wähle die Methode mit den besten Overall-Metriken

4. Dokumentation:
   - Quick Start: code/crisprscreens/docs/method_comparison_quickstart.md
   - Ausführlich: code/crisprscreens/docs/method_comparison_guide.md
   - Beispiel: code/crisprscreens/examples/method_comparison_example.py

5. Typische Interpretation:
   
   Methode A (RRA):
     mean_spearman = 0.75
     mean_jaccard_top_100 = 0.60
     controls_in_top_100 = 5
   
   Methode B (MLE_batch):
     mean_spearman = 0.88
     mean_jaccard_top_100 = 0.79
     controls_in_top_100 = 1
   
   → MLE_batch ist deutlich besser! Verwende diese für finale Analyse.
"""
)
