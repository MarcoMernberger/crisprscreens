import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from crisprscreens.core.qc import (
    qc_replicate_consistency,
    qc_logfc_distribution,
    qc_controls_neutrality,
)
from crisprscreens.models import ReportConfig, ResultReport


def test_qc_replicate_consistency_skips_baseline():
    # delta_df has only non-baseline columns
    rng = np.random.RandomState(0)
    delta_df = pd.DataFrame(
        {
            "A_r1": rng.normal(size=100),
            "A_r2": rng.normal(size=100),
            "B_r1": rng.normal(size=100),
            "B_r2": rng.normal(size=100),
        }
    )

    conditions_dict = {
        "total": ["Base1"],
        "A": ["A_r1", "A_r2"],
        "B": ["B_r1", "B_r2"],
    }

    res = qc_replicate_consistency(
        delta_df, conditions_dict, baseline_condition="total"
    )
    summary = res["summary"]

    # baseline should be skipped
    assert "total" not in summary["condition"].tolist()
    # A and B should be present
    assert set(summary["condition"]) == {"A", "B"}
    # median_corr should be finite
    assert (
        not summary.loc[summary["condition"] == "A", "median_corr"].isna().all()
    )


def test_qc_logfc_distribution_skips_baseline():
    rng = np.random.RandomState(1)
    delta_df = pd.DataFrame(
        {
            "A_r1": rng.normal(size=50),
            "A_r2": rng.normal(size=50),
            "B_r1": rng.normal(size=50),
            "B_r2": rng.normal(size=50),
        }
    )

    conditions_dict = {
        "baseline": ["BaseX"],
        "A": ["A_r1", "A_r2"],
        "B": ["B_r1", "B_r2"],
    }

    df = qc_logfc_distribution(
        delta_df, conditions_dict, baseline_condition="baseline"
    )
    # baseline should not be included
    assert "baseline" not in df["condition"].tolist()
    assert set(df["condition"]) == {"A", "B"}


def test_qc_controls_neutrality_baseline_explicit(tmp_path):
    # Build a tiny count table
    df = pd.DataFrame(
        {
            "sgRNA": ["g1", "g2", "g3", "ctrl1", "ctrl2"],
            "Gene": ["A", "A", "B", "C", "C"],
            "Base1": [100, 120, 110, 100, 90],
            "A_r1": [200, 220, 210, 100, 95],
            "A_r2": [195, 210, 205, 98, 96],
        }
    )
    sample_cols = ["Base1", "A_r1", "A_r2"]
    conditions_dict = {"total": ["Base1"], "A": ["A_r1", "A_r2"]}
    control_ids = {"ctrl1", "ctrl2"}

    res = qc_controls_neutrality(
        df,
        sample_cols,
        control_ids,
        conditions_dict,
        baseline_cols=["Base1"],
        baseline_condition="total",
    )
    assert "controls_good" in res
    assert "metrics" in res
    # replicate consistency should be present and not raise
    assert "replicate_consistency" in res


def test_result_report_mle_and_rra(tmp_path):
    # MLE gene_summary
    gene_df = pd.DataFrame(
        {
            "Gene": [f"G{i}" for i in range(10)],
            "Time|beta": np.linspace(-2, 2, 10),
            "Time|wald-fdr": np.linspace(0.9, 0.001, 10),
        }
    )

    sgrna_rows = []
    for g in gene_df["Gene"]:
        for i in range(3):
            sgrna_rows.append(
                {
                    "Gene": g,
                    "sgRNA": f"{g}_s{i}",
                    "LFC": np.sign(
                        np.random.RandomState(hash(g) % 2).randn() + 0.1
                    ),
                }
            )
    sgrna_df = pd.DataFrame(sgrna_rows)

    # write files
    gene_path = tmp_path / "gene_mle.tsv"
    sgrna_path = tmp_path / "sgrna.tsv"
    gene_df.to_csv(gene_path, sep="\t", index=False)
    sgrna_df.to_csv(sgrna_path, sep="\t", index=False)

    cfg = ReportConfig(project_name="test_mle", out_dir=tmp_path / "out")
    rr = ResultReport(
        cfg,
        gene_path,
        sgrna_summary_path=sgrna_path,
        metadata_path=tmp_path / "meta.tsv",
        qc_json_path=None,
    )
    # create an empty metadata
    (tmp_path / "meta.tsv").write_text("sample\tcondition\n")

    rr.build()

    # Check outputs
    plots = list(
        (tmp_path / "out" / cfg.assets_dirname / cfg.plots_dirname).glob(
            "*.png"
        )
    )
    assert any("volcano" in p.name for p in plots)
    assert any("waterfall" in p.name for p in plots)

    # RRA case
    rra_df = pd.DataFrame(
        {
            "Gene": [f"R{i}" for i in range(6)],
            "pos|fdr": [0.01, 0.2, 0.5, 0.001, 0.3, 0.8],
            "neg|fdr": [0.5, 0.3, 0.2, 0.6, 0.9, 0.4],
        }
    )
    rra_path = tmp_path / "rra.tsv"
    rra_df.to_csv(rra_path, sep="\t", index=False)

    cfg2 = ReportConfig(project_name="test_rra", out_dir=tmp_path / "out2")
    rr2 = ResultReport(cfg2, rra_path)
    rr2.build()

    plots2 = list(
        (tmp_path / "out2" / cfg2.assets_dirname / cfg2.plots_dirname).glob(
            "*.png"
        )
    )
    assert any("volcano_rra" in p.name for p in plots2)

    # ranklist exports
    rnk = rr.export_ranklist(eff="Time")
    assert rnk.exists()
    rnk2 = rr2.export_ranklist()
    assert rnk2.exists()
