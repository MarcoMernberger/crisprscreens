from __future__ import annotations
from pandas import DataFrame
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class SpikeSpec:
    """Spike-in specification for a group contrast."""

    name_prefix: str  # e.g., "SPIKE_POS"
    n_genes: int  # number of spike genes
    guides_per_gene: int = 4  # typical Brunello
    log2_effect: float = 2.0  # +2 => 4x in sorted; -2 => 4x depletion in sorted
    baseline_mean: float = (
        500.0  # expected mean count in total (before scaling)
    )
    dispersion: float = 0.05  # NB dispersion (alpha); Var = mu + alpha*mu^2


def _nbinom_rvs(
    mu: np.ndarray, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample from Negative Binomial with mean mu and dispersion alpha.
    Var = mu + alpha * mu^2
    """
    # Convert to numpy NB parameterization: n, p
    # n = 1/alpha, p = n/(n+mu)
    n = 1.0 / max(alpha, 1e-12)
    p = n / (n + np.maximum(mu, 0.0))
    return rng.negative_binomial(n=n, p=p, size=mu.shape)


def add_spikeins_to_count_table(
    count_df: pd.DataFrame,
    *,
    sample_to_group: Dict[str, str],
    sample_cols: Optional[List[str]] = None,
    group_contrast: Tuple[str, str] = ("sorted", "total"),
    replicate_of: Optional[Dict[str, str]] = None,
    spike_specs: List[SpikeSpec],
    neutral_genes: int = 0,
    neutral_baseline_mean: float = 500.0,
    neutral_dispersion: float = 0.05,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Add synthetic spike-in genes/sgRNAs to a MAGeCK count table.

    - group_contrast = (treated_group, baseline_group), e.g. ("sorted", "total")
    - For each SpikeSpec:
        log2_effect applies to treated_group vs baseline_group
    - replicate_of is optional; if provided, adds replicate-specific noise to mimic reality.
    """
    rng = np.random.default_rng(random_seed)

    if sample_cols is None:
        sample_cols = [
            c for c in count_df.columns if c not in ("sgRNA", "Gene")
        ]

    treated, baseline = group_contrast
    for s in sample_cols:
        if s not in sample_to_group:
            raise KeyError(f"Missing sample_to_group mapping for sample: {s}")

    # Rough library-size scaling from existing data (keeps spikes in a realistic range)
    lib_sizes = count_df[sample_cols].sum(axis=0).astype(float)
    lib_scale = lib_sizes / np.median(lib_sizes)

    # Optional replicate-specific multiplicative noise (mimic batch/rep effects)
    rep_noise = {}
    if replicate_of is not None:
        reps = sorted(set(replicate_of[s] for s in sample_cols))
        for r in reps:
            # log-normal noise ~ +/- 15% by default
            rep_noise[r] = float(np.exp(rng.normal(0.0, 0.15)))

    rows = []

    def _make_gene_rows(
        prefix: str,
        gene_idx: int,
        log2_eff: float,
        base_mean: float,
        alpha: float,
    ):
        gene_name = f"{prefix}_G{gene_idx:03d}"
        for g in range(
            1, 1 + 1
        ):  # gene loop placeholder, keep structure simple
            for guide_i in range(1, 1 + spec.guides_per_gene):
                sgrna_id = f"{gene_name}_{guide_i}"
                mu = np.zeros(len(sample_cols), dtype=float)

                for i, s in enumerate(sample_cols):
                    grp = sample_to_group[s]
                    mu_i = base_mean

                    # Apply group effect
                    if grp == treated:
                        mu_i *= 2.0**log2_eff
                    elif grp == baseline:
                        mu_i *= 1.0
                    else:
                        # For other groups, keep baseline (you can extend this if needed)
                        mu_i *= 1.0

                    # Scale by library size
                    mu_i *= float(lib_scale[s])

                    # Optional replicate noise
                    if replicate_of is not None:
                        mu_i *= rep_noise[replicate_of[s]]

                    mu[i] = mu_i

                counts = _nbinom_rvs(mu=mu, alpha=alpha, rng=rng).astype(int)

                row = {"sgRNA": sgrna_id, "Gene": gene_name}
                row.update({s: int(c) for s, c in zip(sample_cols, counts)})
                rows.append(row)

    # Add specified spikes
    for spec in spike_specs:
        for gi in range(1, spec.n_genes + 1):
            _make_gene_rows(
                spec.name_prefix,
                gi,
                spec.log2_effect,
                spec.baseline_mean,
                spec.dispersion,
            )

    # Add neutral spikes (optional)
    if neutral_genes > 0:
        neutral_spec = SpikeSpec(
            name_prefix="SPIKE_NEUTRAL",
            n_genes=neutral_genes,
            guides_per_gene=4,
            log2_effect=0.0,
            baseline_mean=neutral_baseline_mean,
            dispersion=neutral_dispersion,
        )
        for gi in range(1, neutral_spec.n_genes + 1):
            # Reuse the same helper but with 0 effect
            spec = neutral_spec
            _make_gene_rows(
                spec.name_prefix, gi, 0.0, spec.baseline_mean, spec.dispersion
            )

    spike_df = pd.DataFrame(rows)
    out = pd.concat([count_df, spike_df], ignore_index=True)
    return out


def create_spiked_count_table(
    count_df: DataFrame,
    replicate_of: Dict[str, str],
    sample_to_group: Dict[str, str],
    sample_cols: List[str] = None,
    group_contrast: Tuple[str] = ("sorted", "total"),
    n_genes: int = 20,
    log_effect: float = 2.0,
    baseline_mean: float = 300.0,
    dispersion: float = 0.08,
) -> DataFrame:
    spike_specs = [
        SpikeSpec(
            name_prefix="SPIKE_POS",
            n_genes=n_genes,
            log2_effect=log_effect,
            baseline_mean=baseline_mean,
            dispersion=dispersion,
        ),
        SpikeSpec(
            name_prefix="SPIKE_NEG",
            n_genes=n_genes,
            log2_effect=-log_effect,
            baseline_mean=baseline_mean,
            dispersion=dispersion,
        ),
    ]
    count_spike = add_spikeins_to_count_table(
        count_df,
        sample_cols=sample_cols,
        sample_to_group=sample_to_group,
        group_contrast=group_contrast,
        replicate_of=replicate_of,
        spike_specs=spike_specs,
        neutral_genes=20,
        random_seed=42,
    )
    return count_spike
