"""
Result Report Module

Comprehensive reporting for MAGeCK CRISPR screen results (MLE/RRA).
Generates plots, tables, markdown and PDF reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Union
from datetime import datetime

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from core module (adjust relative path since we moved to models/)
import sys
from pathlib import Path as _Path

_parent = _Path(__file__).parent.parent / "core"
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from ..core.plots import plot_effect_size_vs_reproducibility

try:
    import markdown as md
except Exception:
    md = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        PageBreak,
        Table,
        TableStyle,
        KeepTogether,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


@dataclass
class ReportConfig:
    project_name: str
    out_dir: Union[str, Path] = "report_out"
    assets_dirname: str = "report_assets"
    plots_dirname: str = "plots"
    tables_dirname: str = "tables"
    baseline_condition: str = "total"

    # Primary thresholds used for tables/labels
    fdr_threshold: float = 0.1
    effect_threshold: float = 1.0

    # Labeling
    top_n_labels: int = 15
    top_n_hits_table: int = 50

    # Method mode
    analysis_method: str = "auto"  # "auto", "MLE", "RRA", "RRA+MLE"
    normalization_method: str = (
        "auto"  # "auto", "median", "total", "stable_set", "control"
    )


class ResultReport:
    """
    A report generator for MAGeCK CRISPR screen outputs.

    Inputs:
      - gene_summary: MLE gene_summary.tsv or RRA gene_summary.tsv
      - optional: sgrna_summary, count table, metadata, qc_summary.json

    Outputs:
      - plots (PNG)
      - tables (TSV)
      - report.md
      - report.pdf (optional, requires reportlab)
    """

    def __init__(
        self,
        config: ReportConfig,
        gene_summary_path: Union[str, Path],
        sgrna_summary_path: Optional[Union[str, Path]] = None,
        count_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        qc_json_path: Optional[Union[str, Path]] = None,
    ):
        self.cfg = config
        self.gene_summary_path = Path(gene_summary_path)
        self.sgrna_summary_path = (
            Path(sgrna_summary_path) if sgrna_summary_path else None
        )
        self.count_path = Path(count_path) if count_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.qc_json_path = Path(qc_json_path) if qc_json_path else None

        self.out_dir = Path(self.cfg.out_dir)
        self.assets_dir = self.out_dir / self.cfg.assets_dirname
        self.plots_dir = self.assets_dir / self.cfg.plots_dirname
        self.tables_dir = self.assets_dir / self.cfg.tables_dirname

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.gene_df: pd.DataFrame = self._read_table(self.gene_summary_path)
        self.sgrna_df: Optional[pd.DataFrame] = (
            self._read_table(self.sgrna_summary_path)
            if self.sgrna_summary_path
            else None
        )
        self.count_df: Optional[pd.DataFrame] = (
            self._read_table(self.count_path) if self.count_path else None
        )
        self.meta_df: Optional[pd.DataFrame] = (
            self._read_table(self.metadata_path) if self.metadata_path else None
        )
        self.qc: Optional[dict] = (
            self._read_json(self.qc_json_path) if self.qc_json_path else None
        )

        # Detect analysis type and available effects
        self.analysis_type = self._detect_analysis_type(self.gene_df)
        self.effects = self._detect_effects(self.gene_df, self.analysis_type)

        # Store summary for PDF generation
        self._summary: Optional[dict] = None

    # -----------------------
    # IO helpers
    # -----------------------
    def _read_table(self, path: Optional[Path]) -> pd.DataFrame:
        if path is None:
            raise ValueError("Path is None")
        if not path.exists():
            raise FileNotFoundError(path)
        # TSV by default; works for MAGeCK outputs
        return pd.read_csv(path, sep="\t")

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text())

    # -----------------------
    # Detection
    # -----------------------
    def _detect_analysis_type(self, df: pd.DataFrame) -> str:
        cols = set(df.columns)
        # Heuristic: MLE gene_summary contains "|beta" and "|wald-fdr"
        if any("|beta" in c for c in cols) and any(
            "wald-fdr" in c for c in cols
        ):
            return "MLE"
        # RRA gene_summary contains "neg|fdr" / "pos|fdr" or similar
        if any("neg|fdr" in c for c in cols) or any(
            "pos|fdr" in c for c in cols
        ):
            return "RRA"
        return "UNKNOWN"

    def _detect_effects(
        self, df: pd.DataFrame, analysis_type: str
    ) -> List[str]:
        cols = df.columns.tolist()
        effects: List[str] = []

        if analysis_type == "MLE":
            # Extract effect prefixes like "Time21|beta" -> effect "Time21"
            for c in cols:
                if c.endswith("|beta"):
                    effects.append(c.split("|")[0])
            return sorted(set(effects))

        if analysis_type == "RRA":
            # RRA has two directions; treat as effects "pos" and "neg"
            return ["pos", "neg"]

        return []

    # -----------------------
    # Main API
    # -----------------------
    def build(self, generate_pdf: bool = False) -> None:
        """
        Build full report: QC summary + plots + tables + report.md (+ optional PDF).

        Parameters
        ----------
        generate_pdf : bool
            If True, generate PDF report (requires reportlab)
        """
        summary = self._make_summary()
        self._summary = summary  # Store for PDF generation
        self._write_summary_tables(summary)
        self._make_plots()
        self._write_markdown_report(summary)

        if generate_pdf:
            self.generate_pdf_report()

    # -----------------------
    # Summary
    # -----------------------
    def _make_summary(self) -> dict:
        """
        Create a summary dict used in report.md.
        """
        summary: Dict[str, object] = {
            "project_name": self.cfg.project_name,
            "analysis_type_detected": self.analysis_type,
            "effects": self.effects,
            "normalization_method": (
                self.qc.get("best_norm_method")
                if self.qc
                else self.cfg.normalization_method
            ),
            "preferred_method": (
                self.qc.get("preferred_method")
                if self.qc
                else self.cfg.analysis_method
            ),
            "n_genes": (
                int(self.gene_df["Gene"].nunique())
                if "Gene" in self.gene_df.columns
                else int(self.gene_df.shape[0])
            ),
            "n_rows": int(self.gene_df.shape[0]),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add per-effect counts of hits if MLE
        if self.analysis_type == "MLE":
            effect_stats = {}
            for eff in self.effects:
                beta_col = f"{eff}|beta"
                fdr_col = (
                    f"{eff}|wald-fdr"
                    if f"{eff}|wald-fdr" in self.gene_df.columns
                    else f"{eff}|fdr"
                )
                if (
                    beta_col in self.gene_df.columns
                    and fdr_col in self.gene_df.columns
                ):
                    df = self.gene_df.copy()
                    df[beta_col] = pd.to_numeric(df[beta_col], errors="coerce")
                    df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
                    n_sig = int(
                        (
                            (df[fdr_col] <= self.cfg.fdr_threshold)
                            & (df[beta_col].abs() >= self.cfg.effect_threshold)
                        ).sum()
                    )
                    effect_stats[eff] = {
                        "n_sig": n_sig,
                        "beta_median_abs": float(
                            np.nanmedian(np.abs(df[beta_col].to_numpy()))
                        ),
                        "fdr_min": float(np.nanmin(df[fdr_col].to_numpy())),
                    }
            summary["effect_stats"] = effect_stats

        return summary

    def _write_summary_tables(self, summary: dict) -> None:
        # Write a minimal top-hits table per effect for MLE
        if self.analysis_type == "MLE":
            for eff in self.effects:
                beta_col = f"{eff}|beta"
                fdr_col = (
                    f"{eff}|wald-fdr"
                    if f"{eff}|wald-fdr" in self.gene_df.columns
                    else f"{eff}|fdr"
                )
                if (
                    beta_col not in self.gene_df.columns
                    or fdr_col not in self.gene_df.columns
                ):
                    continue

                df = self.gene_df[["Gene", "sgRNA", beta_col, fdr_col]].copy()
                df[beta_col] = pd.to_numeric(df[beta_col], errors="coerce")
                df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")

                df = df.sort_values(by=fdr_col, ascending=True).head(
                    self.cfg.top_n_hits_table
                )
                out = self.tables_dir / f"top_hits_{eff}.tsv"
                df.to_csv(out, sep="\t", index=False)

    # -----------------------
    # Plots
    # -----------------------
    def _make_plots(self) -> None:
        if self.analysis_type == "MLE":
            for eff in self.effects:
                self._plot_volcano_mle(eff)
                self._plot_waterfall_mle(eff)
                # Optional: effect vs reproducibility if sgrna_df + meta/count exists
                if self.sgrna_df is not None and self.meta_df is not None:
                    self._plot_effect_vs_reproducibility(eff)

            # Optional: decomposition plot if multiple effects present
            if len(self.effects) >= 2:
                self._plot_beta_decomposition()

        elif self.analysis_type == "RRA":
            self._plot_volcano_rra()

    def _plot_volcano_mle(self, eff: str) -> None:
        beta_col = f"{eff}|beta"
        fdr_col = (
            f"{eff}|wald-fdr"
            if f"{eff}|wald-fdr" in self.gene_df.columns
            else f"{eff}|fdr"
        )
        if (
            beta_col not in self.gene_df.columns
            or fdr_col not in self.gene_df.columns
        ):
            return

        df = self.gene_df[["Gene", beta_col, fdr_col]].copy()
        df[beta_col] = pd.to_numeric(df[beta_col], errors="coerce")
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
        df = df.dropna()

        y = -np.log10(np.clip(df[fdr_col].to_numpy(), 1e-300, 1.0))
        x = df[beta_col].to_numpy()

        sig = (df[fdr_col] <= self.cfg.fdr_threshold) & (
            np.abs(x) >= self.cfg.effect_threshold
        )
        pos = sig & (x >= self.cfg.effect_threshold)
        neg = sig & (x <= -self.cfg.effect_threshold)
        nonsig = ~sig

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            x[nonsig], y[nonsig], s=10, alpha=0.6, c="grey", edgecolors="none"
        )
        ax.scatter(x[neg], y[neg], s=10, alpha=0.8, c="blue", edgecolors="none")
        ax.scatter(x[pos], y[pos], s=10, alpha=0.8, c="red", edgecolors="none")

        ax.axvline(
            +self.cfg.effect_threshold,
            linestyle="--",
            linewidth=1,
            color="grey",
        )
        ax.axvline(
            -self.cfg.effect_threshold,
            linestyle="--",
            linewidth=1,
            color="grey",
        )
        ax.axhline(
            -np.log10(self.cfg.fdr_threshold),
            linestyle="--",
            linewidth=1,
            color="grey",
        )

        ax.set_title(f"Volcano (MLE) — {eff}")
        ax.set_xlabel("beta")
        ax.set_ylabel(r"$-\log_{10}(\mathrm{FDR})$")

        out = self.plots_dir / f"volcano_{eff}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

    def _plot_waterfall_mle(self, eff: str) -> None:
        beta_col = f"{eff}|beta"
        if beta_col not in self.gene_df.columns:
            return

        df = self.gene_df[["Gene", beta_col]].copy()
        df[beta_col] = pd.to_numeric(df[beta_col], errors="coerce")
        df = df.dropna().sort_values(by=beta_col, ascending=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(df.shape[0]), df[beta_col].to_numpy(), linewidth=1)
        ax.set_title(f"Waterfall (MLE) — {eff}")
        ax.set_xlabel("Genes ranked by beta")
        ax.set_ylabel("beta")

        out = self.plots_dir / f"waterfall_{eff}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

    def _plot_beta_decomposition(self) -> None:
        """Plot beta decomposition across effects (if 2+ effects)."""
        if len(self.effects) < 2:
            return

        # Simple scatter of first two effects
        eff1, eff2 = self.effects[0], self.effects[1]
        beta1_col = f"{eff1}|beta"
        beta2_col = f"{eff2}|beta"

        if (
            beta1_col not in self.gene_df.columns
            or beta2_col not in self.gene_df.columns
        ):
            return

        df = self.gene_df[["Gene", beta1_col, beta2_col]].copy()
        df[beta1_col] = pd.to_numeric(df[beta1_col], errors="coerce")
        df[beta2_col] = pd.to_numeric(df[beta2_col], errors="coerce")
        df = df.dropna()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(df[beta1_col], df[beta2_col], s=10, alpha=0.5, c="grey")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(f"{eff1} beta")
        ax.set_ylabel(f"{eff2} beta")
        ax.set_title("Beta Decomposition")

        out = self.plots_dir / "beta_decomposition.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

    def _plot_volcano_rra(self) -> None:
        """Implementation for RRA volcano plot.

        x: LFC column if available, else 0
        y: direction-specific FDR (pos|fdr for x>0, neg|fdr for x<0)
        """
        df = self.gene_df.copy()
        # Detect LFC columns
        lfc_col_candidates = [
            c for c in df.columns if "lfc" in c.lower() or "log2fc" in c.lower()
        ]
        lfc_col = lfc_col_candidates[0] if lfc_col_candidates else None

        pos_fdr = next((c for c in df.columns if "pos|fdr" in c), None)
        neg_fdr = next((c for c in df.columns if "neg|fdr" in c), None)
        if pos_fdr is None and neg_fdr is None:
            # Nothing to plot
            return

        if lfc_col is None:
            # Create a proxy LFC from pos/neg direction: sign = pos wins
            # Using ( -log10(pos_fdr) - -log10(neg_fdr) ) as proxy
            pf = -np.log10(
                np.clip(
                    (
                        df[pos_fdr].to_numpy()
                        if pos_fdr in df.columns
                        else np.ones(len(df))
                    ),
                    1e-300,
                    1,
                )
            )
            nf = -np.log10(
                np.clip(
                    (
                        df[neg_fdr].to_numpy()
                        if neg_fdr in df.columns
                        else np.ones(len(df))
                    ),
                    1e-300,
                    1,
                )
            )
            proxy = pf - nf
            df["__proxy_lfc__"] = proxy
            x = proxy
        else:
            x = pd.to_numeric(df[lfc_col], errors="coerce").to_numpy()

        # determine y per point
        y_vals = []
        for xi, r in zip(x, df.iterrows()):
            i = r[0]
            row = r[1]
            if np.isnan(xi):
                y_vals.append(np.nan)
            elif xi >= 0:
                f = row[pos_fdr] if pos_fdr in row.index else np.nan
                y_vals.append(f)
            else:
                f = row[neg_fdr] if neg_fdr in row.index else np.nan
                y_vals.append(f)

        y = -np.log10(np.clip(np.array(y_vals, dtype=float), 1e-300, 1.0))

        sig = y >= -np.log10(self.cfg.fdr_threshold)
        nonsig = ~sig

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            x[nonsig], y[nonsig], s=10, alpha=0.6, c="grey", edgecolors="none"
        )
        if sig.any():
            ax.scatter(
                x[sig], y[sig], s=12, alpha=0.9, c="red", edgecolors="none"
            )

        ax.set_title("Volcano (RRA)")
        ax.set_xlabel("LFC (proxy if missing)")
        ax.set_ylabel(r"$-\log_{10}(\mathrm{FDR_{directional}})$")

        out = self.plots_dir / "volcano_rra.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

    def _plot_effect_vs_reproducibility(self, eff: str) -> None:
        """
        Build effect-size vs reproducibility using sgRNA-level LFCs.

        We prefer to reuse core.plot helper `plot_effect_size_vs_reproducibility` when
        available: it accepts gene_summary_df and sgrna_summary_df and computes
        per-gene consistency metrics and a scatter plot.
        """
        if self.sgrna_df is None:
            return

        beta_col = f"{eff}|beta" if self.analysis_type == "MLE" else None
        fdr_col = f"{eff}|wald-fdr" if self.analysis_type == "MLE" else None

        try:
            fig, ax, metrics = plot_effect_size_vs_reproducibility(
                gene_summary_df=(
                    self.gene_df if beta_col is not None else self.gene_df
                ),
                effect_col=beta_col if beta_col is not None else "beta",
                fdr_col=fdr_col if fdr_col is not None else "fdr",
                sgrna_summary_df=self.sgrna_df,
                gene_col="Gene",
                fdr_threshold=self.cfg.fdr_threshold,
                title=f"Effect vs Reproducibility ({eff})",
            )
        except Exception:
            return

        out = self.plots_dir / f"effect_vs_reproducibility_{eff}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

    def export_ranklist(self, eff: Optional[str] = None) -> Path:
        """
        Export a rank list suitable for GSEA-like tools.
        For MLE: use beta (signed) or abs(beta) depending on user choice.
        For RRA: use -log10(min(pos|fdr, neg|fdr)) with a sign proxy.
        Returns path to ranklist file.
        """
        out = self.tables_dir / (f"ranklist_{eff if eff else 'global'}.rnk")

        if self.analysis_type == "MLE":
            if eff is None:
                # choose first effect
                eff = self.effects[0] if self.effects else None
            if eff is None:
                raise ValueError("No effect available for rank list")
            beta_col = f"{eff}|beta"
            if beta_col not in self.gene_df.columns:
                raise ValueError(f"Effect column {beta_col} not found")
            df = self.gene_df[["Gene", beta_col]].copy()
            df[beta_col] = pd.to_numeric(df[beta_col], errors="coerce")
            df = df.dropna()
            df = df.set_index("Gene")[beta_col]
            df.to_csv(out, sep="\t", header=False)
            return out

        if self.analysis_type == "RRA":
            df = self.gene_df.copy()
            pos_fdr = next((c for c in df.columns if "pos|fdr" in c), None)
            neg_fdr = next((c for c in df.columns if "neg|fdr" in c), None)
            # compute signed score
            pos_score = (
                -np.log10(np.clip(df[pos_fdr], 1e-300, 1))
                if pos_fdr in df.columns
                else 0
            )
            neg_score = (
                -np.log10(np.clip(df[neg_fdr], 1e-300, 1))
                if neg_fdr in df.columns
                else 0
            )
            score = pos_score - neg_score
            r = pd.Series(score, index=df["Gene"].values)
            r.to_csv(out, sep="\t", header=False)
            return out

        raise ValueError("Unknown analysis type for rank list export")

    # -----------------------
    # Report writer
    # -----------------------
    def _write_markdown_report(self, summary: dict) -> None:
        """
        Write report.md referencing generated plots and tables. Uses a minimal
        template and writes both markdown and (optionally) rendered HTML.
        """
        template_path = Path(__file__).parent.parent / "templates" / "report.md"
        if template_path.exists():
            template = template_path.read_text(encoding="utf-8")
        else:
            template = "# CRISPR Screen Report - {project_name}\n"

        # Simple replacements
        report_text = template.replace(
            "{{project_name}}", self.cfg.project_name
        )

        # Add exec summary
        report_text += "\n\n## Executive Summary\n"
        report_text += (
            f"- Detected analysis type: {summary['analysis_type_detected']}\n"
        )
        report_text += (
            f"- Normalization: {summary.get('normalization_method')}\n"
        )
        report_text += (
            f"- Preferred method: {summary.get('preferred_method')}\n"
        )
        report_text += f"- Genes: {summary.get('n_genes')}\n"
        report_text += f"- Generated: {summary.get('generated_at')}\n"

        # Add plots section with available images
        report_text += "\n\n## Plots\n"
        # list plots
        for p in sorted(self.plots_dir.glob("*.png")):
            rel = (
                Path(self.cfg.assets_dirname)
                / Path(self.cfg.plots_dirname)
                / p.name
            )
            report_text += f"\n### {p.stem}\n\n![]({rel})\n"

        # Add tables
        report_text += "\n\n## Tables\n"
        for t in sorted(self.tables_dir.glob("*.tsv")):
            rel = (
                Path(self.cfg.assets_dirname)
                / Path(self.cfg.tables_dirname)
                / t.name
            )
            report_text += f"\n- `{rel}`\n"

        report_md = self.out_dir / "report.md"
        report_md.write_text(report_text, encoding="utf-8")

        # Try to render HTML if markdown package is available
        if md is not None:
            out_html = self.out_dir / "report.html"
            out_html.write_text(md.markdown(report_text), encoding="utf-8")

    def render_html(self) -> Optional[Path]:
        report_md = self.out_dir / "report.md"
        if not report_md.exists():
            return None
        if md is None:
            return None
        out_html = self.out_dir / "report.html"
        out_html.write_text(
            md.markdown(report_md.read_text(encoding="utf-8")), encoding="utf-8"
        )
        return out_html

    # -----------------------
    # PDF Generation
    # -----------------------
    def generate_pdf_report(self) -> Optional[Path]:
        """
        Generate comprehensive PDF report with embedded plots and tables.

        Requires reportlab package. If not available, returns None.

        Returns
        -------
        Path or None
            Path to generated PDF, or None if reportlab not available
        """
        if not REPORTLAB_AVAILABLE:
            print(
                "Warning: reportlab not available. Cannot generate PDF report."
            )
            print("Install with: pip install reportlab")
            return None

        pdf_path = self.out_dir / "result_report.pdf"

        try:
            self._build_pdf(pdf_path)
            return pdf_path
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None

    def _build_pdf(self, pdf_path: Path) -> None:
        """Build PDF using reportlab."""
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        )

        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=30,
            alignment=1,  # Center
        )

        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Heading2"],
            fontSize=16,
            textColor=colors.HexColor("#34495e"),
            spaceAfter=20,
            alignment=1,
        )

        # Title page
        story.append(Spacer(1, 1.5 * inch))
        story.append(Paragraph(self.cfg.project_name, title_style))
        story.append(Paragraph("CRISPR Screen Results Report", subtitle_style))
        story.append(Spacer(1, 0.3 * inch))

        if self._summary:
            story.append(
                Paragraph(
                    f"Analysis Type: {self._summary.get('analysis_type_detected', 'N/A')}",
                    styles["Normal"],
                )
            )
            story.append(
                Paragraph(
                    f"Generated: {self._summary.get('generated_at', 'N/A')}",
                    styles["Normal"],
                )
            )

        story.append(PageBreak())

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))

        if self._summary:
            summary_items = [
                f"<b>Analysis Type:</b> {self._summary.get('analysis_type_detected', 'N/A')}",
                f"<b>Number of Genes:</b> {self._summary.get('n_genes', 'N/A')}",
                f"<b>Normalization Method:</b> {self._summary.get('normalization_method', 'N/A')}",
                f"<b>Preferred Analysis:</b> {self._summary.get('preferred_method', 'N/A')}",
            ]

            for item in summary_items:
                story.append(Paragraph(f"• {item}", styles["BodyText"]))
                story.append(Spacer(1, 0.1 * inch))

            # Effect statistics for MLE
            if self.analysis_type == "MLE" and "effect_stats" in self._summary:
                story.append(Spacer(1, 0.3 * inch))
                story.append(Paragraph("Effect Statistics", styles["Heading2"]))

                for eff, stats in self._summary["effect_stats"].items():
                    story.append(Paragraph(f"<b>{eff}</b>", styles["Heading3"]))
                    story.append(
                        Paragraph(
                            f"Significant genes: {stats['n_sig']} (FDR ≤ {self.cfg.fdr_threshold})",
                            styles["BodyText"],
                        )
                    )
                    story.append(
                        Paragraph(
                            f"Median |beta|: {stats['beta_median_abs']:.3f}",
                            styles["BodyText"],
                        )
                    )
                    story.append(Spacer(1, 0.15 * inch))

        story.append(PageBreak())

        # Plots section
        story.append(Paragraph("Visualizations", styles["Heading1"]))
        story.append(Spacer(1, 0.3 * inch))

        plot_files = sorted(self.plots_dir.glob("*.png"))
        for plot_path in plot_files:
            plot_title = plot_path.stem.replace("_", " ").title()

            # Create plot section
            plot_section = [
                Paragraph(plot_title, styles["Heading2"]),
                Spacer(1, 0.1 * inch),
            ]

            # Add image
            try:
                img = Image(str(plot_path), width=5 * inch, height=3.5 * inch)
                img.hAlign = "CENTER"
                plot_section.append(img)
                plot_section.append(Spacer(1, 0.3 * inch))
            except Exception as e:
                plot_section.append(
                    Paragraph(f"Error loading image: {e}", styles["BodyText"])
                )

            # Keep plot and title together
            story.extend([KeepTogether(plot_section)])

            # Page break after every 2 plots to avoid cramming
            if len(story) % 4 == 0:
                story.append(PageBreak())

        # Tables section
        if any(self.tables_dir.glob("*.tsv")):
            story.append(PageBreak())
            story.append(Paragraph("Data Tables", styles["Heading1"]))
            story.append(Spacer(1, 0.2 * inch))

            table_files = sorted(self.tables_dir.glob("*.tsv"))
            for table_path in table_files:
                rel_path = table_path.relative_to(self.out_dir)
                story.append(
                    Paragraph(
                        f"• {table_path.stem}: <font color='blue'>{rel_path}</font>",
                        styles["BodyText"],
                    )
                )
                story.append(Spacer(1, 0.1 * inch))

        # Build PDF
        doc.build(story)
