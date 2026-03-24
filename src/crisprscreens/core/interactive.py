import json
from typing import Callable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pandas import DataFrame

from .plots import _scale


def select_top_n(
    n: int,
    effect_col: str,
    fdr_col: str | None = None,
    fdr_threshold: float = 0.05,
) -> Callable[[DataFrame], pd.Series]:
    """
    Helper function to select the top N entries based on the absolute effect size.
    Returns a function that can be applied to a DataFrame.
    """

    def selector(df: DataFrame) -> pd.Series:
        df["_abs_effect"] = df[effect_col].abs()
        if fdr_col:
            mask = (df[fdr_col] < fdr_threshold) & (
                df["_abs_effect"].rank(method="first", ascending=False) <= n
            )
        else:
            mask = df["_abs_effect"].rank(method="first", ascending=False) <= n
        return mask

    return selector


def plot_effect_size(
    df: DataFrame,
    effect_col: str,
    p_value_col: str,
    label_col: str | None = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
    select: list[str] | None | Callable = None,
    ylabel: str = "Gene",
    output_html: str | None = "plot.html",
    output_image: str | None = None,
    scale_x: Literal["log"] | Callable | None = None,
    scale_y: Literal["log"] | Callable | None = None,
    plot_rank: bool = True,
    log_offset: float = 1e-10,  # Offset gegen -inf bei sehr kleinen p-Werten
) -> go.Figure:

    # --- Daten vorbereiten ---
    df = df.copy()
    if effect_col not in df.columns:
        raise KeyError(f"Effect column '{effect_col}' not found in DataFrame")

    df[effect_col] = pd.to_numeric(df[effect_col], errors="coerce")

    if label_col is not None:
        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in DataFrame")
        df["label"] = df[label_col].astype(str)
    else:
        df["label"] = df.index.astype(str)

    df = df.dropna(subset=[effect_col])
    df = df.assign(_abs_effect=df[effect_col].abs())

    if plot_rank:
        df = df.sort_values(effect_col, ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("_abs_effect", ascending=True).reset_index(
            drop=True
        )

    df["rank"] = df.index.astype(int)

    if df.shape[0] == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to plot",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=14,
        )
        return fig

    # --- Selected Gene bestimmen ---
    if select is not None:
        if callable(select):
            sel_mask = select(df)
            sel_labels = set(df.loc[sel_mask, "label"])
        else:
            sel_labels = set(select)
    else:
        sel_labels = set()

    # --- Achsen bestimmen ---
    if plot_rank:
        raw_x = df["rank"].to_numpy(dtype=np.float64)
        raw_y = df[effect_col].to_numpy(dtype=np.float64)
        x_title_base = "Rank"
        y_title_base = effect_col
        scaled_x = raw_x
        scaled_y = _scale(raw_y, scale_y)
        show_vline = False
        show_hline = True
    else:
        raw_x = df[effect_col].to_numpy(dtype=np.float64)
        raw_y = df[p_value_col].to_numpy(dtype=np.float64)
        x_title_base = effect_col
        y_title_base = p_value_col
        scaled_x = _scale(raw_x, scale_x)
        # -log10 mit Offset für p-Werte
        scaled_y = _scale(raw_y, scale_y, offset=log_offset)
        show_vline = True
        show_hline = True

    def axis_label(base, scale, is_rank=False):
        if is_rank:
            return base
        if isinstance(scale, str) and scale == "log":
            return f"-log10({base})"
        return base

    x_title = axis_label(x_title_base, scale_x, is_rank=plot_rank)
    y_title = axis_label(y_title_base, scale_y)

    df["_x"] = scaled_x
    df["_y"] = scaled_y
    colors = ["red" if v >= 0 else "blue" for v in df[effect_col].values]
    df["_color"] = colors

    mask_sel = df["label"].isin(sel_labels)
    df_sel = df[mask_sel]
    df_rest = df[~mask_sel]

    hovertemplate = (
        "<b>%{text}</b><br>"
        f"{x_title_base}: %{{x:.4f}}<br>"
        f"{y_title_base}: %{{y:.4f}}<extra></extra>"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_rest["_x"],
            y=df_rest["_y"],
            mode="markers",
            name="genes",
            marker=dict(color=df_rest["_color"], size=6, symbol="circle"),
            text=df_rest["label"],
            hovertemplate=hovertemplate,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_sel["_x"],
            y=df_sel["_y"],
            mode="markers+text",
            name="selected",
            marker=dict(
                color=df_sel["_color"],
                size=9,
                symbol="circle",
                line=dict(width=1.5, color="black"),
            ),
            text=df_sel["label"],
            textposition="top center",
            hovertemplate=hovertemplate,
        )
    )

    if show_hline:
        fig.add_hline(
            y=center_y, line_dash="dash", line_color="gray", opacity=0.7
        )
    if show_vline:
        fig.add_vline(
            x=center_x, line_dash="dash", line_color="gray", opacity=0.7
        )

    fig.update_layout(
        title=f"{'Rank vs ' + effect_col if plot_rank else f'{effect_col} vs {p_value_col}'} — {ylabel} Labels",
        xaxis_title=x_title,
        yaxis_title=y_title,
        dragmode="zoom",
        hovermode="closest",
        template="plotly_white",
    )

    # --- JS Vorbereitung ---
    initial_scale_y = (
        "log" if isinstance(scale_y, str) and scale_y == "log" else "linear"
    )
    # Im rank-Modus: initial ascending=True (kleinster Effekt links)
    initial_ascending = "true"

    all_data = (
        df[["label", "_x", "_y", "_color"]]
        .rename(columns={"_x": "x", "_y": "y", "_color": "color"})
        .to_dict(orient="list")
    )
    all_data["raw_x"] = raw_x.tolist()
    all_data["raw_y"] = raw_y.tolist()
    # Für Rank-Toggle: rohe Effektwerte zum Neusortieren
    all_data["effect"] = df[effect_col].tolist()
    all_data["label_list"] = df["label"].tolist()
    # nach all_data["label_list"] = ...
    all_data["effect"] = df[effect_col].tolist()
    all_data["label_list"] = df["label"].tolist()
    initial_selected = list(sel_labels)

    post_script = f"""
    const allData = {json.dumps(all_data)};
    const initialSelected = {json.dumps(initial_selected)};
    const plotRank = {'true' if plot_rank else 'false'};
    const logOffset = {log_offset};
    let currentScaleY = '{initial_scale_y}';
    let ascending = {initial_ascending};
    let spreadStrength = 1.0;  // Multiplikator für Label-Spreading

    function applyScaleY(values) {{
        if (currentScaleY === 'log') {{
            return values.map(v => -Math.log10(Math.max(v, logOffset)));
        }}
        return [...values];
    }}

    function getReranked() {{
        const n = allData.effect.length;
        const indices = Array.from({{length: n}}, (_, i) => i);
        indices.sort((a, b) =>
            ascending
                ? allData.effect[a] - allData.effect[b]
                : allData.effect[b] - allData.effect[a]
        );
        const newRank = new Array(n);
        indices.forEach((origIdx, rank) => {{ newRank[origIdx] = rank; }});
        return newRank;
    }}

    function getScaledData() {{
        const y = applyScaleY(allData.raw_y);
        let x;
        if (plotRank) {{
            const ranks = getReranked();
            x = ranks.map(r => r);
        }} else {{
            x = [...allData.raw_x];
        }}
        return {{ x, y }};
    }}

    // Label-Spreading: verschiebt Labels iterativ auseinander
    function spreadLabels(positions, strength) {{
        const spread = positions.map(p => ({{...p}}));
        const iterations = 50;
        const minDist = strength * 0.04;  // minimaler Abstand in Achseneinheiten (normiert)

        for (let iter = 0; iter < iterations; iter++) {{
            for (let i = 0; i < spread.length; i++) {{
                for (let j = i + 1; j < spread.length; j++) {{
                    const dx = (spread[j].lx - spread[i].lx);
                    const dy = (spread[j].ly - spread[i].ly);
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < minDist && dist > 0) {{
                        const push = (minDist - dist) / 2;
                        const nx = dx / dist, ny = dy / dist;
                        spread[i].lx -= nx * push;
                        spread[i].ly -= ny * push;
                        spread[j].lx += nx * push;
                        spread[j].ly += ny * push;
                    }}
                }}
            }}
        }}
        return spread;
    }}

    // Labels als Plotly Annotations rendern (erlaubt freie Positionierung + Linie zum Punkt)
    function renderLabels(scaled, selIdx) {{
        if (selIdx.length === 0) {{
            Plotly.relayout(plotDiv, {{annotations: []}});
            return;
        }}

        // Achsenbereiche für Normierung holen
        const layout = plotDiv.layout;
        const xRange = layout.xaxis.range || [Math.min(...scaled.x), Math.max(...scaled.x)];
        const yRange = layout.yaxis.range || [Math.min(...scaled.y), Math.max(...scaled.y)];
        const xSpan = xRange[1] - xRange[0] || 1;
        const ySpan = yRange[1] - yRange[0] || 1;

        // Normierte Positionen für Spreading
        let positions = selIdx.map(i => ({{
            idx: i,
            px: scaled.x[i],  // Punkt-Position (fix)
            py: scaled.y[i],
            lx: scaled.x[i] / xSpan,  // Label-Position normiert (wird verschoben)
            ly: scaled.y[i] / ySpan,
        }}));

        if (spreadStrength > 0) {{
            positions = spreadLabels(positions, spreadStrength);
        }}

        // Zurück in Achseneinheiten
        const annotations = positions.map(p => ({{
            x: p.px,          // Pfeil zeigt auf Datenpunkt
            y: p.py,
            ax: p.lx * xSpan, // Label-Position
            ay: p.ly * ySpan,
            axref: 'x',
            ayref: 'y',
            text: allData.label[p.idx],
            showarrow: true,
            arrowhead: 2,
            arrowsize: 0.8,
            arrowwidth: 1,
            arrowcolor: '#666',
            font: {{size: 11}},
            bgcolor: 'rgba(255,255,255,0.0)',
            bordercolor: 'rgba(0,0,0,0.0)',
            borderpad: 2,
        }}));

        Plotly.relayout(plotDiv, {{annotations}});
    }}

    const plotDiv = document.getElementById('{{plot_id}}');
    const container = plotDiv.parentElement;

    const controls = document.createElement('div');
    controls.style.cssText = 'margin:10px 0; font-family:sans-serif; display:flex; gap:12px; align-items:center; flex-wrap:wrap;';
    controls.innerHTML = `
        <label style="font-weight:bold;">Label genes:</label>
        <input id="gene-input" type="text"
            placeholder="e.g. BRCA1, TP53, MYC"
            style="width:300px; padding:5px 8px; border:1px solid #ccc; border-radius:4px; font-size:13px;"
            value="${{initialSelected.join(', ')}}">
        <button id="gene-apply"
            style="padding:5px 12px; background:#4a90d9; color:white; border:none; border-radius:4px; cursor:pointer; font-size:13px;">
            Apply
        </button>
        <button id="gene-clear"
            style="padding:5px 12px; background:#aaa; color:white; border:none; border-radius:4px; cursor:pointer; font-size:13px;">
            Clear
        </button>
        ${{plotRank ? `
        <span style="margin-left:8px; font-weight:bold;">Sort:</span>
        <button id="ascending-toggle"
            style="padding:5px 12px; background:#7b68cc; color:white; border:none; border-radius:4px; cursor:pointer; font-size:13px;">
            ↑ Ascending
        </button>` : `
        <span style="margin-left:8px; font-weight:bold;">Scale Y ({y_title_base}):</span>
        <button id="scale-y-toggle"
            style="padding:5px 12px; background:#e8a838; color:white; border:none; border-radius:4px; cursor:pointer; font-size:13px;">
            ${{currentScaleY === 'log' ? 'Y: Linear' : 'Y: -log10'}}
        </button>`}}
        <span style="margin-left:8px; font-weight:bold;">Label spread:</span>
        <input id="spread-slider" type="range" min="0" max="5" step="0.1" value="1.0"
            style="width:120px; cursor:pointer;">
        <span id="spread-value" style="font-size:12px; color:#444; min-width:24px;">1.0</span>
        <button id="spread-reset"
            style="padding:5px 10px; background:#e07b5a; color:white; border:none; border-radius:4px; cursor:pointer; font-size:13px;">
            Reset
        </button>
        <span id="gene-status" style="font-size:12px; color:#666; margin-left:8px;"></span>
    `;
    container.insertBefore(controls, plotDiv);

    function getCurrentLabels() {{
        const input = document.getElementById('gene-input').value;
        return input.split(/[,\\n\\s]+/).filter(g => g.length > 0);
    }}

    function updatePlot() {{
        const scaled = getScaledData();
        const selSet = new Set(getCurrentLabels().map(g => g.trim()).filter(g => g));

        const restIdx = [], selIdx = [];
        allData.label.forEach((label, i) => {{
            if (selSet.has(label)) selIdx.push(i);
            else restIdx.push(i);
        }});

        const get = (arr, idx) => idx.map(i => arr[i]);

        // Trace 0: alle Gene ohne Labels (mode=markers)
        Plotly.restyle(plotDiv, {{
            x: [get(scaled.x, restIdx)],
            y: [get(scaled.y, restIdx)],
            text: [get(allData.label, restIdx)],
            'marker.color': [get(allData.color, restIdx)],
            mode: ['markers'],
        }}, [0]);

        // Trace 1: selected Gene als Punkte ohne Plotly-Text (Labels via Annotations)
        Plotly.restyle(plotDiv, {{
            x: [get(scaled.x, selIdx)],
            y: [get(scaled.y, selIdx)],
            text: [get(allData.label, selIdx)],
            'marker.color': [get(allData.color, selIdx)],
            mode: ['markers'],   // kein 'markers+text' mehr — Annotations übernehmen
        }}, [1]);

        // Annotations für Label-Spreading
        renderLabels(scaled, selIdx);

        if (plotRank) {{
            document.getElementById('ascending-toggle').textContent =
                ascending ? '↑ Ascending' : '↓ Descending';
            Plotly.relayout(plotDiv, {{'xaxis.title': 'Rank'}});
        }} else {{
            const yLabel = currentScaleY === 'log'
                ? '-log10({y_title_base})'
                : '{y_title_base}';
            Plotly.relayout(plotDiv, {{'yaxis.title': yLabel}});
            document.getElementById('scale-y-toggle').textContent =
                currentScaleY === 'log' ? 'Y: Linear' : 'Y: -log10';
        }}

        const found = selIdx.length;
        const notFound = [...selSet].filter(g => !allData.label.includes(g));
        let status = found > 0 ? `${{found}} gene(s) labeled` : '';
        if (notFound.length > 0) status += ` | not found: ${{notFound.join(', ')}}`;
        document.getElementById('gene-status').textContent = status;
    }}

    // --- Event Listeners ---
    document.getElementById('gene-apply').addEventListener('click', updatePlot);
    document.getElementById('gene-input').addEventListener('keydown', e => {{
        if (e.key === 'Enter') updatePlot();
    }});
    document.getElementById('gene-clear').addEventListener('click', () => {{
        document.getElementById('gene-input').value = '';
        updatePlot();
    }});

    if (plotRank) {{
        document.getElementById('ascending-toggle').addEventListener('click', () => {{
            ascending = !ascending;
            updatePlot();
        }});
    }} else {{
        document.getElementById('scale-y-toggle').addEventListener('click', () => {{
            currentScaleY = currentScaleY === 'log' ? 'linear' : 'log';
            updatePlot();
        }});
    }}

    // Spread Slider: live update beim Ziehen
    document.getElementById('spread-slider').addEventListener('input', e => {{
        spreadStrength = parseFloat(e.target.value);
        document.getElementById('spread-value').textContent = spreadStrength.toFixed(1);
        updatePlot();
    }});

    // Reset Button: Slider auf 1.0, Labels neu rendern
    document.getElementById('spread-reset').addEventListener('click', () => {{
        spreadStrength = 1.0;
        document.getElementById('spread-slider').value = '1.0';
        document.getElementById('spread-value').textContent = '1.0';
        updatePlot();
    }});
    """

    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "effect_size_snapshot",
            "height": 1200,
            "width": 1600,
            "scale": 2,
        },
        "scrollZoom": True,
        "displayModeBar": True,
    }

    if output_html is not None:
        fig.write_html(
            output_html,
            include_plotlyjs="cdn",
            config=config,
            post_script=post_script,
        )
        print(f"HTML saved: {output_html}")

    if output_image is not None:
        pio.write_image(fig, output_image, width=1600, height=1200, scale=2)
        print(f"Image saved: {output_image}")

    return fig
