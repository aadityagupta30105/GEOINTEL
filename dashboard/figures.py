"""
Dashboard Figure Builders
=========================
Pure figure construction. Every function in this module takes analysed data
and returns a styled Plotly figure; none of them touch Streamlit state, which
keeps the visualisation layer testable in isolation.
"""

from __future__ import annotations

from typing import Any, Final

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analysis.graph_builder import NetworkStats
from dashboard.geodata import COUNTRY_GEO, format_gdp, marker_size
from dashboard.theme import (
    ACCENT,
    ACCENT_ALT,
    BLOC_COLORS,
    BORDER,
    MUTED,
    NEGATIVE,
    POSITIVE,
    SURFACE,
    TEXT,
    apply_chart_layout,
    apply_geo_layout,
    rgba,
)

__all__ = [
    "build_network_figure",
    "build_bloc_figure",
    "build_radar_chart",
    "build_bilateral_chart",
    "build_tone_heatmap",
    "build_pagerank_bar",
    "build_temporal_line",
    "build_activity_figure",
    "build_stat_rows",
]

# Edge rendering bounds.
_EDGE_MIN_ALPHA: Final[float] = 0.08
_EDGE_MAX_ALPHA: Final[float] = 0.75
_EDGE_MIN_WIDTH: Final[float] = 0.4
_EDGE_MAX_WIDTH: Final[float] = 3.5
_EDGE_WIDTH_DIVISOR: Final[float] = 18.0

_RADAR_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("pagerank", "PageRank"),
    ("betweenness", "Betweenness"),
    ("eigenvector", "Eigenvector"),
    ("in_degree_weighted", "In-Degree"),
    ("out_degree_weighted", "Out-Degree"),
)


def _edge_alpha(tone: float) -> float:
    """Map edge tone magnitude onto an opacity.

    Parameters
    ----------
    tone : float
        Normalised tone in ``[-1, 1]``.

    Returns
    -------
    float
        Opacity within the configured bounds.
    """
    return min(_EDGE_MAX_ALPHA, max(_EDGE_MIN_ALPHA, abs(tone) * 2.0))


def _edge_width(weight: float) -> float:
    """Map interaction volume onto a stroke width.

    Parameters
    ----------
    weight : float
        Edge weight.

    Returns
    -------
    float
        Stroke width in pixels, within the configured bounds.
    """
    return max(_EDGE_MIN_WIDTH, min(weight / _EDGE_WIDTH_DIVISOR, _EDGE_MAX_WIDTH))


def _geo_edge_trace(
    source: str,
    target: str,
    color: str,
    width: float,
) -> go.Scattergeo:
    """Build a single great-circle edge trace between two countries.

    Parameters
    ----------
    source, target : str
        ISO-3 codes present in :data:`COUNTRY_GEO`.
    color : str
        CSS colour for the stroke.
    width : float
        Stroke width in pixels.

    Returns
    -------
    plotly.graph_objects.Scattergeo
        Line trace excluded from the legend and from hover.
    """
    origin, destination = COUNTRY_GEO[source], COUNTRY_GEO[target]
    return go.Scattergeo(
        lat=[origin.lat, destination.lat, None],
        lon=[origin.lon, destination.lon, None],
        mode="lines",
        line={"width": width, "color": color},
        hoverinfo="none",
        showlegend=False,
    )


def _node_hover(
    code: str,
    metrics_df: pd.DataFrame,
    extra: str = "",
) -> str:
    """Render the hover card for one country node.

    Parameters
    ----------
    code : str
        ISO-3 country code.
    metrics_df : pandas.DataFrame
        Metrics frame indexed by country code.
    extra : str, optional
        Additional pre-rendered HTML appended to the card.

    Returns
    -------
    str
        HTML hover content.
    """
    geo = COUNTRY_GEO[code]
    if code in metrics_df.index:
        row = metrics_df.loc[code]
        pagerank = float(row["pagerank"])
        conflict = float(row["conflict_ratio"])
        events = int(row["total_events"])
    else:
        pagerank, conflict, events = 0.0, 0.0, 0

    return (
        f"<b>{geo.name}</b> [{code}]<br>"
        f"GDP (PPP): {format_gdp(geo.gdp_ppp_billions)}<br>"
        f"PageRank: {pagerank:.5f}<br>"
        f"Conflict ratio: {conflict:.1%}<br>"
        f"Interactions: {events:,}"
        f"{extra}"
    )


def build_network_figure(
    G: nx.DiGraph,
    metrics_df: pd.DataFrame,
    stats: NetworkStats,
    min_weight: float,
    max_nodes: int,
    color_by: str,
) -> go.Figure:
    """Project the interaction network onto a world map.

    Node position is the country centroid, node diameter encodes GDP at PPP,
    node colour encodes the selected metric, and edge colour encodes
    cooperative against conflictual tone.

    Parameters
    ----------
    G : networkx.DiGraph
        Interaction graph.
    metrics_df : pandas.DataFrame
        Metrics frame indexed by country code, sorted by PageRank.
    stats : NetworkStats
        Global statistics, consulted for the community assignment.
    min_weight : float
        Minimum edge weight to render.
    max_nodes : int
        Maximum number of countries to render, taken from the PageRank order.
    color_by : {"PageRank", "Conflict Ratio", "Community"}
        Node colour encoding.

    Returns
    -------
    plotly.graph_objects.Figure
        Styled geographic figure.
    """
    community_map = stats.get("community_map", {})

    visible = [
        code for code in metrics_df.head(max_nodes).index if code in COUNTRY_GEO
    ]
    subgraph = G.subgraph(visible)

    traces: list[go.Scattergeo] = [
        _geo_edge_trace(
            source,
            target,
            rgba(
                POSITIVE if data.get("tone", 0.0) >= 0 else NEGATIVE,
                _edge_alpha(data.get("tone", 0.0)),
            ),
            _edge_width(data.get("weight", 0.0)),
        )
        for source, target, data in subgraph.edges(data=True)
        if data.get("weight", 0.0) >= min_weight
    ]

    latitudes: list[float] = []
    longitudes: list[float] = []
    sizes: list[float] = []
    colors: list[float] = []
    labels: list[str] = []
    hovers: list[str] = []

    for code in subgraph.nodes():
        geo = COUNTRY_GEO[code]
        latitudes.append(geo.lat)
        longitudes.append(geo.lon)
        sizes.append(marker_size(geo.gdp_ppp_billions))
        labels.append(code)
        hovers.append(_node_hover(code, metrics_df))

        if code in metrics_df.index and color_by == "PageRank":
            colors.append(float(metrics_df.loc[code, "pagerank"]))
        elif code in metrics_df.index and color_by == "Conflict Ratio":
            colors.append(float(metrics_df.loc[code, "conflict_ratio"]))
        else:
            colors.append(float(community_map.get(code, 0)))

    traces.append(go.Scattergeo(
        lat=latitudes,
        lon=longitudes,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont={"size": 9, "color": TEXT},
        hovertext=hovers,
        hoverinfo="text",
        marker={
            "size": sizes,
            "sizemode": "diameter",
            "color": colors,
            "colorscale": "RdYlGn_r" if color_by == "Conflict Ratio" else "Viridis",
            "showscale": True,
            "colorbar": {
                "title": {"text": color_by, "font": {"color": TEXT, "size": 11}},
                "thickness": 10,
                "len": 0.45,
                "tickfont": {"color": MUTED, "size": 9},
                "bgcolor": "rgba(17,24,39,0.80)",
                "bordercolor": BORDER,
                "outlinecolor": BORDER,
            },
            "opacity": 0.92,
            "line": {"width": 1.2, "color": "#0f172a"},
        },
        showlegend=False,
    ))

    return apply_geo_layout(go.Figure(data=traces), height=580)


def build_bloc_figure(
    G: nx.DiGraph,
    metrics_df: pd.DataFrame,
    assignments: dict[str, str],
    affinity: dict[str, dict[str, float]],
    poles: list[str],
    min_weight: float,
) -> go.Figure:
    """Project bloc assignment onto a world map.

    Nodes are coloured by assigned bloc and grouped into one legend entry per
    pole. Edges take the colour of the source bloc when cooperative and the
    conflict colour when not. Poles themselves are drawn larger with a ring.

    Parameters
    ----------
    G : networkx.DiGraph
        Interaction graph.
    metrics_df : pandas.DataFrame
        Metrics frame indexed by country code.
    assignments : dict of str to str
        Country to pole assignment.
    affinity : dict
        Full affinity table, rendered into the hover cards.
    poles : list of str
        Power poles in display order.
    min_weight : float
        Minimum edge weight to render.

    Returns
    -------
    plotly.graph_objects.Figure
        Styled geographic figure.
    """
    pole_index = {pole: index for index, pole in enumerate(poles)}
    traces: list[go.Scattergeo] = []

    for source, target, data in G.edges(data=True):
        if data.get("weight", 0.0) < min_weight:
            continue
        if source not in COUNTRY_GEO or target not in COUNTRY_GEO:
            continue

        tone = data.get("tone", 0.0)
        alpha = min(0.55, max(0.06, abs(tone) * 1.8))
        bloc = assignments.get(source, "")
        bloc_color = BLOC_COLORS[pole_index.get(bloc, 0) % len(BLOC_COLORS)]
        color = rgba(bloc_color if tone >= 0 else NEGATIVE, alpha)

        traces.append(_geo_edge_trace(
            source, target, color, max(0.3, min(data["weight"] / 18.0, 3.0)),
        ))

    grouped: dict[str, list[str]] = {pole: [] for pole in poles}
    for code, bloc in assignments.items():
        if code in COUNTRY_GEO and bloc in grouped:
            grouped[bloc].append(code)

    for index, pole in enumerate(poles):
        members = sorted(grouped[pole])
        color = BLOC_COLORS[index % len(BLOC_COLORS)]

        hovers: list[str] = []
        for code in members:
            scores = affinity.get(code, {})
            if scores:
                ranked = sorted(scores.items(), key=lambda item: -item[1])
                bars = "".join(
                    f"{name}: {'|' * int(max(0, (score + 1) / 2 * 12))} {score:+.3f}<br>"
                    for name, score in ranked
                )
                extra = f"<br><b>Affinity</b><br>{bars}"
            else:
                extra = "<br><b>Pole</b>"
            hovers.append(_node_hover(code, metrics_df, extra))

        traces.append(go.Scattergeo(
            lat=[COUNTRY_GEO[code].lat for code in members],
            lon=[COUNTRY_GEO[code].lon for code in members],
            mode="markers+text",
            name=f"{pole} bloc",
            text=members,
            textposition="top center",
            textfont={"size": 9, "color": color},
            hovertext=hovers,
            hoverinfo="text",
            marker={
                "size": [
                    marker_size(
                        COUNTRY_GEO[code].gdp_ppp_billions,
                        base=22.0 if code == pole else 8.0,
                        span=52.0 if code == pole else 42.0,
                    )
                    for code in members
                ],
                "sizemode": "diameter",
                "color": color,
                "opacity": 0.88,
                "line": {
                    "width": [3 if code == pole else 1 for code in members],
                    "color": ["#ffffff" if code == pole else SURFACE for code in members],
                },
            },
        ))

    return apply_geo_layout(go.Figure(data=traces), height=560)


def build_radar_chart(metrics_df: pd.DataFrame, country: str) -> go.Figure:
    """Build a normalised centrality profile for one country.

    Each axis is min-max normalised across all countries in the frame, so the
    polygon shows relative standing rather than absolute magnitude.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Metrics frame indexed by country code.
    country : str
        ISO-3 country code.

    Returns
    -------
    plotly.graph_objects.Figure
        Polar figure, empty when the country is absent.
    """
    if country not in metrics_df.index:
        return apply_chart_layout(go.Figure(), height=300, show_legend=False)

    row = metrics_df.loc[country]

    def normalise(column: str) -> float:
        low, high = metrics_df[column].min(), metrics_df[column].max()
        return float((row[column] - low) / (high - low)) if high > low else 0.0

    values = [normalise(column) for column, _ in _RADAR_METRICS]
    axes = [label for _, label in _RADAR_METRICS]

    figure = go.Figure(go.Scatterpolar(
        r=[*values, values[0]],
        theta=[*axes, axes[0]],
        fill="toself",
        fillcolor=rgba(ACCENT, 0.15),
        line={"color": ACCENT, "width": 2},
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        name=country,
    ))
    figure.update_layout(
        polar={
            "radialaxis": {
                "visible": True, "range": [0, 1],
                "tickfont": {"color": MUTED, "size": 9},
                "gridcolor": BORDER, "linecolor": BORDER,
            },
            "angularaxis": {
                "tickfont": {"color": TEXT, "size": 10},
                "gridcolor": BORDER, "linecolor": BORDER,
            },
            "bgcolor": "rgba(0,0,0,0)",
        },
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Space Mono, monospace"},
        title={
            "text": f"{country} influence profile",
            "font": {"color": ACCENT, "family": "Syne, sans-serif", "size": 13},
            "x": 0, "xanchor": "left",
        },
        margin={"l": 40, "r": 40, "t": 46, "b": 24},
        showlegend=False,
        hoverlabel={"bgcolor": SURFACE, "bordercolor": BORDER, "font": {"color": TEXT}},
    )
    return figure


def build_bilateral_chart(G: nx.DiGraph, country_a: str, country_b: str) -> go.Figure:
    """Compare event-type volume in each direction of a dyad.

    Parameters
    ----------
    G : networkx.DiGraph
        Interaction graph.
    country_a, country_b : str
        ISO-3 country codes.

    Returns
    -------
    plotly.graph_objects.Figure
        Grouped bar figure, empty when neither direction exists.
    """
    forward: dict[str, int] = (
        dict(G[country_a][country_b].get("event_types", {}))
        if G.has_edge(country_a, country_b)
        else {}
    )
    reverse: dict[str, int] = (
        dict(G[country_b][country_a].get("event_types", {}))
        if G.has_edge(country_b, country_a)
        else {}
    )

    categories = sorted(set(forward) | set(reverse))
    if not categories:
        return apply_chart_layout(go.Figure(), height=320, show_legend=False)

    figure = go.Figure(data=[
        go.Bar(
            name=f"{country_a} to {country_b}",
            x=categories,
            y=[forward.get(category, 0) for category in categories],
            marker_color=ACCENT,
            opacity=0.85,
            hovertemplate="%{x}: %{y} events<extra></extra>",
        ),
        go.Bar(
            name=f"{country_b} to {country_a}",
            x=categories,
            y=[reverse.get(category, 0) for category in categories],
            marker_color=ACCENT_ALT,
            opacity=0.85,
            hovertemplate="%{x}: %{y} events<extra></extra>",
        ),
    ])
    figure.update_layout(barmode="group")
    return apply_chart_layout(
        figure, height=320, title=f"{country_a} / {country_b} event breakdown"
    )


def build_tone_heatmap(
    G: nx.DiGraph,
    metrics_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """Build the directed tone matrix for the leading countries.

    Parameters
    ----------
    G : networkx.DiGraph
        Interaction graph.
    metrics_df : pandas.DataFrame
        Metrics frame indexed by country code, sorted by PageRank.
    top_n : int, optional
        Number of countries on each axis.

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap figure. Absent dyads are rendered as zero.
    """
    countries = metrics_df.head(top_n).index.tolist()
    matrix = np.zeros((len(countries), len(countries)), dtype="float64")

    for row_index, source in enumerate(countries):
        for column_index, target in enumerate(countries):
            if G.has_edge(source, target):
                matrix[row_index][column_index] = G[source][target]["tone"]

    figure = go.Figure(go.Heatmap(
        z=matrix,
        x=countries,
        y=countries,
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
        xgap=1,
        ygap=1,
        colorbar={
            "title": {"text": "Tone", "font": {"color": TEXT, "size": 11}},
            "tickfont": {"color": MUTED, "size": 9},
            "thickness": 10,
            "outlinecolor": BORDER,
        },
        hovertemplate="%{y} to %{x}: %{z:.3f}<extra></extra>",
    ))
    return apply_chart_layout(
        figure,
        height=520,
        title="Bilateral tone matrix (green cooperative, red conflictual)",
        show_legend=False,
    )


def build_pagerank_bar(metrics_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Rank the leading countries by PageRank, coloured by conflict ratio.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Metrics frame indexed by country code.
    top_n : int, optional
        Number of countries to plot.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar figure.
    """
    top = metrics_df.head(top_n)

    figure = go.Figure(go.Bar(
        x=top.index.tolist(),
        y=top["pagerank"].tolist(),
        marker={
            "color": top["conflict_ratio"].tolist(),
            "colorscale": "RdYlGn_r",
            "cmin": 0, "cmax": 1,
            "colorbar": {
                "title": {"text": "Conflict", "font": {"color": TEXT, "size": 11}},
                "tickformat": ".0%",
                "tickfont": {"color": MUTED, "size": 9},
                "thickness": 10,
                "outlinecolor": BORDER,
            },
            "line": {"width": 0},
        },
        customdata=np.stack([top["conflict_ratio"].to_numpy()], axis=-1),
        hovertemplate=(
            "<b>%{x}</b><br>PageRank: %{y:.5f}"
            "<br>Conflict ratio: %{customdata[0]:.1%}<extra></extra>"
        ),
    ))
    figure.update_yaxes(title_text="PageRank")
    return apply_chart_layout(
        figure, height=360, title="PageRank influence ranking", show_legend=False
    )


def build_temporal_line(
    frame: pd.DataFrame,
    column: str,
    title: str,
    color: str,
    zero_line: bool = False,
) -> go.Figure:
    """Plot one metric across temporal snapshots.

    Parameters
    ----------
    frame : pandas.DataFrame
        Per-period statistics carrying a ``period`` column.
    column : str
        Column to plot.
    title : str
        Chart title.
    color : str
        Series colour.
    zero_line : bool, optional
        Whether to draw a reference line at zero.

    Returns
    -------
    plotly.graph_objects.Figure
        Line figure with markers.
    """
    figure = go.Figure(go.Scatter(
        x=frame["period"],
        y=frame[column],
        mode="lines+markers",
        line={"color": color, "width": 2},
        marker={"size": 7, "color": color, "line": {"width": 0}},
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    if zero_line:
        figure.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
    apply_chart_layout(figure, height=300, title=title, show_legend=False)
    # Period labels are ordinal categories. Without this, Plotly parses
    # "2024-01" as a date and interpolates ticks that are not data points.
    figure.update_xaxes(type="category")
    return figure


def build_activity_figure(frame: pd.DataFrame) -> go.Figure:
    """Plot interaction volume against conflict rate over time.

    Parameters
    ----------
    frame : pandas.DataFrame
        Per-period statistics carrying ``period``, ``edges`` and
        ``conflict_rate`` columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Dual-axis figure.
    """
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=frame["period"], y=frame["edges"],
        name="Interactions",
        marker_color=ACCENT, opacity=0.65,
        hovertemplate="%{x}: %{y:,} interactions<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=frame["period"], y=frame["conflict_rate"],
        name="Conflict rate", yaxis="y2",
        line={"color": NEGATIVE, "width": 2},
        marker={"size": 7, "color": NEGATIVE},
        mode="lines+markers",
        hovertemplate="%{x}: %{y:.1%} conflictual<extra></extra>",
    ))

    apply_chart_layout(figure, height=340, title="Network activity and conflict rate")
    figure.update_xaxes(type="category")
    figure.update_layout(
        yaxis={"title": {"text": "Interactions", "font": {"color": MUTED, "size": 11}}},
        yaxis2={
            "title": {"text": "Conflict rate", "font": {"color": MUTED, "size": 11}},
            "overlaying": "y", "side": "right", "tickformat": ".0%",
            "showgrid": False, "zeroline": False,
            "tickfont": {"color": MUTED, "size": 10},
        },
    )
    return figure


def build_stat_rows(stats: NetworkStats) -> list[tuple[str, str]]:
    """Format global statistics as label and value pairs.

    Parameters
    ----------
    stats : NetworkStats
        Global topology statistics.

    Returns
    -------
    list of tuple of (str, str)
        Display-ready rows, excluding the community assignment.
    """
    rows: list[tuple[str, str]] = []
    for key, value in stats.items():
        if key == "community_map":
            continue
        label = key.replace("_", " ").title()
        rendered: Any = f"{value:.4f}" if isinstance(value, float) else f"{value:,}"
        rows.append((label, str(rendered)))
    return rows
