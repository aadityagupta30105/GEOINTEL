"""
Dashboard Theme
===============
Single source of truth for the dashboard palette, typography and Plotly
layout defaults. Keeping these values in one module guarantees that every
figure, card and table renders against the same surface colours.

Palette
-------
``BACKGROUND``  Deep midnight base.
``SURFACE``     Raised panel colour for cards and the sidebar.
``BORDER``      Hairline separators.
``ACCENT``      Primary technical cyan, used for emphasis and primary series.
``ACCENT_ALT``  Secondary amber, reserved for polarization indicators.
``POSITIVE``    Cooperation.
``NEGATIVE``    Conflict.
``CAUTION``     Mixed or neutral posture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import plotly.graph_objects as go

__all__ = [
    "BACKGROUND",
    "SURFACE",
    "BORDER",
    "ACCENT",
    "ACCENT_ALT",
    "POSITIVE",
    "NEGATIVE",
    "CAUTION",
    "TEXT",
    "MUTED",
    "BLOC_COLORS",
    "GLOBAL_CSS",
    "apply_chart_layout",
    "apply_geo_layout",
    "tone_color",
    "rgba",
    "sequential_gradient",
    "diverging_gradient",
]

BACKGROUND: Final[str] = "#0a0e1a"
SURFACE: Final[str] = "#111827"
SURFACE_RAISED: Final[str] = "#151d2e"
BORDER: Final[str] = "#1e293b"
ACCENT: Final[str] = "#00d4ff"
ACCENT_ALT: Final[str] = "#ff6b35"
POSITIVE: Final[str] = "#22c55e"
NEGATIVE: Final[str] = "#ef4444"
CAUTION: Final[str] = "#f59e0b"
TEXT: Final[str] = "#e2e8f0"
MUTED: Final[str] = "#64748b"

# Ordered categorical palette for bloc and community assignment.
BLOC_COLORS: Final[tuple[str, ...]] = (
    ACCENT, ACCENT_ALT, POSITIVE, CAUTION, "#a78bfa", "#ec4899",
)

# Map surface colours, kept distinct from panel colours for legibility.
_LAND: Final[str] = "#1a2235"
_OCEAN: Final[str] = "#0d1421"
_COUNTRY_LINE: Final[str] = "#253248"
_COASTLINE: Final[str] = "#2d3f5a"

# Tone magnitude at which a relationship reads as clearly directional.
_TONE_EPSILON: Final[float] = 0.05


GLOBAL_CSS: Final[str] = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

  :root {{
    --bg: {BACKGROUND};
    --surface: {SURFACE};
    --surface-raised: {SURFACE_RAISED};
    --border: {BORDER};
    --accent: {ACCENT};
    --accent-alt: {ACCENT_ALT};
    --positive: {POSITIVE};
    --negative: {NEGATIVE};
    --caution: {CAUTION};
    --text: {TEXT};
    --muted: {MUTED};
  }}

  .stApp {{ background: var(--bg); color: var(--text); }}

  h1, h2, h3, h4 {{
    font-family: 'Syne', 'Segoe UI', sans-serif !important;
    letter-spacing: -0.015em;
  }}
  body, p, span, div, label {{
    font-family: 'Space Mono', 'Consolas', monospace !important;
    font-size: 0.85rem;
  }}

  /* Masthead */
  .masthead {{
    display: flex;
    align-items: baseline;
    gap: 14px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
  }}
  .masthead-mark {{
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
  }}
  .masthead-rule {{
    color: var(--border);
    font-size: 1.4rem;
    line-height: 1;
  }}
  .masthead-sub {{
    color: var(--muted);
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
  }}

  /* Section headings */
  .section-title {{
    font-family: 'Syne', sans-serif !important;
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 10px;
    margin: 6px 0 12px 0;
  }}
  .section-note {{
    color: var(--muted);
    font-size: 0.75rem;
    margin: -6px 0 12px 13px;
  }}

  /* Metric cards */
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 14px 16px;
    margin: 4px 0;
  }}
  .metric-value {{
    font-family: 'Syne', sans-serif !important;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.15;
  }}
  .metric-label {{
    color: var(--muted);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 2px;
  }}

  /* Narrative panel */
  .narrative-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent-alt);
    border-radius: 4px;
    padding: 18px 20px;
    line-height: 1.75;
    color: var(--text);
  }}

  /* Key-value readout rows */
  .kv-row {{
    display: flex;
    justify-content: space-between;
    padding: 7px 2px;
    border-bottom: 1px solid var(--border);
  }}
  .kv-key {{
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  .kv-value {{ color: var(--accent); font-weight: 700; }}

  /* Tags */
  .tag {{
    display: inline-block;
    background: rgba(0, 212, 255, 0.10);
    border: 1px solid rgba(0, 212, 255, 0.28);
    color: var(--accent);
    border-radius: 2px;
    padding: 1px 7px;
    font-size: 0.71rem;
    margin: 2px;
  }}
  .tag-red {{
    background: rgba(239, 68, 68, 0.10);
    border-color: rgba(239, 68, 68, 0.28);
    color: var(--negative);
  }}
  .tag-green {{
    background: rgba(34, 197, 94, 0.10);
    border-color: rgba(34, 197, 94, 0.28);
    color: var(--positive);
  }}

  /* Status strip */
  .status-line {{
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.04em;
  }}

  [data-testid="stSidebar"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }}
  .stSelectbox > div > div {{ background: var(--surface-raised) !important; }}

  div[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px;
  }}

  .stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    border-bottom: 1px solid var(--border);
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.76rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}
</style>
"""


def apply_chart_layout(
    fig: go.Figure,
    height: int = 320,
    title: str | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Apply the platform layout to a Cartesian Plotly figure.

    Removes gridlines, draws explicit axis lines, and makes the plotting
    surface transparent so the figure sits flush on the application
    background.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to restyle, modified in place.
    height : int, optional
        Figure height in pixels.
    title : str or None, optional
        Chart title. Rendered in the accent colour when supplied.
    show_legend : bool, optional
        Whether to display the legend.

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, for chaining.
    """
    axis_style: dict[str, Any] = {
        "showgrid": False,
        "zeroline": False,
        "showline": True,
        "linecolor": BORDER,
        "linewidth": 1,
        "ticks": "outside",
        "tickcolor": BORDER,
        "tickfont": {"color": MUTED, "size": 10},
        "title": {"font": {"color": MUTED, "size": 11}},
    }

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Space Mono, monospace", "size": 11},
        margin={"l": 8, "r": 8, "t": 44 if title else 12, "b": 8},
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "y": -0.18,
            "font": {"color": MUTED, "size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": BORDER,
            "font": {"color": TEXT, "family": "Space Mono, monospace", "size": 11},
        },
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    if title:
        fig.update_layout(
            title={
                "text": title,
                "font": {"color": ACCENT, "family": "Syne, sans-serif", "size": 14},
                "x": 0,
                "xanchor": "left",
            }
        )
    return fig


def apply_geo_layout(fig: go.Figure, height: int = 580) -> go.Figure:
    """Apply the platform layout to a geographic Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure carrying ``Scattergeo`` traces, modified in place.
    height : int, optional
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, for chaining.
    """
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor=_LAND,
        showocean=True, oceancolor=_OCEAN,
        showlakes=True, lakecolor=_OCEAN,
        showrivers=False,
        showcountries=True, countrycolor=_COUNTRY_LINE, countrywidth=0.5,
        showcoastlines=True, coastlinecolor=_COASTLINE, coastlinewidth=0.6,
        showframe=False,
        bgcolor=BACKGROUND,
    )
    fig.update_layout(
        height=height,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        geo={"bgcolor": BACKGROUND},
        font={"color": TEXT, "family": "Space Mono, monospace", "size": 11},
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": BORDER,
            "font": {"color": TEXT, "family": "Space Mono, monospace", "size": 11},
        },
        legend={
            "orientation": "h", "y": -0.02, "x": 0.5, "xanchor": "center",
            "font": {"color": TEXT, "size": 11},
            "bgcolor": "rgba(17,24,39,0.85)",
            "bordercolor": _COUNTRY_LINE, "borderwidth": 1,
        },
    )
    return fig


def tone_color(tone: float) -> str:
    """Map a normalised tone value onto the status palette.

    Parameters
    ----------
    tone : float
        Normalised tone in ``[-1, 1]``.

    Returns
    -------
    str
        Hex colour: positive, negative or caution.
    """
    if tone > _TONE_EPSILON:
        return POSITIVE
    if tone < -_TONE_EPSILON:
        return NEGATIVE
    return CAUTION


# --- Table shading ----------------------------------------------------------
#
# Pandas ships ``Styler.background_gradient``, but it requires matplotlib and
# renders in matplotlib colour maps that sit outside this palette. The helpers
# below produce the same effect from the platform colours alone, so table
# shading carries no plotting dependency and matches the rest of the surface.

_CELL_MIN_ALPHA: Final[float] = 0.04
_CELL_MAX_ALPHA: Final[float] = 0.52


def rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex colour to an ``rgba()`` string.

    Parameters
    ----------
    hex_color : str
        Colour as ``#rrggbb``.
    alpha : float
        Opacity in ``[0, 1]``.

    Returns
    -------
    str
        CSS ``rgba()`` colour.
    """
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return f"rgba({red},{green},{blue},{alpha:.3f})"


def _cell_style(hex_color: str, intensity: float) -> str:
    """Build the CSS declaration for one shaded table cell.

    Parameters
    ----------
    hex_color : str
        Base colour of the ramp.
    intensity : float
        Ramp position in ``[0, 1]``.

    Returns
    -------
    str
        CSS declarations for the cell.
    """
    intensity = min(1.0, max(0.0, intensity))
    alpha = _CELL_MIN_ALPHA + intensity * (_CELL_MAX_ALPHA - _CELL_MIN_ALPHA)
    return f"background-color: {rgba(hex_color, alpha)}; color: {TEXT};"


def sequential_gradient(
    values: Sequence[float],
    hex_color: str = ACCENT,
) -> list[str]:
    """Shade a column by rank within its own range.

    The lowest value receives the faintest wash and the highest the strongest,
    which reproduces the intent of a sequential colour map without importing
    one. A constant column is shaded uniformly at the midpoint.

    Parameters
    ----------
    values : sequence of float
        Column values.
    hex_color : str, optional
        Base colour of the ramp.

    Returns
    -------
    list of str
        One CSS declaration per input value.
    """
    numeric = [
        float(value) if value is not None and math.isfinite(float(value)) else math.nan
        for value in values
    ]
    finite = [value for value in numeric if not math.isnan(value)]

    if not finite:
        return ["" for _ in numeric]

    low, high = min(finite), max(finite)
    span = high - low

    return [
        ""
        if math.isnan(value)
        else _cell_style(hex_color, 0.5 if span == 0 else (value - low) / span)
        for value in numeric
    ]


def diverging_gradient(
    values: Sequence[float],
    center: float = 0.0,
    negative_color: str = NEGATIVE,
    positive_color: str = POSITIVE,
) -> list[str]:
    """Shade a column outward from a neutral centre.

    Values below ``center`` shade toward the conflict colour and values above
    it toward the cooperation colour, with intensity proportional to distance
    from the centre.

    Parameters
    ----------
    values : sequence of float
        Column values.
    center : float, optional
        Neutral point of the scale.
    negative_color, positive_color : str, optional
        Ramp endpoints.

    Returns
    -------
    list of str
        One CSS declaration per input value.
    """
    numeric = [
        float(value) if value is not None and math.isfinite(float(value)) else math.nan
        for value in values
    ]
    finite = [value for value in numeric if not math.isnan(value)]

    if not finite:
        return ["" for _ in numeric]

    extent = max((abs(value - center) for value in finite), default=0.0)

    styles: list[str] = []
    for value in numeric:
        if math.isnan(value):
            styles.append("")
            continue
        offset = value - center
        intensity = 0.0 if extent == 0 else abs(offset) / extent
        styles.append(
            _cell_style(positive_color if offset >= 0 else negative_color, intensity)
        )
    return styles
