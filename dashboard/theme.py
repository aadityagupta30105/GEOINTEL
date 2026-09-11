"""
Dashboard Theme
===============
Single source of truth for the dashboard palette, typography and Plotly
layout defaults. Keeping these values in one module guarantees that every
figure, card and table renders against the same surface colours.

Identity
--------
The surface follows the Automobili Lamborghini visual language: an unbroken
black ground, a single gold accent, and hard geometry. Chroma is rationed on
purpose. Gold carries emphasis, the tone pair carries cooperation and
conflict, and nothing else is coloured at all, so a red edge on the network
map reads as a finding rather than as decoration.

Geometry is the second carrier of the identity. Panels are cut at 45 degrees
at two opposing corners rather than rounded, section titles are marked with a
hexagon, and rules are hairlines. Nothing in the layer is decorative for its
own sake: the clipped corner is what distinguishes a data panel from the
ground, and the hexagon is what distinguishes a section from a paragraph.

Palette
-------
``BACKGROUND``  Pure black ground.
``SURFACE``     Panel colour for cards and the sidebar.
``BORDER``      Hairline separators.
``ACCENT``      Oro Elios gold. Primary emphasis and the primary series.
``ACCENT_ALT``  Giallo. Reserved for polarization and alert states.
``POSITIVE``    Cooperation.
``NEGATIVE``    Rosso. Conflict.
``CAUTION``     Mixed or neutral posture.

Typography
----------
``Chakra Petch`` for display, chosen for its cut corners and geometric axis,
set uppercase with wide tracking. ``Barlow`` for running text. ``JetBrains
Mono`` for figures, so that digits align in columns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import plotly.graph_objects as go

__all__ = [
    "BACKGROUND",
    "SURFACE",
    "SURFACE_RAISED",
    "BORDER",
    "ACCENT",
    "ACCENT_ALT",
    "POSITIVE",
    "NEGATIVE",
    "CAUTION",
    "TEXT",
    "MUTED",
    "BLOC_COLORS",
    "FONT_DISPLAY",
    "FONT_BODY",
    "FONT_MONO",
    "FIGURE_DISPLAY_FONT",
    "FIGURE_LABEL_FONT",
    "NODE_COLORSCALE",
    "NODE_CONFLICT_COLORSCALE",
    "PANEL_CLIP",
    "GLOBAL_CSS",
    "apply_chart_layout",
    "apply_geo_layout",
    "tone_color",
    "rgba",
    "overlay",
    "sequential_gradient",
    "diverging_gradient",
]

BACKGROUND: Final[str] = "#000000"
SURFACE: Final[str] = "#0b0b0b"
SURFACE_RAISED: Final[str] = "#141414"
BORDER: Final[str] = "#262626"
ACCENT: Final[str] = "#c9a227"
ACCENT_ALT: Final[str] = "#ffce00"
POSITIVE: Final[str] = "#3fbf7f"
NEGATIVE: Final[str] = "#d92b2b"
CAUTION: Final[str] = "#c9a227"
TEXT: Final[str] = "#f2f2f2"
MUTED: Final[str] = "#8a8a8a"

# Ordered categorical palette for bloc and community assignment. Gold leads,
# then hues held far enough apart to stay separable against black.
BLOC_COLORS: Final[tuple[str, ...]] = (
    ACCENT, "#e8e8e8", POSITIVE, NEGATIVE, "#7a8fa6", "#8c6fb0",
)

# Font stacks. Declared once so that CSS and Plotly cannot disagree.
FONT_DISPLAY: Final[str] = "'Chakra Petch', 'Segoe UI', sans-serif"
FONT_BODY: Final[str] = "'Barlow', 'Segoe UI', sans-serif"
FONT_MONO: Final[str] = "'JetBrains Mono', 'Consolas', monospace"

# Plotly parses its own font stacks and rejects the quoting CSS requires, so
# the same families are restated unquoted. Chart labels are set in the mono
# face: axis ticks and hover readouts are figures, and figures should align.
FIGURE_DISPLAY_FONT: Final[str] = "Chakra Petch, Segoe UI, sans-serif"
FIGURE_LABEL_FONT: Final[str] = "JetBrains Mono, Consolas, monospace"

# Continuous colour scales, built from the palette rather than taken from
# Plotly's defaults. Viridis and RdYlGn carry their own identity and would be
# the only place on the surface where colour is not one of ours.
#
# Sequential: black through the gold ramp, for a magnitude with no meaningful
# midpoint such as PageRank.
NODE_COLORSCALE: Final[tuple[tuple[float, str], ...]] = (
    (0.00, "#241d09"),
    (0.35, "#6b5615"),
    (0.70, ACCENT),
    (1.00, "#f2dd93"),
)

# Diverging: cooperation through neutral gold to conflict, for a ratio whose
# midpoint is the story.
NODE_CONFLICT_COLORSCALE: Final[tuple[tuple[float, str], ...]] = (
    (0.00, POSITIVE),
    (0.50, ACCENT),
    (1.00, NEGATIVE),
)

# Map surface colours. The map is monochrome by design so that the only
# chroma on it is the tone of the edges drawn over it.
#
# These sit deliberately above the panel colours rather than at them. A
# landmass rendered at the panel value disappears into a black ocean: the
# geometry is drawn, and nothing is legible. Land has to read as a plate, and
# the two line weights have to separate a national border from a coastline,
# which needs more separation than the flat surfaces do.
_LAND: Final[str] = "#232323"
_OCEAN: Final[str] = "#000000"
_COUNTRY_LINE: Final[str] = "#3a3a3a"
_COASTLINE: Final[str] = "#5c5c5c"

# Tone magnitude at which a relationship reads as clearly directional.
_TONE_EPSILON: Final[float] = 0.05

# Corner cut applied to panels, in pixels. Two opposing corners only: cutting
# all four reads as a label, cutting two reads as a machined part. Exported so
# that inline styles composed outside this module cut the same corners.
_CUT: Final[str] = "10px"
PANEL_CLIP: Final[str] = (
    f"polygon({_CUT} 0, 100% 0, 100% calc(100% - {_CUT}), "
    f"calc(100% - {_CUT}) 100%, 0 100%, 0 {_CUT})"
)


GLOBAL_CSS: Final[str] = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=Barlow:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

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
    --font-display: {FONT_DISPLAY};
    --font-body: {FONT_BODY};
    --font-mono: {FONT_MONO};
    --panel-clip: {PANEL_CLIP};
  }}

  .stApp {{ background: var(--bg); color: var(--text); }}
  .stApp a {{ color: var(--accent); }}

  h1, h2, h3, h4 {{
    font-family: var(--font-display) !important;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--text);
  }}
  body, p, span, div, label, li {{
    font-family: var(--font-body) !important;
    font-size: 0.86rem;
  }}
  code, pre, .stCode {{ font-family: var(--font-mono) !important; }}

  /* Streamlit draws its icons as ligatures in Material Symbols: the sidebar
     collapse control, expander chevrons, the alert glyphs. The blanket span
     rule above is stronger than Streamlit's own, so without this the
     ligatures never form and every icon renders as its literal name -
     "keyboard_double_arrow_right" in place of an arrow. */
  [data-testid="stIconMaterial"],
  span[class*="material-symbols"],
  span[class*="material-icons"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
  }}

  /* Masthead: the marque sets the type, the hairline sets the register. */
  .masthead {{
    display: flex;
    align-items: baseline;
    gap: 16px;
    border-bottom: 1px solid var(--accent);
    padding-bottom: 14px;
    margin-bottom: 6px;
  }}
  .masthead-mark {{
    font-family: var(--font-display) !important;
    font-size: 2.0rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: var(--text);
    line-height: 1;
    text-transform: uppercase;
  }}
  .masthead-rule {{
    color: var(--accent);
    font-size: 1.5rem;
    line-height: 1;
  }}
  .masthead-sub {{
    color: var(--muted);
    font-family: var(--font-display) !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.28em;
  }}

  /* Section headings, marked with a hexagon rather than a bar. */
  .section-title {{
    font-family: var(--font-display) !important;
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 10px 0 12px 0;
  }}
  .section-title::before {{
    content: '';
    width: 9px;
    height: 10px;
    background: var(--accent);
    clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
    flex: none;
  }}
  .section-note {{
    color: var(--muted);
    font-size: 0.76rem;
    line-height: 1.6;
    margin: -6px 0 14px 18px;
  }}

  /* Metric cards */
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    clip-path: var(--panel-clip);
    padding: 15px 17px 16px 17px;
    margin: 4px 0;
  }}
  .metric-value {{
    font-family: var(--font-display) !important;
    /* Six cards share the strip. A fixed size overflows the narrowest of
       them and wraps a figure across two lines, which reads as two figures;
       the clamp lets the type shrink instead. */
    font-size: clamp(1.05rem, 2.1vw, 1.6rem);
    font-weight: 700;
    letter-spacing: 0.01em;
    color: var(--accent);
    line-height: 1.15;
    /* Streamlit sets overflow-wrap: break-word on its containers, which
       splits a figure across two lines mid-digit and turns one number into
       what looks like two. A metric is a single token; it never wraps. */
    white-space: nowrap;
    overflow-wrap: normal;
    word-break: keep-all;
    font-variant-numeric: tabular-nums;
  }}
  .metric-label {{
    color: var(--muted);
    font-family: var(--font-display) !important;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    line-height: 1.35;
    margin-top: 6px;
    overflow-wrap: normal;
    word-break: keep-all;
    hyphens: none;
  }}

  /* Narrative panel */
  .narrative-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    clip-path: var(--panel-clip);
    padding: 20px 24px;
    line-height: 1.8;
    color: var(--text);
  }}

  /* Key-value readout rows */
  .kv-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 16px;
    padding: 8px 2px;
    border-bottom: 1px solid var(--border);
  }}
  .kv-key {{
    color: var(--muted);
    font-family: var(--font-display) !important;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
  }}
  .kv-value {{
    color: var(--accent);
    font-family: var(--font-mono) !important;
    font-weight: 500;
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
  }}

  /* Tags */
  .tag {{
    display: inline-block;
    background: rgba(201, 162, 39, 0.08);
    border: 1px solid rgba(201, 162, 39, 0.32);
    color: var(--accent);
    font-family: var(--font-mono) !important;
    padding: 1px 8px;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    margin: 2px;
  }}
  .tag-red {{
    background: rgba(217, 43, 43, 0.09);
    border-color: rgba(217, 43, 43, 0.34);
    color: var(--negative);
  }}
  .tag-green {{
    background: rgba(63, 191, 127, 0.09);
    border-color: rgba(63, 191, 127, 0.34);
    color: var(--positive);
  }}

  /* Status strip */
  .status-line {{
    color: var(--muted);
    font-family: var(--font-mono) !important;
    font-size: 0.72rem;
    letter-spacing: 0.02em;
    line-height: 1.6;
  }}

  [data-testid="stSidebar"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }}
  [data-testid="stSidebar"] hr {{ border-color: var(--border); }}

  /* Inputs sit on the raised surface with a gold focus ring. */
  .stSelectbox > div > div,
  .stMultiSelect > div > div,
  .stTextInput > div > div,
  .stTextArea > div > div,
  .stDateInput > div > div,
  .stNumberInput > div > div {{
    background: var(--surface-raised) !important;
    border-color: var(--border) !important;
    border-radius: 0 !important;
  }}
  .stSelectbox > div > div:focus-within,
  .stTextArea > div > div:focus-within,
  .stTextInput > div > div:focus-within {{
    border-color: var(--accent) !important;
  }}

  .stButton > button {{
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 0;
    font-family: var(--font-display) !important;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    transition: border-color 120ms ease, color 120ms ease;
  }}
  .stButton > button:hover {{
    border-color: var(--accent);
    color: var(--accent);
  }}
  .stButton > button[kind="primary"] {{
    background: var(--accent);
    border-color: var(--accent);
    color: #000000;
    font-weight: 600;
  }}
  .stButton > button[kind="primary"]:hover {{
    background: var(--accent-alt);
    border-color: var(--accent-alt);
    color: #000000;
  }}

  .stDownloadButton > button {{
    border-radius: 0;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text);
    font-family: var(--font-display) !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
  }}
  .stDownloadButton > button:hover {{
    border-color: var(--accent);
    color: var(--accent);
  }}

  div[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0;
    padding: 13px;
  }}

  /* Sliders and toggles adopt the accent rather than the Streamlit default. */
  .stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: var(--accent) !important;
  }}
  [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {{
    background: var(--accent) !important;
  }}

  .stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid var(--border);
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: var(--font-display) !important;
    font-size: 0.73rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color: var(--text); }}
  .stTabs [aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent);
  }}
  .stTabs [data-baseweb="tab-highlight"] {{ background: transparent; }}

  /* Expanders and dataframes squared off to match the panels. */
  .streamlit-expanderHeader, [data-testid="stExpander"] details {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0;
  }}
  [data-testid="stExpander"] summary {{
    font-family: var(--font-display) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.74rem;
  }}
  [data-testid="stDataFrame"] {{ border: 1px solid var(--border); }}

  hr {{ border-color: var(--border); }}
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
        font={"color": TEXT, "family": FIGURE_LABEL_FONT, "size": 11},
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
            "font": {"color": TEXT, "family": FIGURE_LABEL_FONT, "size": 11},
        },
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    if title:
        fig.update_layout(
            title={
                "text": title,
                "font": {"color": ACCENT, "family": FIGURE_DISPLAY_FONT, "size": 14},
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
        font={"color": TEXT, "family": FIGURE_LABEL_FONT, "size": 11},
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": BORDER,
            "font": {"color": TEXT, "family": FIGURE_LABEL_FONT, "size": 11},
        },
        legend={
            "orientation": "h", "y": -0.02, "x": 0.5, "xanchor": "center",
            "font": {"color": TEXT, "size": 11},
            "bgcolor": overlay(SURFACE, 0.85),
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


def overlay(hex_color: str = SURFACE, alpha: float = 0.85) -> str:
    """Return a translucent panel colour for use over a figure.

    Legends and colour bars are drawn on top of the plotting surface and need
    a backing that reads as a panel without hiding what is beneath it. Routing
    them through this helper rather than a literal keeps them tied to the
    palette.

    Parameters
    ----------
    hex_color : str, optional
        Panel colour as ``#rrggbb``.
    alpha : float, optional
        Opacity in ``[0, 1]``.

    Returns
    -------
    str
        CSS ``rgba()`` colour.
    """
    return rgba(hex_color, alpha)


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
