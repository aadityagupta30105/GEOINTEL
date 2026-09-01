"""
Geopolitical Intelligence Dashboard
===================================
Streamlit application shell: data acquisition, caching, filter state and page
layout. Figure construction lives in :mod:`dashboard.figures`, styling in
:mod:`dashboard.theme`, reference geography in :mod:`dashboard.geodata` and
bloc analysis in :mod:`dashboard.blocs`.

Caching contract
----------------
``@st.cache_data`` memoises frames and derived tables keyed by scalar
arguments. ``@st.cache_resource`` memoises graph objects, which are mutable
and must not be copied per session. Graph builders receive the event frame as
an underscore-prefixed argument (excluded from hashing) alongside an explicit
content fingerprint, so a filter change invalidates the entry without
serialising the frame on every rerun.

Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import networkx as nx
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.graph_builder import (  # noqa: E402
    NetworkStats,
    build_graph,
    build_temporal_graphs,
    compute_metrics,
    compute_network_stats,
)
from analysis.narrator import GeopoliticalNarrator  # noqa: E402
from dashboard.blocs import POLE_PRESETS, compute_blocs  # noqa: E402
from dashboard.figures import (  # noqa: E402
    build_activity_figure,
    build_bilateral_chart,
    build_bloc_figure,
    build_network_figure,
    build_pagerank_bar,
    build_radar_chart,
    build_stat_rows,
    build_temporal_line,
    build_tone_heatmap,
)
from dashboard.theme import (  # noqa: E402
    ACCENT,
    ACCENT_ALT,
    BLOC_COLORS,
    GLOBAL_CSS,
    NEGATIVE,
    POSITIVE,
    diverging_gradient,
    sequential_gradient,
    tone_color,
)
from utils.logging_config import OK, WARN, get_logger  # noqa: E402

_log = get_logger(__name__)

OUTPUT_DIR: Final[Path] = _ROOT / "output"
EVENTS_CSV: Final[Path] = OUTPUT_DIR / "events_clean.csv"

# Columns the pipeline export must provide before it can be loaded.
_REQUIRED_EXPORT_COLUMNS: Final[frozenset[str]] = frozenset({
    "Actor1CountryCode", "Actor2CountryCode", "tone_norm", "event_type", "date",
})

_SOURCE_LABELS: Final[dict[str, str]] = {
    "pipeline_output": "Pipeline export",
    "gdelt_live": "GDELT live fetch",
    "mock": "Synthetic simulation",
}

_GDELT_LIVE_MAX_DAYS: Final[int] = 7


st.set_page_config(
    page_title="GeoIntel - Geopolitical Network Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# --- Data acquisition -------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=300)
def load_data(
    source: str,
    n_events: int = 5000,
    days_back: int = 90,
    gdelt_start: str = "",
    gdelt_end: str = "",
) -> tuple[pd.DataFrame, str]:
    """Load the event stream from the selected source.

    Sources
    -------
    ``pipeline_output``
        Reads ``output/events_clean.csv`` written by ``main.py``. This is the
        authoritative source and reflects whatever the last run collected.
    ``gdelt_live``
        Fetches GDELT daily exports for the requested window.
    ``mock``
        Generates synthetic events for offline operation.

    Every path is exception-guarded: a failure yields an empty frame and an
    explanatory label rather than raising into the render loop.

    Parameters
    ----------
    source : {"pipeline_output", "gdelt_live", "mock"}
        Selected source.
    n_events : int, optional
        Synthetic event count.
    days_back : int, optional
        Window length for synthetic generation.
    gdelt_start, gdelt_end : str, optional
        Explicit GDELT window as ``YYYY-MM-DD``.

    Returns
    -------
    tuple of (pandas.DataFrame, str)
        The event frame and a human-readable provenance label.
    """
    from data.gdelt_collector import collect_gdelt_range, generate_mock_data, preprocess

    if source == "pipeline_output":
        if not EVENTS_CSV.exists():
            return pd.DataFrame(), "Pipeline export not found - run main.py first"
        try:
            frame = pd.read_csv(EVENTS_CSV)
        except (OSError, pd.errors.ParserError) as exc:
            _log.error("Failed to read %s: %s", EVENTS_CSV, exc)
            return pd.DataFrame(), f"Pipeline export unreadable - {exc}"

        missing = _REQUIRED_EXPORT_COLUMNS - set(frame.columns)
        if missing:
            return pd.DataFrame(), (
                f"Pipeline export missing columns: {', '.join(sorted(missing))}"
            )
        return frame, f"Pipeline export - {len(frame):,} events"

    if source == "gdelt_live":
        try:
            if gdelt_start and gdelt_end:
                start = datetime.strptime(gdelt_start, "%Y-%m-%d")
                end = datetime.strptime(gdelt_end, "%Y-%m-%d")
            else:
                end = datetime.now()
                start = end - timedelta(days=min(days_back, _GDELT_LIVE_MAX_DAYS))

            frame = preprocess(collect_gdelt_range(start, end, target_rows_per_day=2000))
            if not frame.empty:
                return frame, (
                    f"GDELT live - {len(frame):,} events, "
                    f"{start.date()} to {end.date()}"
                )
            return pd.DataFrame(), "GDELT live - no bilateral events in window"
        except Exception as exc:  # Network and parse failures must not crash the UI.
            _log.error("GDELT live fetch failed: %s", exc)
            return pd.DataFrame(), f"GDELT live fetch failed - {exc}"

    end = datetime.now()
    start = end - timedelta(days=days_back)
    frame = preprocess(generate_mock_data(start, end, n_events))
    return frame, f"Synthetic simulation - {len(frame):,} events"


def fingerprint(frame: pd.DataFrame) -> str:
    """Compute a content fingerprint for cache invalidation.

    Hashing the frame content is materially cheaper than the JSON round-trip
    it replaces, and unlike a length-based key it detects filter changes that
    preserve row count.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame to fingerprint.

    Returns
    -------
    str
        Stable hexadecimal digest of the frame contents.
    """
    if frame.empty:
        return "empty"
    digest = int(pd.util.hash_pandas_object(frame, index=True).sum())
    return f"{digest & 0xFFFFFFFFFFFFFFFF:016x}-{len(frame)}"


@st.cache_resource(show_spinner=False)
def load_graph(
    _events: pd.DataFrame,
    cache_key: str,
) -> tuple[nx.DiGraph, pd.DataFrame, NetworkStats]:
    """Build the static graph and its analytics for a filtered event frame.

    Parameters
    ----------
    _events : pandas.DataFrame
        Filtered event frame. Excluded from the cache key by the underscore
        prefix; ``cache_key`` carries the identity instead.
    cache_key : str
        Fingerprint produced by :func:`fingerprint`.

    Returns
    -------
    tuple of (networkx.DiGraph, pandas.DataFrame, NetworkStats)
        Graph, node metrics and global statistics.
    """
    graph = build_graph(_events)
    return graph, compute_metrics(graph), compute_network_stats(graph)


@st.cache_resource(show_spinner=False)
def load_temporal(
    _events: pd.DataFrame,
    cache_key: str,
    period: str = "month",
) -> dict[str, nx.DiGraph]:
    """Build temporal snapshot graphs for a filtered event frame.

    Parameters
    ----------
    _events : pandas.DataFrame
        Filtered event frame, excluded from the cache key.
    cache_key : str
        Fingerprint produced by :func:`fingerprint`.
    period : {"month", "quarter", "year"}, optional
        Temporal granularity.

    Returns
    -------
    dict of str to networkx.DiGraph
        Snapshot graphs keyed by period label.
    """
    return build_temporal_graphs(_events, period=period)


@st.cache_data(show_spinner=False)
def temporal_statistics(_temporal: dict[str, nx.DiGraph], cache_key: str) -> pd.DataFrame:
    """Reduce temporal snapshots to a per-period statistics frame.

    Only global statistics are computed. Node-level centralities are not
    required by any temporal view and computing them per period dominates the
    render cost.

    Parameters
    ----------
    _temporal : dict of str to networkx.DiGraph
        Snapshot graphs, excluded from the cache key.
    cache_key : str
        Fingerprint identifying the underlying event frame.

    Returns
    -------
    pandas.DataFrame
        One row per period, sorted chronologically.
    """
    records = []
    for period in sorted(_temporal):
        snapshot = _temporal[period]
        stats = compute_network_stats(snapshot)
        records.append({
            "period": period,
            "nodes": stats["nodes"],
            "edges": stats["edges"],
            "avg_tone": stats["avg_tone"],
            "conflict_rate": stats["negative_edge_ratio"],
            "ggpi": stats["ggpi"],
            "modularity": stats["modularity"],
        })
    return pd.DataFrame(records)


@st.cache_resource(show_spinner=False)
def get_narrator(provider: str = "auto") -> GeopoliticalNarrator:
    """Return the shared narrator instance.

    Cached as a resource so the per-instance summary cache survives reruns and
    provider calls are not repeated on every interaction.

    Parameters
    ----------
    provider : str, optional
        Requested narrative provider.

    Returns
    -------
    GeopoliticalNarrator
        Shared narrator.
    """
    return GeopoliticalNarrator(provider=provider)  # type: ignore[arg-type]


# --- Rendering helpers ------------------------------------------------------

def masthead() -> None:
    """Render the application masthead."""
    st.markdown(
        "<div class='masthead'>"
        "<span class='masthead-mark'>GEOINTEL</span>"
        "<span class='masthead-rule'>/</span>"
        "<span class='masthead-sub'>Geopolitical Relationship Network Analysis</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def section_title(title: str, note: str | None = None) -> None:
    """Render a section heading with an optional explanatory note.

    Parameters
    ----------
    title : str
        Heading text.
    note : str or None, optional
        Secondary line rendered beneath the heading.
    """
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if note:
        st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)


def metric_card(container: object, value: str, label: str, color: str = ACCENT) -> None:
    """Render a single KPI card.

    Parameters
    ----------
    container : object
        Streamlit container exposing ``markdown``.
    value : str
        Pre-formatted metric value.
    label : str
        Metric name, rendered upper case by the stylesheet.
    color : str, optional
        Value colour.
    """
    container.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value' style='color:{color}'>{value}</div>"
        f"<div class='metric-label'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[pd.DataFrame, str, str, bool]:
    """Render the sidebar and resolve the active event frame.

    Returns
    -------
    tuple of (pandas.DataFrame, str, str, bool)
        The loaded frame, its provenance label, the selected event-type filter
        and whether narrative generation is enabled.
    """
    with st.sidebar:
        section_title("Data source")

        # Prefer the pipeline export, but only when it exists. On a hosted
        # deployment the output directory is not committed, so defaulting to
        # it would halt the app on first load; synthetic data keeps the
        # dashboard usable with no prior pipeline run.
        source_options = ["pipeline_output", "gdelt_live", "mock"]
        default_source = "pipeline_output" if EVENTS_CSV.exists() else "mock"

        source = st.radio(
            "Source",
            options=source_options,
            format_func=lambda key: _SOURCE_LABELS[key],
            index=source_options.index(default_source),
            label_visibility="collapsed",
            help="The pipeline export reads output/events_clean.csv written by main.py",
            key="source",
        )

        n_events, days_back = 5000, 90
        gdelt_start = gdelt_end = ""

        if source == "mock":
            n_events = st.slider("Events to simulate", 1000, 10000, 5000, 500)
            days_back = st.slider("Window (days)", 30, 365, 90, 30)
        elif source == "gdelt_live":
            col_from, col_to = st.columns(2)
            with col_from:
                gdelt_start = st.date_input(
                    "From",
                    value=datetime.now().date() - timedelta(days=_GDELT_LIVE_MAX_DAYS),
                    key="gdelt_from",
                ).strftime("%Y-%m-%d")
            with col_to:
                gdelt_end = st.date_input(
                    "To", value=datetime.now().date(), key="gdelt_to"
                ).strftime("%Y-%m-%d")
            st.markdown(
                "<div class='status-line'>[WARN] Full daily exports are retrieved. "
                "Allow roughly five seconds per day.</div>",
                unsafe_allow_html=True,
            )
        elif EVENTS_CSV.exists():
            modified = datetime.fromtimestamp(EVENTS_CSV.stat().st_mtime)
            st.markdown(
                f"<div class='status-line'>{OK} events_clean.csv "
                f"&middot; {modified.strftime('%Y-%m-%d %H:%M')}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='status-line'>{WARN} events_clean.csv not found. "
                "Run <code>python main.py</code> to generate it.</div>",
                unsafe_allow_html=True,
            )

        if st.button("Reload data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        with st.spinner("Loading event stream"):
            events, provenance = load_data(
                source, n_events, days_back, gdelt_start, gdelt_end
            )

        if events.empty:
            st.markdown(
                f"<div class='status-line'>[ERROR] {provenance}</div>",
                unsafe_allow_html=True,
            )
            st.stop()

        st.markdown(
            f"<div class='status-line'>{OK} {provenance}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        section_title("Filters")
        event_types = ["All", *sorted(events["event_type"].dropna().unique().tolist())]
        selected_type = st.selectbox("Event type", event_types, key="event_type_filter")

        st.markdown("---")
        section_title("Narratives")
        use_llm = st.toggle(
            "Enable generated summaries",
            value=False,
            key="use_llm",
            help=(
                "Uses ANTHROPIC_API_KEY or OPENAI_API_KEY when present, "
                "otherwise a deterministic offline generator."
            ),
        )

        st.markdown("---")
        st.markdown(
            "<div class='status-line'>Engine: NetworkX + DistilBERT<br>"
            "GeoIntel pipeline v1.1</div>",
            unsafe_allow_html=True,
        )

    return events, provenance, selected_type, use_llm


def render_kpi_strip(stats: NetworkStats) -> None:
    """Render the global KPI strip.

    Parameters
    ----------
    stats : NetworkStats
        Global topology statistics.
    """
    columns = st.columns(6)
    metric_card(columns[0], f"{stats['nodes']:,}", "Countries")
    metric_card(columns[1], f"{stats['edges']:,}", "Interactions")
    metric_card(
        columns[2], f"{stats['avg_tone']:+.3f}", "Avg sentiment",
        POSITIVE if stats["avg_tone"] > 0 else NEGATIVE,
    )
    metric_card(
        columns[3], f"{stats['negative_edge_ratio']:.1%}", "Conflict rate", NEGATIVE
    )
    metric_card(columns[4], f"{stats['num_communities']:,}", "Blocs detected")
    metric_card(columns[5], f"{stats['ggpi']:.3f}", "GGPI", ACCENT_ALT)


def render_network_tab(
    graph: nx.DiGraph,
    metrics: pd.DataFrame,
    stats: NetworkStats,
) -> None:
    """Render the network map and the bloc analyser.

    Parameters
    ----------
    graph : networkx.DiGraph
        Interaction graph.
    metrics : pandas.DataFrame
        Node metrics.
    stats : NetworkStats
        Global statistics.
    """
    section_title(
        "Geopolitical network",
        "Node position is the country centroid. Diameter encodes GDP at PPP; "
        "edge colour encodes cooperative against conflictual tone.",
    )

    control_a, control_b, control_c = st.columns(3)
    with control_a:
        min_weight = st.slider("Minimum edge weight", 1, 20, 3, 1, key="net_weight")
    with control_b:
        node_ceiling = max(30, graph.number_of_nodes())
        max_nodes = st.slider(
            "Countries shown", 10, node_ceiling,
            min(30, node_ceiling), 5, key="net_nodes",
        )
    with control_c:
        color_by = st.selectbox(
            "Colour nodes by",
            ["PageRank", "Conflict Ratio", "Community"],
            key="net_color",
        )

    st.plotly_chart(
        build_network_figure(graph, metrics, stats, min_weight, max_nodes, color_by),
        use_container_width=True,
    )
    st.markdown(
        "<div style='display:flex; gap:8px; flex-wrap:wrap;'>"
        "<span class='tag'>Node diameter = GDP (PPP, 2023)</span>"
        "<span class='tag-green'>Green edge = cooperative</span>"
        "<span class='tag-red'>Red edge = conflictual</span>"
        "<span class='tag'>Edge width = interaction volume</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    section_title(
        "Bloc analyser",
        "Each country is scored against every pole by bilateral tone, one-hop "
        "propagation and a curated alignment prior, then assigned to its "
        "strongest pole.",
    )

    available = sorted(metrics.index.tolist())
    preset_column, poles_column = st.columns([1, 2])
    with preset_column:
        preset = st.selectbox("Preset", list(POLE_PRESETS), key="bloc_preset")
    with poles_column:
        poles = st.multiselect(
            "Power poles (2 to 4)",
            options=available,
            default=[code for code in POLE_PRESETS[preset] if code in available],
            max_selections=4,
            key="bloc_poles",
        )

    if len(poles) < 2:
        st.markdown(
            "<div class='status-line'>[INFO] Select at least two poles to "
            "compute bloc assignment.</div>",
            unsafe_allow_html=True,
        )
        return

    assignments, affinity = compute_blocs(graph, poles)
    st.plotly_chart(
        build_bloc_figure(graph, metrics, assignments, affinity, poles, min_weight),
        use_container_width=True,
    )

    membership: dict[str, list[str]] = {pole: [] for pole in poles}
    for code, bloc in assignments.items():
        if bloc in membership:
            membership[bloc].append(code)

    bloc_columns = st.columns(len(poles))
    for index, pole in enumerate(poles):
        members = sorted(membership[pole])
        color = BLOC_COLORS[index % len(BLOC_COLORS)]
        pills = "".join(
            f"<span style='display:inline-block;background:{color}18;"
            f"border:1px solid {color}40;color:{color};border-radius:2px;"
            f"padding:1px 7px;margin:2px;font-size:0.71rem'>{code}</span>"
            for code in members
        )
        with bloc_columns[index]:
            st.markdown(
                f"<div style='border-left:3px solid {color};padding:10px 14px;"
                f"background:rgba(17,24,39,0.8);border-radius:3px'>"
                f"<div style='color:{color};font-weight:700;font-size:0.95rem;"
                f"font-family:Syne,sans-serif'>{pole} bloc</div>"
                f"<div style='color:#94a3b8;font-size:0.72rem;margin:2px 0 8px 0'>"
                f"{len(members)} countries</div>{pills}</div>",
                unsafe_allow_html=True,
            )

    with st.expander("Full affinity scores", expanded=False):
        table = pd.DataFrame(affinity).T.round(4)
        table.index.name = "Country"
        table["Assigned bloc"] = table.index.map(lambda code: assignments.get(code, "-"))
        score_columns = [pole for pole in poles if pole in table.columns]
        st.dataframe(
            table.style.apply(diverging_gradient, subset=score_columns),
            use_container_width=True,
        )


def render_rankings_tab(metrics: pd.DataFrame) -> None:
    """Render the influence ranking table, radar profile and bar chart.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Node metrics.
    """
    section_title("Country influence rankings")

    table_column, radar_column = st.columns([3, 2])

    with table_column:
        top_n = st.slider("Countries listed", 5, 30, 15, key="rank_n")
        display = metrics.head(top_n)[
            ["pagerank", "betweenness", "eigenvector", "conflict_ratio", "total_events"]
        ].rename(columns={
            "pagerank": "PageRank",
            "betweenness": "Betweenness",
            "eigenvector": "Eigenvector",
            "conflict_ratio": "Conflict %",
            "total_events": "Events",
        })
        display = display.assign(**{"Conflict %": (display["Conflict %"] * 100).round(1)})
        display.index.name = "Country"
        st.dataframe(
            display.style
            .apply(sequential_gradient, hex_color=ACCENT, subset=["PageRank"])
            .apply(sequential_gradient, hex_color=NEGATIVE, subset=["Conflict %"])
            .format({"PageRank": "{:.6f}", "Betweenness": "{:.6f}",
                     "Eigenvector": "{:.6f}", "Conflict %": "{:.1f}",
                     "Events": "{:,.0f}"}),
            use_container_width=True,
        )

    with radar_column:
        selected = st.selectbox(
            "Radar profile", metrics.head(20).index.tolist(), key="radar_country"
        )
        st.plotly_chart(build_radar_chart(metrics, selected), use_container_width=True)

    st.plotly_chart(build_pagerank_bar(metrics), use_container_width=True)


def render_bilateral_tab(
    graph: nx.DiGraph,
    use_llm: bool,
) -> None:
    """Render the bilateral relationship analyser.

    Parameters
    ----------
    graph : networkx.DiGraph
        Interaction graph.
    use_llm : bool
        Whether narrative generation is enabled.
    """
    section_title("Bilateral relationship analysis")

    countries = sorted(graph.nodes())
    if len(countries) < 2:
        st.markdown(
            "<div class='status-line'>[WARN] At least two countries are required.</div>",
            unsafe_allow_html=True,
        )
        return

    column_a, column_b = st.columns(2)
    with column_a:
        country_a = st.selectbox("Country A", countries, index=0, key="bilateral_a")
    with column_b:
        alternatives = [code for code in countries if code != country_a]
        country_b = st.selectbox("Country B", alternatives, index=0, key="bilateral_b")

    if not st.button("Analyse relationship", type="primary"):
        return

    forward = dict(graph[country_a][country_b]) if graph.has_edge(country_a, country_b) else {}
    reverse = dict(graph[country_b][country_a]) if graph.has_edge(country_b, country_a) else {}

    tones = [edge["tone"] for edge in (forward, reverse) if edge]
    avg_tone = sum(tones) / len(tones) if tones else 0.0

    stats_column, chart_column = st.columns([2, 3])

    with stats_column:
        metric_card(
            st, f"{avg_tone:+.4f}", "Average sentiment tone", tone_color(avg_tone)
        )
        posture = (
            "Cooperative" if avg_tone > 0.1
            else "Conflictual" if avg_tone < -0.1
            else "Mixed / neutral"
        )
        rows = [("Relationship", posture)]
        if forward:
            rows.append((
                f"{country_a} to {country_b}",
                f"{forward.get('num_events', 0):,} events / "
                f"{forward.get('dominant_type', 'n/a')}",
            ))
        if reverse:
            rows.append((
                f"{country_b} to {country_a}",
                f"{reverse.get('num_events', 0):,} events / "
                f"{reverse.get('dominant_type', 'n/a')}",
            ))
        if not forward and not reverse:
            rows.append(("Status", "No direct interactions recorded"))

        st.markdown(
            "".join(
                f"<div class='kv-row'><span class='kv-key'>{key}</span>"
                f"<span class='kv-value'>{value}</span></div>"
                for key, value in rows
            ),
            unsafe_allow_html=True,
        )

    with chart_column:
        st.plotly_chart(
            build_bilateral_chart(graph, country_a, country_b), use_container_width=True
        )

    if use_llm:
        section_title("Intelligence summary")
        with st.spinner("Generating analysis"):
            summary = get_narrator().summarize_bilateral(graph, country_a, country_b)
        st.markdown(
            f"<div class='narrative-box'>{summary}</div>", unsafe_allow_html=True
        )


def render_temporal_tab(temporal: dict[str, nx.DiGraph], cache_key: str) -> None:
    """Render temporal evolution charts.

    Parameters
    ----------
    temporal : dict of str to networkx.DiGraph
        Snapshot graphs.
    cache_key : str
        Fingerprint of the underlying event frame.
    """
    section_title("Temporal network evolution")

    if len(temporal) < 2:
        st.markdown(
            "<div class='status-line'>[WARN] At least two periods are required. "
            "Widen the date range.</div>",
            unsafe_allow_html=True,
        )
        return

    frame = temporal_statistics(temporal, cache_key)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            build_temporal_line(
                frame, "avg_tone", "Average sentiment over time", ACCENT, zero_line=True
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            build_temporal_line(frame, "ggpi", "Polarization index over time", ACCENT_ALT),
            use_container_width=True,
        )

    st.plotly_chart(build_activity_figure(frame), use_container_width=True)


def render_overview_tab(
    graph: nx.DiGraph,
    metrics: pd.DataFrame,
    stats: NetworkStats,
    use_llm: bool,
) -> None:
    """Render global statistics, community membership and the tone matrix.

    Parameters
    ----------
    graph : networkx.DiGraph
        Interaction graph.
    metrics : pandas.DataFrame
        Node metrics.
    stats : NetworkStats
        Global statistics.
    use_llm : bool
        Whether narrative generation is enabled.
    """
    section_title("Global network overview")

    stats_column, blocs_column = st.columns(2)

    with stats_column:
        st.markdown(
            "<div class='section-note' style='margin-left:0'>Network statistics</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "".join(
                f"<div class='kv-row'><span class='kv-key'>{label}</span>"
                f"<span class='kv-value'>{value}</span></div>"
                for label, value in build_stat_rows(stats)
            ),
            unsafe_allow_html=True,
        )

    with blocs_column:
        st.markdown(
            "<div class='section-note' style='margin-left:0'>Detected communities</div>",
            unsafe_allow_html=True,
        )
        communities: dict[int, list[str]] = {}
        for code, index in stats.get("community_map", {}).items():
            communities.setdefault(index, []).append(code)

        for index, members in sorted(communities.items()):
            color = BLOC_COLORS[index % len(BLOC_COLORS)]
            tags = "".join(
                f"<span class='tag' style='border-color:{color}; color:{color}'>{code}</span>"
                for code in sorted(members)
            )
            st.markdown(
                f"<div style='margin:10px 0;'>"
                f"<div style='color:#64748b;font-size:0.68rem;letter-spacing:0.12em;"
                f"text-transform:uppercase;margin-bottom:4px;'>"
                f"Bloc {index + 1} &middot; {len(members)} countries</div>{tags}</div>",
                unsafe_allow_html=True,
            )

    st.plotly_chart(build_tone_heatmap(graph, metrics, top_n=15), use_container_width=True)

    if use_llm:
        section_title("Global intelligence summary")
        with st.spinner("Generating executive summary"):
            summary = get_narrator().summarize_network(graph, stats, metrics)
        st.markdown(
            f"<div class='narrative-box'>{summary}</div>", unsafe_allow_html=True
        )


def main() -> None:
    """Compose and render the dashboard."""
    masthead()

    events, _provenance, selected_type, use_llm = render_sidebar()

    filtered = (
        events if selected_type == "All"
        else events[events["event_type"] == selected_type]
    )

    if filtered.empty:
        st.markdown(
            f"<div class='status-line'>[WARN] No events match the filter "
            f"'{selected_type}'.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    cache_key = fingerprint(filtered)

    with st.spinner("Building geopolitical network"):
        graph, metrics, stats = load_graph(filtered, cache_key)

    if graph.number_of_nodes() == 0:
        st.markdown(
            "<div class='status-line'>[WARN] The filtered event stream produced "
            "an empty network.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    render_kpi_strip(stats)
    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "Network graph",
        "Influence rankings",
        "Bilateral analysis",
        "Temporal trends",
        "Network overview",
    ])

    with tabs[0]:
        render_network_tab(graph, metrics, stats)
    with tabs[1]:
        render_rankings_tab(metrics)
    with tabs[2]:
        render_bilateral_tab(graph, use_llm)
    with tabs[3]:
        with st.spinner("Building temporal snapshots"):
            temporal = load_temporal(filtered, cache_key)
        render_temporal_tab(temporal, cache_key)
    with tabs[4]:
        render_overview_tab(graph, metrics, stats, use_llm)


if __name__ == "__main__":
    main()
