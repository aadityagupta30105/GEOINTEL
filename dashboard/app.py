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

import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

import networkx as nx
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.graph_builder import (  # noqa: E402
    AGGREGATE_COLUMNS,
    NetworkStats,
    aggregate_events,
    build_graph_from_aggregates,
    build_temporal_graphs_from_aggregates,
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
    FONT_DISPLAY,
    GLOBAL_CSS,
    MUTED,
    NEGATIVE,
    PANEL_CLIP,
    POSITIVE,
    SURFACE,
    diverging_gradient,
    sequential_gradient,
    tone_color,
)
from data.store import DEFAULT_DB_PATH, EventStore, is_read_only_sql  # noqa: E402
from utils.logging_config import OK, WARN, get_logger  # noqa: E402

_log = get_logger(__name__)

OUTPUT_DIR: Final[Path] = _ROOT / "output"
EVENTS_CSV: Final[Path] = OUTPUT_DIR / "events_clean.csv"

# Columns the pipeline export must provide before it can be loaded.
_REQUIRED_EXPORT_COLUMNS: Final[frozenset[str]] = frozenset({
    "Actor1CountryCode", "Actor2CountryCode", "tone_norm", "event_type", "date",
})

_SOURCE_LABELS: Final[dict[str, str]] = {
    "database": "Event store (SQL)",
    "pipeline_output": "Pipeline export",
    "gdelt_live": "GDELT live fetch",
    "mock": "Synthetic simulation",
}

_GDELT_LIVE_MAX_DAYS: Final[int] = 7

DB_PATH: Final[Path] = DEFAULT_DB_PATH

# Sentinel used by the event-type selector for "no restriction".
_ALL_TYPES: Final[str] = "All"

# Row cap applied to SQL console results, so that a careless SELECT cannot
# pull a multi-million row table into the browser.
_CONSOLE_ROW_LIMIT: Final[int] = 5000

# Starting point offered in the SQL console. Deliberately unfiltered: a
# default query that returns nothing on a small store teaches the operator
# that the console is broken rather than that their predicate was narrow.
_CONSOLE_DEFAULT_QUERY: Final[str] = """SELECT source, target, num_events,
       ROUND(tone, 3) AS tone, conflict_count, coop_count
  FROM v_edges
 ORDER BY num_events DESC
 LIMIT 50"""


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


# --- Store-backed acquisition -----------------------------------------------
#
# The store path never materialises the event table. Filters are pushed into
# SQL and only the dyad aggregate crosses into pandas, which is what keeps a
# full year responsive. Every loader below is keyed on scalars alone so that
# Streamlit can cache it without hashing a frame.

@st.cache_data(show_spinner=False, ttl=300)
def store_profile(db_path: str) -> dict[str, object] | None:
    """Summarise the store so the sidebar can offer sensible defaults.

    Parameters
    ----------
    db_path : str
        Path to the SQLite store.

    Returns
    -------
    dict or None
        Statistics, event types and country codes, or ``None`` when the store
        is absent, unreadable or empty.
    """
    if not Path(db_path).exists():
        return None
    try:
        with EventStore(db_path, read_only=True) as store:
            stats = store.stats()
            if not stats["dyad_rows"]:
                return None
            return {
                "stats": dict(stats),
                "event_types": store.event_types(),
                "countries": store.countries(),
            }
    except (sqlite3.Error, FileNotFoundError, OSError) as exc:
        _log.error("Store unreadable at %s: %s", db_path, exc)
        return None


@st.cache_data(show_spinner=False)
def store_aggregates(
    db_path: str,
    start: str,
    end: str,
    event_types: tuple[str, ...],
    countries: tuple[str, ...],
    period: str = "",
) -> pd.DataFrame:
    """Query the dyad aggregate for a window, filtered in SQL.

    Parameters
    ----------
    db_path : str
        Path to the SQLite store.
    start, end : str
        Inclusive window bounds as ``YYYY-MM-DD``.
    event_types : tuple of str
        Event-type restriction; empty means no restriction.
    countries : tuple of str
        Country restriction; empty means no restriction.
    period : str, optional
        Period granularity for temporal snapshots, or ``""`` for a single
        aggregate over the whole window.

    Returns
    -------
    pandas.DataFrame
        Aggregate frame, empty when the query matched nothing.
    """
    with EventStore(db_path, read_only=True) as store:
        return store.dyad_aggregates(
            start=start,
            end=end,
            event_types=list(event_types) or None,
            countries=list(countries) or None,
            period=period or None,
        )


@st.cache_data(show_spinner=False)
def store_coverage(db_path: str) -> pd.DataFrame:
    """Read per-month collection coverage from the store.

    Parameters
    ----------
    db_path : str
        Path to the SQLite store.

    Returns
    -------
    pandas.DataFrame
        Coverage frame, empty when nothing has been harvested.
    """
    with EventStore(db_path, read_only=True) as store:
        return store.coverage()


@st.cache_data(show_spinner=False)
def store_schema(db_path: str) -> pd.DataFrame:
    """Describe the store's tables and views for the SQL console.

    Parameters
    ----------
    db_path : str
        Path to the SQLite store.

    Returns
    -------
    pandas.DataFrame
        One row per column.
    """
    with EventStore(db_path, read_only=True) as store:
        return store.schema()


@st.cache_data(show_spinner=False)
def run_console_query(db_path: str, sql: str) -> tuple[pd.DataFrame, str]:
    """Execute an operator-supplied query under two independent guards.

    The statement must pass :func:`data.store.is_read_only_sql`, and it runs
    on a connection opened in SQLite read-only mode. Either guard alone would
    do; both are cheap, and a write reaching the store from a text box is not
    a failure worth risking.

    Parameters
    ----------
    db_path : str
        Path to the SQLite store.
    sql : str
        Statement to run.

    Returns
    -------
    tuple of (pandas.DataFrame, str)
        Result rows and a status message. The frame is empty when the
        statement was rejected or failed.
    """
    if not is_read_only_sql(sql):
        return pd.DataFrame(), (
            "Rejected: the console accepts a single SELECT or WITH statement."
        )

    try:
        with EventStore(db_path, read_only=True) as store:
            frame = store.query(f"SELECT * FROM ({sql.strip().rstrip(';')}) "
                                f"LIMIT {_CONSOLE_ROW_LIMIT}")
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        return pd.DataFrame(), f"SQL error: {exc}"

    if frame.empty:
        return frame, "The query ran and matched no rows."

    truncated = (
        f" (truncated to {_CONSOLE_ROW_LIMIT:,})"
        if len(frame) >= _CONSOLE_ROW_LIMIT
        else ""
    )
    return frame, f"{len(frame):,} row(s){truncated}"


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
    _aggregates: pd.DataFrame,
    cache_key: str,
) -> tuple[nx.DiGraph, pd.DataFrame, NetworkStats]:
    """Build the static graph and its analytics for a dyad aggregate.

    Parameters
    ----------
    _aggregates : pandas.DataFrame
        Frame carrying :data:`analysis.graph_builder.AGGREGATE_COLUMNS`.
        Excluded from the cache key by the underscore prefix; ``cache_key``
        carries the identity instead.
    cache_key : str
        Fingerprint produced by :func:`fingerprint`.

    Returns
    -------
    tuple of (networkx.DiGraph, pandas.DataFrame, NetworkStats)
        Graph, node metrics and global statistics.
    """
    graph = build_graph_from_aggregates(_aggregates)
    return graph, compute_metrics(graph), compute_network_stats(graph)


@st.cache_resource(show_spinner=False)
def load_temporal(
    _aggregates: pd.DataFrame,
    cache_key: str,
) -> dict[str, nx.DiGraph]:
    """Build temporal snapshot graphs from a period-tagged aggregate.

    Parameters
    ----------
    _aggregates : pandas.DataFrame
        Aggregate frame carrying a ``period`` column, excluded from the cache
        key.
    cache_key : str
        Fingerprint produced by :func:`fingerprint`.

    Returns
    -------
    dict of str to networkx.DiGraph
        Snapshot graphs keyed by period label.
    """
    return build_temporal_graphs_from_aggregates(_aggregates)


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


@dataclass(frozen=True, slots=True)
class Selection:
    """The active data selection resolved by the sidebar.

    One object describes where the aggregate came from and how it was
    filtered, which lets :func:`main` stay agnostic about the source. The
    store path fills ``db_path`` and leaves ``events`` empty; the frame paths
    do the reverse.

    Attributes
    ----------
    source : str
        Key from :data:`_SOURCE_LABELS`.
    provenance : str
        Human-readable description of what was loaded.
    aggregates : pandas.DataFrame
        Dyad aggregate over the selected window.
    temporal : pandas.DataFrame
        The same aggregate tagged with a ``period`` column.
    db_path : str
        Store path, empty for the frame-backed sources.
    use_llm : bool
        Whether narrative generation is enabled.
    """

    source: str
    provenance: str
    aggregates: pd.DataFrame
    temporal: pd.DataFrame
    db_path: str
    use_llm: bool


def _default_source() -> str:
    """Choose the source to select on first load.

    The store is preferred when it holds data, then the pipeline export, then
    synthetic generation. Defaulting to something that does not exist would
    halt the app on a fresh checkout or a hosted deployment, where neither the
    store nor the output directory is committed.

    Returns
    -------
    str
        Key from :data:`_SOURCE_LABELS`.
    """
    if store_profile(str(DB_PATH)):
        return "database"
    return "pipeline_output" if EVENTS_CSV.exists() else "mock"


def _render_store_controls(profile: dict[str, object]) -> tuple[
    pd.DataFrame, pd.DataFrame, str
]:
    """Render the store filters and run the resulting queries.

    Every control here maps onto a ``WHERE`` clause rather than onto a pandas
    mask, so widening the window costs a query and not a reload.

    Parameters
    ----------
    profile : dict
        Store summary from :func:`store_profile`.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame, str)
        The windowed aggregate, the same aggregate tagged by period, and a
        provenance label.
    """
    stats = profile["stats"]
    first = str(stats["first_date"])
    last = str(stats["last_date"])
    lower = date.fromisoformat(first)
    upper = date.fromisoformat(last)

    st.markdown(
        f"<div class='status-line'>{OK} {int(stats['events']):,} events &middot; "
        f"{first} to {last}<br>{int(stats['countries']):,} countries &middot; "
        f"{int(stats['size_bytes']) / (1024 ** 2):,.0f} MB on disk</div>",
        unsafe_allow_html=True,
    )

    window = st.date_input(
        "Window",
        value=(lower, upper),
        min_value=lower,
        max_value=upper,
        key="store_window",
    )
    # A date_input in range mode returns a one-element tuple mid-edit, between
    # the first and second click. Hold the previous end date until the second
    # arrives rather than querying an unintended window.
    start_date, end_date = window if len(window) == 2 else (window[0], upper)

    available_types = [str(value) for value in profile["event_types"]]
    chosen_types = st.multiselect(
        "Event types",
        options=available_types,
        default=[],
        key="store_types",
        help="Empty means every type.",
    )

    chosen_countries = st.multiselect(
        "Countries",
        options=[str(value) for value in profile["countries"]],
        default=[],
        key="store_countries",
        help="Empty means every country. Matches either side of a dyad.",
    )

    period = st.selectbox(
        "Temporal granularity",
        options=["month", "quarter", "year"],
        key="store_period",
    )

    start, end = start_date.isoformat(), end_date.isoformat()
    types = tuple(chosen_types)
    countries = tuple(chosen_countries)

    with st.spinner("Querying event store"):
        aggregates = store_aggregates(str(DB_PATH), start, end, types, countries)
        temporal = store_aggregates(
            str(DB_PATH), start, end, types, countries, period=period
        )

    events = int(aggregates["num_events"].sum()) if not aggregates.empty else 0
    return aggregates, temporal, (
        f"Event store - {events:,} events over {start} to {end}"
    )


def _render_frame_controls(source: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Render the controls for the frame-backed sources and load them.

    Parameters
    ----------
    source : {"pipeline_output", "gdelt_live", "mock"}
        Selected source.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame, str)
        The aggregate, the period-tagged aggregate, and a provenance label.
    """
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
            "Allow roughly five seconds per day. For anything longer than a "
            "week use <code>python -m data.harvest</code> and read the "
            "store.</div>",
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

    selected_type = st.selectbox(
        "Event type",
        [_ALL_TYPES, *sorted(events["event_type"].dropna().unique().tolist())],
        key="event_type_filter",
    )
    if selected_type != _ALL_TYPES:
        events = events[events["event_type"] == selected_type]
        provenance = f"{provenance} / {selected_type}"

    if events.empty:
        st.markdown(
            f"<div class='status-line'>[WARN] No events match "
            f"'{selected_type}'.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    period = st.selectbox(
        "Temporal granularity",
        options=["month", "quarter", "year"],
        key="frame_period",
    )

    aggregates = aggregate_events(events)
    temporal = _tag_periods(events, period)
    return aggregates, temporal, provenance


def _tag_periods(events: pd.DataFrame, period: str) -> pd.DataFrame:
    """Aggregate an event frame per period, matching the store's period labels.

    Parameters
    ----------
    events : pandas.DataFrame
        Preprocessed events carrying a parseable ``date`` column.
    period : {"month", "quarter", "year"}
        Granularity.

    Returns
    -------
    pandas.DataFrame
        Aggregate carrying an extra ``period`` column, empty when no row has a
        parseable date.
    """
    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()]

    if frame.empty:
        return pd.DataFrame(columns=["period", *AGGREGATE_COLUMNS])

    if period == "month":
        labels = frame["date"].dt.to_period("M").astype(str)
    elif period == "quarter":
        labels = frame["date"].dt.to_period("Q").astype(str)
    else:
        labels = frame["date"].dt.year.astype(str)

    return pd.concat(
        [
            aggregate_events(group).assign(period=str(label))
            for label, group in frame.groupby(labels)
        ],
        ignore_index=True,
    )


def render_sidebar() -> Selection:
    """Render the sidebar and resolve the active data selection.

    Returns
    -------
    Selection
        The loaded aggregates, provenance and narrative setting.
    """
    with st.sidebar:
        section_title("Data source")

        profile = store_profile(str(DB_PATH))
        source_options = [
            key for key in _SOURCE_LABELS
            if key != "database" or profile is not None
        ]
        default_source = _default_source()

        source = st.radio(
            "Source",
            options=source_options,
            format_func=lambda key: _SOURCE_LABELS[key],
            index=source_options.index(default_source),
            label_visibility="collapsed",
            help=(
                "The event store is the authoritative source. Fill it with "
                "python -m data.harvest."
            ),
            key="source",
        )

        if st.button("Reload data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        section_title("Filters")

        if source == "database" and profile is not None:
            aggregates, temporal, provenance = _render_store_controls(profile)
        else:
            aggregates, temporal, provenance = _render_frame_controls(source)

        if aggregates.empty:
            st.markdown(
                f"<div class='status-line'>[WARN] {provenance} matched no "
                "events. Widen the filters.</div>",
                unsafe_allow_html=True,
            )
            st.stop()

        st.markdown(
            f"<div class='status-line'>{OK} {provenance}</div>",
            unsafe_allow_html=True,
        )

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
            "<div class='status-line'>Engine: NetworkX + SQLite<br>"
            "GeoIntel pipeline v1.2</div>",
            unsafe_allow_html=True,
        )

    return Selection(
        source=source,
        provenance=provenance,
        aggregates=aggregates,
        temporal=temporal,
        db_path=str(DB_PATH) if source == "database" else "",
        use_llm=use_llm,
    )
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
                f"background:{SURFACE};clip-path:{PANEL_CLIP}'>"
                f"<div style='color:{color};font-weight:700;font-size:0.95rem;"
                f"font-family:{FONT_DISPLAY};letter-spacing:0.1em;"
                f"text-transform:uppercase'>{pole} bloc</div>"
                f"<div style='color:{MUTED};font-size:0.72rem;margin:2px 0 8px 0'>"
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
                f"<div style='color:{MUTED};font-size:0.68rem;letter-spacing:0.16em;"
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


def render_query_tab(db_path: str) -> None:
    """Render the SQL console over the event store.

    The rest of the dashboard answers the questions it was built to answer.
    This tab exists so the operator can ask one it was not: the schema is on
    screen, the query box is free-form, and the result is exportable. Both
    guards described in :func:`run_console_query` apply.

    Parameters
    ----------
    db_path : str
        Path to the SQLite store.
    """
    section_title(
        "SQL console",
        "Read-only. A single SELECT or WITH statement runs against the store; "
        f"results are capped at {_CONSOLE_ROW_LIMIT:,} rows.",
    )

    schema = store_schema(db_path)
    coverage = store_coverage(db_path)

    editor_column, schema_column = st.columns([3, 2])

    with schema_column:
        st.markdown(
            "<div class='section-note' style='margin-left:0'>Schema</div>",
            unsafe_allow_html=True,
        )
        for name, columns in schema.groupby("object", sort=False):
            kind = str(columns["kind"].iloc[0])
            fields = ", ".join(str(value) for value in columns["column"])
            st.markdown(
                f"<div style='margin-bottom:9px'>"
                f"<span class='tag'>{kind}</span> "
                f"<span style='color:{ACCENT};font-weight:600'>{name}</span>"
                f"<div class='status-line' style='margin-top:3px'>{fields}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with editor_column:
        query = st.text_area(
            "Query",
            value=_CONSOLE_DEFAULT_QUERY,
            height=200,
            key="console_sql",
            label_visibility="collapsed",
        )
        run = st.button("Run query", type="primary", key="console_run")

    if not run:
        return

    with st.spinner("Executing"):
        result, message = run_console_query(db_path, query)

    if result.empty:
        st.markdown(
            f"<div class='status-line'>[WARN] {message}</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div class='status-line'>{OK} {message}</div>", unsafe_allow_html=True
    )
    st.dataframe(result, use_container_width=True, height=420)
    st.download_button(
        "Download CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="geointel_query.csv",
        mime="text/csv",
        key="console_download",
    )

    if not coverage.empty:
        with st.expander("Collection coverage by month", expanded=False):
            st.dataframe(coverage, use_container_width=True)


def main() -> None:
    """Compose and render the dashboard."""
    masthead()

    selection = render_sidebar()
    cache_key = fingerprint(selection.aggregates)

    with st.spinner("Building geopolitical network"):
        graph, metrics, stats = load_graph(selection.aggregates, cache_key)

    if graph.number_of_nodes() == 0:
        st.markdown(
            "<div class='status-line'>[WARN] The selection produced an empty "
            "network. Widen the window or clear a filter.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    render_kpi_strip(stats)
    st.markdown("<br>", unsafe_allow_html=True)

    labels = [
        "Network graph",
        "Influence rankings",
        "Bilateral analysis",
        "Temporal trends",
        "Network overview",
    ]
    if selection.db_path:
        labels.append("SQL console")

    tabs = st.tabs(labels)

    with tabs[0]:
        render_network_tab(graph, metrics, stats)
    with tabs[1]:
        render_rankings_tab(metrics)
    with tabs[2]:
        render_bilateral_tab(graph, selection.use_llm)
    with tabs[3]:
        with st.spinner("Building temporal snapshots"):
            temporal = load_temporal(selection.temporal, f"{cache_key}-temporal")
        render_temporal_tab(temporal, cache_key)
    with tabs[4]:
        render_overview_tab(graph, metrics, stats, selection.use_llm)
    if selection.db_path:
        with tabs[5]:
            render_query_tab(selection.db_path)


if __name__ == "__main__":
    main()
