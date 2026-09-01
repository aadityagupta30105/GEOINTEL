"""
Geopolitical Graph Builder
==========================
Constructs directed weighted graphs from preprocessed event data and derives
node-level centrality metrics, global topology statistics and temporal
snapshots.

Edge attributes
---------------
``weight``          Aggregate interaction strength under the selected scheme.
``distance``        Reciprocal of ``weight``; the shortest-path cost used by
                    distance-based centralities.
``tone``            Mean normalised sentiment in ``[-1, 1]``.
``num_events``      Event count on the directed dyad.
``mentions``        Summed media mentions.
``dominant_type``   Most frequent event type, ties broken alphabetically.
``event_types``     Full event-type histogram.
``conflict_count``  Events classified as conflictual.
``coop_count``      Events classified as cooperative.
"""

from __future__ import annotations

from typing import Any, Final, Literal, TypedDict

import networkx as nx
import numpy as np
import pandas as pd

from utils.logging_config import OK, WARN, get_logger

__all__ = [
    "NetworkStats",
    "BilateralSummary",
    "METRIC_COLUMNS",
    "build_graph",
    "build_temporal_graphs",
    "get_undirected",
    "compute_metrics",
    "compute_network_stats",
    "get_bilateral_summary",
]

_log = get_logger(__name__)

WeightScheme = Literal["frequency", "tone", "mentions"]
TemporalPeriod = Literal["month", "quarter", "year"]

# Event types contributing to the conflict and cooperation counters.
_CONFLICT_TYPES: Final[frozenset[str]] = frozenset({"Conflict", "Military/Conflict"})
_COOPERATION_TYPES: Final[frozenset[str]] = frozenset({"Cooperation", "Trade/Aid"})

# GGPI component weights. Documented here as the single source of truth.
GGPI_MODULARITY_WEIGHT: Final[float] = 0.40
GGPI_NEGATIVE_EDGE_WEIGHT: Final[float] = 0.40
GGPI_NEGATIVE_TONE_WEIGHT: Final[float] = 0.20

# Relationship classification thresholds on mean bilateral tone.
_COOPERATIVE_THRESHOLD: Final[float] = 0.10
_CONFLICTUAL_THRESHOLD: Final[float] = -0.10

_PAGERANK_ALPHA: Final[float] = 0.85
_EIGENVECTOR_MAX_ITER: Final[int] = 500

METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "pagerank",
    "degree_centrality",
    "betweenness",
    "eigenvector",
    "in_degree_weighted",
    "out_degree_weighted",
    "avg_out_tone",
    "avg_in_tone",
    "conflict_ratio",
    "total_events",
)


class NetworkStats(TypedDict):
    """Global topology statistics for a single graph snapshot.

    Attributes
    ----------
    nodes, edges : int
        Graph order and size.
    density : float
        Directed edge density.
    reciprocity : float
        Fraction of directed edges whose reverse edge also exists.
    avg_tone : float
        Mean edge tone across the network.
    negative_edge_ratio : float
        Fraction of edges carrying negative tone.
    modularity : float
        Modularity of the detected community partition.
    num_communities : int
        Number of detected communities.
    ggpi : float
        Global Geopolitical Polarization Index in ``[0, 1]``.
    community_map : dict
        Node to community index assignment. Not JSON-serialised by callers
        that persist the remaining scalar fields.
    """

    nodes: int
    edges: int
    density: float
    reciprocity: float
    avg_tone: float
    negative_edge_ratio: float
    modularity: float
    num_communities: int
    ggpi: float
    community_map: dict[str, int]


class BilateralSummary(TypedDict):
    """Directed relationship record for a single country pair."""

    country_a: str
    country_b: str
    a_to_b: dict[str, Any] | None
    b_to_a: dict[str, Any] | None
    relationship_type: str
    dominant_tone: float


def _edge_weight(
    scheme: WeightScheme,
    num_events: int,
    avg_tone: float,
    mentions: int,
) -> float:
    """Compute an edge weight under the requested scheme.

    Parameters
    ----------
    scheme : {"frequency", "tone", "mentions"}
        Weighting scheme.
    num_events : int
        Event count on the dyad.
    avg_tone : float
        Mean normalised tone on the dyad.
    mentions : int
        Summed media mentions on the dyad.

    Returns
    -------
    float
        The edge weight. Always strictly positive for realised edges under
        the ``frequency`` and ``mentions`` schemes.
    """
    if scheme == "frequency":
        return float(num_events)
    if scheme == "tone":
        return abs(avg_tone) * num_events
    return float(mentions)


def build_graph(df: pd.DataFrame, weight_by: WeightScheme = "frequency") -> nx.DiGraph:
    """Build a directed weighted graph from preprocessed event data.

    Aggregation is fully vectorised: events are grouped by directed dyad and
    event type rather than iterated row by row, which keeps construction cost
    proportional to the number of dyads rather than the number of events.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed events carrying ``Actor1CountryCode``,
        ``Actor2CountryCode``, ``tone_norm``, ``event_type`` and
        ``NumMentions``.
    weight_by : {"frequency", "tone", "mentions"}, optional
        Edge weighting scheme.

    Returns
    -------
    networkx.DiGraph
        Directed graph with the documented edge and node attributes. Empty
        when ``df`` contains no rows.
    """
    graph = nx.DiGraph()

    if df.empty:
        _log.warning("%s Graph construction received an empty event frame", WARN)
        return graph

    pair_keys = ["Actor1CountryCode", "Actor2CountryCode"]

    working = df[pair_keys + ["tone_norm", "event_type"]].copy()
    working["mentions"] = (
        pd.to_numeric(df["NumMentions"], errors="coerce").fillna(1).astype("int64")
    )

    aggregated = working.groupby(pair_keys, sort=False).agg(
        num_events=("tone_norm", "size"),
        tone_sum=("tone_norm", "sum"),
        mentions=("mentions", "sum"),
    )

    # Event-type histogram per directed dyad, aligned to the aggregate index.
    type_matrix = (
        working.groupby(pair_keys + ["event_type"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(aggregated.index, fill_value=0)
    )
    type_labels: list[str] = [str(label) for label in type_matrix.columns]

    conflict_columns = [c for c in type_matrix.columns if c in _CONFLICT_TYPES]
    coop_columns = [c for c in type_matrix.columns if c in _COOPERATION_TYPES]
    conflict_totals = (
        type_matrix[conflict_columns].sum(axis=1)
        if conflict_columns
        else pd.Series(0, index=type_matrix.index)
    )
    coop_totals = (
        type_matrix[coop_columns].sum(axis=1)
        if coop_columns
        else pd.Series(0, index=type_matrix.index)
    )

    type_counts = type_matrix.to_numpy()

    for position, ((source, target), row) in enumerate(aggregated.iterrows()):
        num_events = int(row["num_events"])
        avg_tone = float(row["tone_sum"]) / num_events if num_events else 0.0
        mentions = int(row["mentions"])

        histogram = {
            label: int(count)
            for label, count in zip(type_labels, type_counts[position])
            if count
        }
        # Deterministic tie-break: highest count, then alphabetical label.
        dominant_type = (
            min(histogram.items(), key=lambda item: (-item[1], item[0]))[0]
            if histogram
            else "Unknown"
        )

        weight = _edge_weight(weight_by, num_events, avg_tone, mentions)

        graph.add_edge(
            source,
            target,
            weight=weight,
            distance=1.0 / weight if weight > 0 else float("inf"),
            tone=round(avg_tone, 4),
            num_events=num_events,
            mentions=mentions,
            dominant_type=dominant_type,
            event_types=histogram,
            conflict_count=int(conflict_totals.iloc[position]),
            coop_count=int(coop_totals.iloc[position]),
        )

    for node in graph.nodes():
        graph.nodes[node]["label"] = node
        graph.nodes[node]["total_events"] = sum(
            data["num_events"] for _, _, data in graph.out_edges(node, data=True)
        ) + sum(
            data["num_events"] for _, _, data in graph.in_edges(node, data=True)
        )

    _log.info(
        "%s Graph built: %d nodes, %d edges (weight_by=%s)",
        OK, graph.number_of_nodes(), graph.number_of_edges(), weight_by,
    )
    return graph


def build_temporal_graphs(
    df: pd.DataFrame,
    period: TemporalPeriod = "month",
) -> dict[str, nx.DiGraph]:
    """Slice events into period snapshots and build one graph per slice.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed events carrying a parseable ``date`` column.
    period : {"month", "quarter", "year"}, optional
        Temporal granularity.

    Returns
    -------
    dict of str to networkx.DiGraph
        Mapping of period label to snapshot graph, empty when ``df`` is empty.
    """
    if df.empty:
        _log.warning("%s Temporal slicing received an empty event frame", WARN)
        return {}

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()]

    if frame.empty:
        _log.warning("%s No parseable dates available for temporal slicing", WARN)
        return {}

    if period == "month":
        frame["period"] = frame["date"].dt.to_period("M").astype(str)
    elif period == "quarter":
        frame["period"] = frame["date"].dt.to_period("Q").astype(str)
    else:
        frame["period"] = frame["date"].dt.year.astype(str)

    graphs = {
        str(label): build_graph(group) for label, group in frame.groupby("period")
    }

    periods = sorted(graphs)
    _log.info(
        "%s Built %d temporal graphs (%s): %s to %s",
        OK, len(graphs), period, periods[0], periods[-1],
    )
    return graphs


def get_undirected(G: nx.DiGraph) -> nx.Graph:
    """Collapse a directed graph into an undirected one.

    Reciprocal edges are merged: weights, event counts and conflict or
    cooperation counters are summed, tone is averaged, and the event-type
    histograms are combined. Nested attributes are copied rather than aliased
    so that mutating the undirected view cannot corrupt the source graph.

    Parameters
    ----------
    G : networkx.DiGraph
        Source directed graph.

    Returns
    -------
    networkx.Graph
        Undirected projection carrying merged edge attributes.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(G.nodes(data=True))

    for source, target, data in G.edges(data=True):
        if undirected.has_edge(source, target):
            merged = undirected[source][target]
            merged["weight"] += data.get("weight", 0.0)
            merged["num_events"] += data.get("num_events", 0)
            merged["mentions"] += data.get("mentions", 0)
            merged["conflict_count"] += data.get("conflict_count", 0)
            merged["coop_count"] += data.get("coop_count", 0)
            merged["tone"] = (merged["tone"] + data.get("tone", 0.0)) / 2.0
            for label, count in data.get("event_types", {}).items():
                merged["event_types"][label] = merged["event_types"].get(label, 0) + count
            merged["distance"] = (
                1.0 / merged["weight"] if merged["weight"] > 0 else float("inf")
            )
        else:
            attributes = dict(data)
            attributes["event_types"] = dict(data.get("event_types", {}))
            undirected.add_edge(source, target, **attributes)

    return undirected


def _eigenvector_centrality(G: nx.DiGraph) -> dict[str, float]:
    """Compute eigenvector centrality with a deterministic fallback chain.

    Power iteration is attempted first; on convergence failure the dense
    NumPy solver is used, and only if that also fails are zeros returned.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to analyse.

    Returns
    -------
    dict of str to float
        Centrality per node.
    """
    try:
        return nx.eigenvector_centrality(
            G, weight="weight", max_iter=_EIGENVECTOR_MAX_ITER
        )
    except (nx.NetworkXException, ValueError):
        _log.warning("%s Eigenvector power iteration failed; using dense solver", WARN)

    try:
        return nx.eigenvector_centrality_numpy(G, weight="weight")
    except (nx.NetworkXException, ValueError, np.linalg.LinAlgError):
        _log.warning("%s Eigenvector centrality unavailable; reporting zeros", WARN)
        return dict.fromkeys(G.nodes(), 0.0)


def _betweenness_centrality(G: nx.DiGraph) -> dict[str, float]:
    """Compute betweenness centrality over interaction distance.

    NetworkX interprets the ``weight`` keyword as an edge *cost*. Passing the
    interaction volume directly would treat the busiest dyads as the longest
    paths and invert the metric, so the reciprocal ``distance`` attribute is
    used instead.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to analyse. Edges are expected to carry ``distance``.

    Returns
    -------
    dict of str to float
        Normalised betweenness per node.
    """
    try:
        return nx.betweenness_centrality(G, weight="distance", normalized=True)
    except (nx.NetworkXException, ValueError):
        _log.warning("%s Betweenness centrality unavailable; reporting zeros", WARN)
        return dict.fromkeys(G.nodes(), 0.0)


def compute_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """Compute node-level centrality and posture metrics.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph produced by :func:`build_graph`.

    Returns
    -------
    pandas.DataFrame
        Numeric frame indexed by country code and sorted by PageRank
        descending, carrying the columns in :data:`METRIC_COLUMNS`. Empty
        with the correct columns and dtypes when ``G`` has no nodes.
    """
    if G.number_of_nodes() == 0:
        empty = pd.DataFrame(columns=list(METRIC_COLUMNS)).astype("float64")
        empty.index.name = "country"
        return empty

    pagerank = nx.pagerank(G, weight="weight", alpha=_PAGERANK_ALPHA)
    degree_centrality = nx.degree_centrality(G)
    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))
    betweenness = _betweenness_centrality(G)
    eigenvector = _eigenvector_centrality(G)

    records: dict[str, dict[str, float]] = {}
    for node in G.nodes():
        out_tones = [data["tone"] for _, _, data in G.out_edges(node, data=True)]
        in_tones = [data["tone"] for _, _, data in G.in_edges(node, data=True)]

        conflict_total = sum(
            data["conflict_count"] for _, _, data in G.out_edges(node, data=True)
        )
        coop_total = sum(
            data["coop_count"] for _, _, data in G.out_edges(node, data=True)
        )
        classified = conflict_total + coop_total

        records[node] = {
            "pagerank": round(pagerank.get(node, 0.0), 6),
            "degree_centrality": round(degree_centrality.get(node, 0.0), 4),
            "betweenness": round(betweenness.get(node, 0.0), 6),
            "eigenvector": round(eigenvector.get(node, 0.0), 6),
            "in_degree_weighted": round(float(in_degree.get(node, 0.0)), 2),
            "out_degree_weighted": round(float(out_degree.get(node, 0.0)), 2),
            "avg_out_tone": round(float(np.mean(out_tones)) if out_tones else 0.0, 4),
            "avg_in_tone": round(float(np.mean(in_tones)) if in_tones else 0.0, 4),
            "conflict_ratio": round(
                conflict_total / classified if classified else 0.0, 4
            ),
            "total_events": float(G.nodes[node].get("total_events", 0)),
        }

    metrics = pd.DataFrame.from_dict(records, orient="index")
    metrics = metrics[list(METRIC_COLUMNS)].astype("float64")
    metrics = metrics.sort_values("pagerank", ascending=False)
    metrics.index.name = "country"

    _log.info("%s Metrics computed for %d countries", OK, len(metrics))
    return metrics


def _detect_communities(UG: nx.Graph) -> tuple[float, dict[str, int]]:
    """Partition an undirected graph and score the partition.

    Parameters
    ----------
    UG : networkx.Graph
        Undirected projection of the interaction network.

    Returns
    -------
    tuple of (float, dict)
        Modularity of the partition and the node to community mapping. On
        failure the modularity is ``0.0`` and every node is assigned to
        community ``0``.
    """
    if UG.number_of_edges() == 0:
        return 0.0, dict.fromkeys(UG.nodes(), 0)

    try:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(UG, weight="weight"))
        modularity = nx.algorithms.community.quality.modularity(
            UG, communities, weight="weight"
        )
        community_map = {
            node: index
            for index, community in enumerate(communities)
            for node in community
        }
        return float(modularity), community_map
    except (nx.NetworkXException, ValueError, ZeroDivisionError):
        _log.warning("%s Community detection failed; reporting a single bloc", WARN)
        return 0.0, dict.fromkeys(UG.nodes(), 0)


def compute_network_stats(G: nx.DiGraph) -> NetworkStats:
    """Compute global topology statistics including the GGPI.

    The Global Geopolitical Polarization Index is a convex combination of
    community separation, conflict prevalence and negative sentiment::

        GGPI = 0.40 * modularity
             + 0.40 * negative_edge_ratio
             + 0.20 * max(0, -avg_tone)

    Parameters
    ----------
    G : networkx.DiGraph
        Graph produced by :func:`build_graph`.

    Returns
    -------
    NetworkStats
        Populated statistics. Degenerate graphs (no nodes or no edges) return
        zeroed metrics rather than raising.
    """
    undirected = get_undirected(G)
    modularity, community_map = _detect_communities(undirected)

    tones = [data["tone"] for _, _, data in G.edges(data=True)]
    if tones:
        negative_ratio = sum(1 for tone in tones if tone < 0) / len(tones)
        avg_tone = float(np.mean(tones))
    else:
        negative_ratio = 0.0
        avg_tone = 0.0

    density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
    # overall_reciprocity is undefined for graphs without edges.
    reciprocity = nx.overall_reciprocity(G) if G.number_of_edges() > 0 else 0.0

    ggpi = (
        GGPI_MODULARITY_WEIGHT * modularity
        + GGPI_NEGATIVE_EDGE_WEIGHT * negative_ratio
        + GGPI_NEGATIVE_TONE_WEIGHT * max(0.0, -avg_tone)
    )

    stats: NetworkStats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(float(density), 4),
        "reciprocity": round(float(reciprocity), 4),
        "avg_tone": round(avg_tone, 4),
        "negative_edge_ratio": round(negative_ratio, 4),
        "modularity": round(modularity, 4),
        "num_communities": len(set(community_map.values())) if community_map else 0,
        "ggpi": round(ggpi, 4),
        "community_map": community_map,
    }

    _log.info(
        "%s Network statistics: nodes=%d edges=%d density=%.4f ggpi=%.4f",
        OK, stats["nodes"], stats["edges"], stats["density"], stats["ggpi"],
    )
    return stats


def get_bilateral_summary(
    G: nx.DiGraph,
    country_a: str,
    country_b: str,
) -> BilateralSummary:
    """Extract the directed relationship record for a country pair.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph produced by :func:`build_graph`.
    country_a, country_b : str
        ISO-3 country codes.

    Returns
    -------
    BilateralSummary
        Both directed edges where present, the mean tone across the realised
        directions, and a categorical relationship label.
    """
    forward = dict(G[country_a][country_b]) if G.has_edge(country_a, country_b) else None
    reverse = dict(G[country_b][country_a]) if G.has_edge(country_b, country_a) else None

    tones = [edge["tone"] for edge in (forward, reverse) if edge is not None]

    if tones:
        mean_tone = float(np.mean(tones))
        if mean_tone > _COOPERATIVE_THRESHOLD:
            relationship = "Cooperative"
        elif mean_tone < _CONFLICTUAL_THRESHOLD:
            relationship = "Conflictual"
        else:
            relationship = "Neutral/Mixed"
    else:
        mean_tone = 0.0
        relationship = "Neutral"

    return {
        "country_a": country_a,
        "country_b": country_b,
        "a_to_b": forward,
        "b_to_a": reverse,
        "relationship_type": relationship,
        "dominant_tone": round(mean_tone, 4),
    }
