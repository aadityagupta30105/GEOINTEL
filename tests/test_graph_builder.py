"""Tests for graph construction, centrality metrics and global statistics."""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd
import pytest

from analysis.graph_builder import (
    GGPI_MODULARITY_WEIGHT,
    GGPI_NEGATIVE_EDGE_WEIGHT,
    GGPI_NEGATIVE_TONE_WEIGHT,
    METRIC_COLUMNS,
    build_graph,
    build_temporal_graphs,
    compute_metrics,
    compute_network_stats,
    get_bilateral_summary,
    get_undirected,
)


class TestBuildGraph:
    """Edge and node aggregation over a fully known event frame."""

    def test_edges_match_directed_dyads(self, tiny_events: pd.DataFrame) -> None:
        graph = build_graph(tiny_events)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 2
        assert graph.has_edge("USA", "CHN")
        assert graph.has_edge("CHN", "USA")

    def test_edge_aggregates_are_exact(self, tiny_events: pd.DataFrame) -> None:
        edge = build_graph(tiny_events)["USA"]["CHN"]
        assert edge["num_events"] == 3
        assert edge["mentions"] == 35
        assert edge["tone"] == pytest.approx((-0.50 - 0.30 + 0.20) / 3, abs=1e-4)
        assert edge["conflict_count"] == 2
        assert edge["coop_count"] == 1
        assert edge["event_types"] == {
            "Conflict": 1, "Military/Conflict": 1, "Cooperation": 1
        }

    def test_distance_is_the_reciprocal_of_weight(self, tiny_events: pd.DataFrame) -> None:
        edge = build_graph(tiny_events)["USA"]["CHN"]
        assert edge["distance"] == pytest.approx(1.0 / edge["weight"])

    def test_node_total_events_sums_both_directions(self, tiny_events: pd.DataFrame) -> None:
        graph = build_graph(tiny_events)
        assert graph.nodes["USA"]["total_events"] == 4
        assert graph.nodes["CHN"]["total_events"] == 4

    def test_dominant_type_breaks_ties_alphabetically(self) -> None:
        frame = pd.DataFrame([
            {"Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
             "tone_norm": 0.1, "event_type": "Conflict", "NumMentions": 1},
            {"Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
             "tone_norm": 0.1, "event_type": "Cooperation", "NumMentions": 1},
        ])
        assert build_graph(frame)["USA"]["CHN"]["dominant_type"] == "Conflict"

    @pytest.mark.parametrize(
        ("scheme", "expected"),
        [("frequency", 3.0), ("mentions", 35.0)],
    )
    def test_weight_schemes(
        self, tiny_events: pd.DataFrame, scheme: str, expected: float
    ) -> None:
        edge = build_graph(tiny_events, weight_by=scheme)["USA"]["CHN"]
        assert edge["weight"] == pytest.approx(expected)

    def test_tone_scheme_uses_absolute_mean(self, tiny_events: pd.DataFrame) -> None:
        edge = build_graph(tiny_events, weight_by="tone")["USA"]["CHN"]
        assert edge["weight"] == pytest.approx(abs((-0.50 - 0.30 + 0.20) / 3) * 3)

    def test_empty_frame_yields_empty_graph(self) -> None:
        graph = build_graph(pd.DataFrame())
        assert graph.number_of_nodes() == 0

    def test_matches_the_reference_row_wise_aggregation(self, events: pd.DataFrame) -> None:
        """The vectorised builder must agree with a naive per-row aggregation."""
        graph = build_graph(events)

        expected: dict[tuple[str, str], dict[str, float]] = {}
        for row in events.itertuples(index=False):
            key = (row.Actor1CountryCode, row.Actor2CountryCode)
            bucket = expected.setdefault(key, {"n": 0, "tone": 0.0, "mentions": 0})
            bucket["n"] += 1
            bucket["tone"] += row.tone_norm
            bucket["mentions"] += int(row.NumMentions)

        assert graph.number_of_edges() == len(expected)
        for (source, target), bucket in expected.items():
            edge = graph[source][target]
            assert edge["num_events"] == bucket["n"]
            assert edge["mentions"] == bucket["mentions"]
            assert edge["tone"] == pytest.approx(bucket["tone"] / bucket["n"], abs=1e-4)


class TestUndirectedProjection:
    """Merge semantics when collapsing reciprocal edges."""

    def test_merges_reciprocal_edges(self, tiny_events: pd.DataFrame) -> None:
        undirected = get_undirected(build_graph(tiny_events))
        assert undirected.number_of_edges() == 1
        edge = undirected["USA"]["CHN"]
        assert edge["num_events"] == 4
        assert edge["conflict_count"] == 2
        assert edge["coop_count"] == 2

    def test_does_not_alias_nested_attributes(self, tiny_events: pd.DataFrame) -> None:
        graph = build_graph(tiny_events)
        undirected = get_undirected(graph)
        undirected["USA"]["CHN"]["event_types"]["Injected"] = 99
        assert "Injected" not in graph["USA"]["CHN"]["event_types"]


class TestMetrics:
    """Shape, dtype and ordering of the node metrics frame."""

    def test_columns_and_dtypes(self, graph: nx.DiGraph) -> None:
        metrics = compute_metrics(graph)
        assert list(metrics.columns) == list(METRIC_COLUMNS)
        assert (metrics.dtypes == "float64").all()
        assert metrics.index.name == "country"

    def test_sorted_by_pagerank_descending(self, graph: nx.DiGraph) -> None:
        pagerank = compute_metrics(graph)["pagerank"]
        assert pagerank.is_monotonic_decreasing

    def test_conflict_ratio_is_bounded(self, graph: nx.DiGraph) -> None:
        assert compute_metrics(graph)["conflict_ratio"].between(0.0, 1.0).all()

    def test_conflict_ratio_is_exact(self, tiny_events: pd.DataFrame) -> None:
        metrics = compute_metrics(build_graph(tiny_events))
        # USA outbound: 2 conflict, 1 cooperation.
        assert metrics.loc["USA", "conflict_ratio"] == pytest.approx(2 / 3, abs=1e-4)
        # CHN outbound: 0 conflict, 1 cooperation.
        assert metrics.loc["CHN", "conflict_ratio"] == 0.0

    def test_empty_graph_returns_typed_empty_frame(self) -> None:
        metrics = compute_metrics(nx.DiGraph())
        assert metrics.empty
        assert list(metrics.columns) == list(METRIC_COLUMNS)

    def test_betweenness_uses_distance_not_volume(self) -> None:
        """A high-volume path must be a short path, not a long one.

        The chain A-B-C carries heavy traffic while A-D-C is light. Under a
        correct distance metric the shortest path runs through B, so B must
        outrank D on betweenness.
        """
        frame = pd.DataFrame(
            [
                {"Actor1CountryCode": "AAA", "Actor2CountryCode": "BBB",
                 "tone_norm": 0.1, "event_type": "Cooperation", "NumMentions": 1}
            ] * 50
            + [
                {"Actor1CountryCode": "BBB", "Actor2CountryCode": "CCC",
                 "tone_norm": 0.1, "event_type": "Cooperation", "NumMentions": 1}
            ] * 50
            + [
                {"Actor1CountryCode": "AAA", "Actor2CountryCode": "DDD",
                 "tone_norm": 0.1, "event_type": "Cooperation", "NumMentions": 1},
                {"Actor1CountryCode": "DDD", "Actor2CountryCode": "CCC",
                 "tone_norm": 0.1, "event_type": "Cooperation", "NumMentions": 1},
            ]
        )
        metrics = compute_metrics(build_graph(frame))
        assert metrics.loc["BBB", "betweenness"] > metrics.loc["DDD", "betweenness"]


class TestNetworkStats:
    """Global topology statistics and the GGPI definition."""

    def test_required_keys_present(self, graph: nx.DiGraph) -> None:
        stats = compute_network_stats(graph)
        assert set(stats) == {
            "nodes", "edges", "density", "reciprocity", "avg_tone",
            "negative_edge_ratio", "modularity", "num_communities",
            "ggpi", "community_map",
        }

    def test_ggpi_matches_the_documented_formula(self, graph: nx.DiGraph) -> None:
        stats = compute_network_stats(graph)
        expected = (
            GGPI_MODULARITY_WEIGHT * stats["modularity"]
            + GGPI_NEGATIVE_EDGE_WEIGHT * stats["negative_edge_ratio"]
            + GGPI_NEGATIVE_TONE_WEIGHT * max(0.0, -stats["avg_tone"])
        )
        assert stats["ggpi"] == pytest.approx(expected, abs=5e-4)

    def test_ggpi_weights_sum_to_one(self) -> None:
        total = (
            GGPI_MODULARITY_WEIGHT
            + GGPI_NEGATIVE_EDGE_WEIGHT
            + GGPI_NEGATIVE_TONE_WEIGHT
        )
        assert total == pytest.approx(1.0)

    def test_ggpi_is_bounded(self, graph: nx.DiGraph) -> None:
        assert 0.0 <= compute_network_stats(graph)["ggpi"] <= 1.0

    def test_negative_edge_ratio_is_exact(self, tiny_events: pd.DataFrame) -> None:
        stats = compute_network_stats(build_graph(tiny_events))
        # USA->CHN mean tone is negative; CHN->USA is positive.
        assert stats["negative_edge_ratio"] == pytest.approx(0.5)

    def test_empty_graph_does_not_raise(self) -> None:
        """overall_reciprocity is undefined for empty graphs and must be guarded."""
        stats = compute_network_stats(nx.DiGraph())
        assert stats["nodes"] == 0
        assert stats["reciprocity"] == 0.0
        assert stats["ggpi"] == 0.0
        assert stats["num_communities"] == 0

    def test_single_node_graph_does_not_raise(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("USA")
        stats = compute_network_stats(graph)
        assert stats["nodes"] == 1
        assert stats["density"] == 0.0
        assert math.isfinite(stats["ggpi"])

    def test_stats_are_json_serialisable_without_community_map(
        self, graph: nx.DiGraph
    ) -> None:
        import json

        stats = compute_network_stats(graph)
        payload = {k: v for k, v in stats.items() if k != "community_map"}
        assert json.loads(json.dumps(payload)) == payload


class TestTemporalGraphs:
    """Period slicing behaviour."""

    def test_month_slicing_labels(self, events: pd.DataFrame) -> None:
        graphs = build_temporal_graphs(events, period="month")
        assert set(graphs) == {"2024-01", "2024-02", "2024-03"}

    def test_quarter_slicing_labels(self, events: pd.DataFrame) -> None:
        assert set(build_temporal_graphs(events, period="quarter")) == {"2024Q1"}

    def test_year_slicing_labels(self, events: pd.DataFrame) -> None:
        assert set(build_temporal_graphs(events, period="year")) == {"2024"}

    def test_slices_partition_the_event_stream(self, events: pd.DataFrame) -> None:
        graphs = build_temporal_graphs(events, period="month")
        total = sum(
            data["num_events"]
            for graph in graphs.values()
            for _, _, data in graph.edges(data=True)
        )
        assert total == len(events)

    def test_empty_frame_returns_no_graphs(self) -> None:
        assert build_temporal_graphs(pd.DataFrame()) == {}

    def test_unparseable_dates_return_no_graphs(self) -> None:
        frame = pd.DataFrame([{
            "Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
            "tone_norm": 0.1, "event_type": "Cooperation",
            "NumMentions": 1, "date": "not-a-date",
        }])
        assert build_temporal_graphs(frame) == {}


class TestBilateralSummary:
    """Directed relationship extraction."""

    def test_returns_both_directions(self, tiny_events: pd.DataFrame) -> None:
        summary = get_bilateral_summary(build_graph(tiny_events), "USA", "CHN")
        assert summary["a_to_b"] is not None
        assert summary["b_to_a"] is not None
        assert summary["country_a"] == "USA"

    def test_missing_pair_is_neutral(self, graph: nx.DiGraph) -> None:
        summary = get_bilateral_summary(graph, "ZZZ", "QQQ")
        assert summary["relationship_type"] == "Neutral"
        assert summary["dominant_tone"] == 0.0
        assert summary["a_to_b"] is None

    def test_one_directional_pair_is_classified(self) -> None:
        frame = pd.DataFrame([{
            "Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
            "tone_norm": -0.8, "event_type": "Conflict", "NumMentions": 1,
        }])
        summary = get_bilateral_summary(build_graph(frame), "USA", "CHN")
        assert summary["relationship_type"] == "Conflictual"
        assert summary["b_to_a"] is None

    def test_returned_edges_are_copies(self, tiny_events: pd.DataFrame) -> None:
        graph = build_graph(tiny_events)
        summary = get_bilateral_summary(graph, "USA", "CHN")
        summary["a_to_b"]["tone"] = 99.0
        assert graph["USA"]["CHN"]["tone"] != 99.0
