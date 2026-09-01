"""Tests for the dashboard visualisation layer and bloc analysis."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import pytest

from analysis.graph_builder import build_graph, compute_metrics, compute_network_stats
from dashboard.blocs import (
    GEOPOLITICAL_PRIORS,
    POLE_PRESETS,
    PRIOR_WEIGHT,
    bilateral_tone,
    compute_blocs,
)
from dashboard.figures import (
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
from dashboard.geodata import COUNTRY_GEO, MAX_GDP_PPP_BILLIONS, format_gdp, marker_size


@pytest.fixture(scope="module")
def analysed(events: pd.DataFrame) -> tuple[nx.DiGraph, pd.DataFrame, dict]:
    """Graph, metrics and statistics over the deterministic event frame."""
    graph = build_graph(events)
    return graph, compute_metrics(graph), compute_network_stats(graph)


class TestGeodata:
    """Reference geography integrity."""

    def test_coordinates_are_valid(self) -> None:
        for code, entry in COUNTRY_GEO.items():
            assert -90 <= entry.lat <= 90, code
            assert -180 <= entry.lon <= 180, code

    def test_codes_are_iso3_shaped(self) -> None:
        for code in COUNTRY_GEO:
            assert len(code) == 3 and code.isupper(), code

    def test_gdp_values_are_positive(self) -> None:
        for code, entry in COUNTRY_GEO.items():
            assert entry.gdp_ppp_billions > 0, code

    def test_synthetic_countries_are_mappable(self) -> None:
        """Every code the generator can emit must have reference geography."""
        from data.gdelt_collector import _MOCK_COUNTRIES

        missing = sorted(set(_MOCK_COUNTRIES) - set(COUNTRY_GEO))
        assert not missing, f"Synthetic codes without geography: {missing}"

    def test_prior_countries_are_mappable(self) -> None:
        missing = sorted(set(GEOPOLITICAL_PRIORS) - set(COUNTRY_GEO))
        assert not missing, f"Prior codes without geography: {missing}"


class TestGdpPresentation:
    """GDP is stored in billions; display must convert correctly."""

    def test_trillion_scale_conversion(self) -> None:
        """Regression: 26900 billion previously rendered as '$26900.0T'."""
        assert format_gdp(26900) == "$26.9T"
        assert format_gdp(33000) == "$33.0T"

    def test_billion_scale_values(self) -> None:
        assert format_gdp(520) == "$520B"
        assert format_gdp(5) == "$5B"

    def test_marker_size_is_monotonic_in_gdp(self) -> None:
        assert marker_size(100) < marker_size(1000) < marker_size(MAX_GDP_PPP_BILLIONS)

    def test_marker_size_respects_the_base_floor(self) -> None:
        assert marker_size(0) == pytest.approx(8.0)


class TestBlocs:
    """Bloc scoring and assignment."""

    def test_presets_reference_known_codes(self) -> None:
        for poles in POLE_PRESETS.values():
            for code in poles:
                assert code in COUNTRY_GEO

    def test_prior_weight_is_a_valid_blend(self) -> None:
        assert 0.0 <= PRIOR_WEIGHT <= 1.0

    def test_priors_are_within_range(self) -> None:
        for country, scores in GEOPOLITICAL_PRIORS.items():
            for pole, value in scores.items():
                assert -1.0 <= value <= 1.0, f"{country}/{pole}"

    def test_every_country_is_assigned(self, analysed: tuple) -> None:
        graph, _, _ = analysed
        assignments, _ = compute_blocs(graph, ["USA", "CHN"])
        assert set(assignments) == set(graph.nodes())

    def test_poles_belong_to_their_own_bloc(self, analysed: tuple) -> None:
        graph, _, _ = analysed
        assignments, _ = compute_blocs(graph, ["USA", "CHN", "RUS"])
        for pole in ("USA", "CHN", "RUS"):
            assert assignments[pole] == pole

    def test_affinity_covers_every_pole(self, analysed: tuple) -> None:
        graph, _, _ = analysed
        poles = ["USA", "CHN"]
        _, affinity = compute_blocs(graph, poles)
        for scores in affinity.values():
            assert set(scores) == set(poles)

    def test_poles_are_absent_from_the_affinity_table(self, analysed: tuple) -> None:
        graph, _, _ = analysed
        _, affinity = compute_blocs(graph, ["USA", "CHN"])
        assert "USA" not in affinity and "CHN" not in affinity

    def test_priors_anchor_assignment_on_sparse_data(self) -> None:
        """A strongly aligned state must not drift on a near-empty network."""
        frame = pd.DataFrame([
            {"Actor1CountryCode": "UKR", "Actor2CountryCode": "RUS",
             "tone_norm": 0.0, "event_type": "Diplomatic", "NumMentions": 1},
            {"Actor1CountryCode": "UKR", "Actor2CountryCode": "USA",
             "tone_norm": 0.0, "event_type": "Diplomatic", "NumMentions": 1},
        ])
        assignments, _ = compute_blocs(build_graph(frame), ["USA", "RUS"])
        assert assignments["UKR"] == "USA"

    def test_bilateral_tone_averages_both_directions(
        self, tiny_events: pd.DataFrame
    ) -> None:
        graph = build_graph(tiny_events)
        expected = (graph["USA"]["CHN"]["tone"] + graph["CHN"]["USA"]["tone"]) / 2
        assert bilateral_tone(graph, "USA", "CHN") == pytest.approx(expected)

    def test_bilateral_tone_for_absent_dyad_is_zero(self, analysed: tuple) -> None:
        graph, _, _ = analysed
        assert bilateral_tone(graph, "ZZZ", "QQQ") == 0.0

    def test_assignment_is_deterministic(self, analysed: tuple) -> None:
        graph, _, _ = analysed
        first, _ = compute_blocs(graph, ["USA", "CHN", "RUS"])
        second, _ = compute_blocs(graph, ["USA", "CHN", "RUS"])
        assert first == second


class TestFigures:
    """Every builder must return a renderable figure."""

    def test_network_figure(self, analysed: tuple) -> None:
        graph, metrics, stats = analysed
        figure = build_network_figure(graph, metrics, stats, 1, 25, "PageRank")
        assert isinstance(figure, go.Figure)
        assert len(figure.data) >= 1

    @pytest.mark.parametrize("color_by", ["PageRank", "Conflict Ratio", "Community"])
    def test_network_figure_color_modes(self, analysed: tuple, color_by: str) -> None:
        graph, metrics, stats = analysed
        assert build_network_figure(graph, metrics, stats, 1, 20, color_by).data

    def test_bloc_figure(self, analysed: tuple) -> None:
        graph, metrics, _ = analysed
        poles = ["USA", "CHN"]
        assignments, affinity = compute_blocs(graph, poles)
        figure = build_bloc_figure(graph, metrics, assignments, affinity, poles, 1)
        legend_names = {trace.name for trace in figure.data if trace.name}
        assert legend_names == {"USA bloc", "CHN bloc"}

    def test_radar_chart(self, analysed: tuple) -> None:
        _, metrics, _ = analysed
        figure = build_radar_chart(metrics, metrics.index[0])
        assert len(figure.data) == 1

    def test_radar_chart_for_unknown_country(self, analysed: tuple) -> None:
        _, metrics, _ = analysed
        assert build_radar_chart(metrics, "ZZZ").data == ()

    def test_bilateral_chart_both_directions(self, tiny_events: pd.DataFrame) -> None:
        figure = build_bilateral_chart(build_graph(tiny_events), "USA", "CHN")
        assert len(figure.data) == 2

    def test_bilateral_chart_one_direction_does_not_raise(self) -> None:
        """Regression: the reverse edge was dereferenced without a guard."""
        frame = pd.DataFrame([{
            "Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
            "tone_norm": 0.2, "event_type": "Cooperation", "NumMentions": 1,
        }])
        graph = build_graph(frame)
        assert len(build_bilateral_chart(graph, "USA", "CHN").data) == 2
        assert len(build_bilateral_chart(graph, "CHN", "USA").data) == 2

    def test_bilateral_chart_absent_pair_is_empty(self, analysed: tuple) -> None:
        graph, _, _ = analysed
        assert build_bilateral_chart(graph, "ZZZ", "QQQ").data == ()

    def test_tone_heatmap_is_square(self, analysed: tuple) -> None:
        graph, metrics, _ = analysed
        figure = build_tone_heatmap(graph, metrics, top_n=10)
        assert figure.data[0].z.shape == (10, 10)

    def test_pagerank_bar(self, analysed: tuple) -> None:
        _, metrics, _ = analysed
        figure = build_pagerank_bar(metrics, top_n=10)
        assert len(figure.data[0].x) == 10

    def test_temporal_line_and_activity(self) -> None:
        frame = pd.DataFrame({
            "period": ["2024-01", "2024-02"],
            "avg_tone": [0.01, -0.02],
            "ggpi": [0.28, 0.31],
            "edges": [1000, 900],
            "conflict_rate": [0.48, 0.51],
        })
        assert build_temporal_line(frame, "ggpi", "GGPI", "#ff6b35").data
        assert len(build_activity_figure(frame).data) == 2

    def test_temporal_axes_are_categorical(self) -> None:
        """Period labels are ordinal; Plotly must not parse them as dates."""
        frame = pd.DataFrame({
            "period": ["2024-01", "2024-02", "2024-03"],
            "ggpi": [0.28, 0.31, 0.29],
            "edges": [1000, 900, 950],
            "conflict_rate": [0.48, 0.51, 0.49],
        })
        assert build_temporal_line(frame, "ggpi", "GGPI", "#ff6b35").layout.xaxis.type == "category"
        assert build_activity_figure(frame).layout.xaxis.type == "category"

    def test_figures_use_transparent_or_platform_backgrounds(
        self, analysed: tuple
    ) -> None:
        _, metrics, _ = analysed
        layout = build_pagerank_bar(metrics).layout
        assert layout.paper_bgcolor == "rgba(0,0,0,0)"
        assert layout.plot_bgcolor == "rgba(0,0,0,0)"

    def test_cartesian_figures_hide_gridlines(self, analysed: tuple) -> None:
        _, metrics, _ = analysed
        layout = build_pagerank_bar(metrics).layout
        assert layout.xaxis.showgrid is False
        assert layout.yaxis.showgrid is False

    def test_stat_rows_exclude_the_community_map(self, analysed: tuple) -> None:
        _, _, stats = analysed
        labels = {label for label, _ in build_stat_rows(stats)}
        assert "Community Map" not in labels
        assert "Ggpi" in labels or "GGPI" in labels or "Ggpi" in labels


class TestTableShading:
    """Palette-native gradients replacing matplotlib colour maps.

    ``Styler.background_gradient`` raises ``ImportError`` without matplotlib,
    which is not a platform dependency. These helpers must cover the same
    ground using only the platform palette.
    """

    def test_sequential_gradient_intensity_increases_with_value(self) -> None:
        from dashboard.theme import ACCENT, sequential_gradient

        low, _, high = sequential_gradient([0.1, 0.5, 0.9], ACCENT)
        assert _alpha_of(low) < _alpha_of(high)

    def test_sequential_gradient_uses_the_requested_colour(self) -> None:
        from dashboard.theme import NEGATIVE, sequential_gradient

        assert "rgba(239,68,68" in sequential_gradient([1.0, 2.0], NEGATIVE)[1]

    def test_sequential_gradient_handles_a_constant_column(self) -> None:
        from dashboard.theme import ACCENT, sequential_gradient

        styles = sequential_gradient([5.0, 5.0, 5.0], ACCENT)
        assert len(set(styles)) == 1

    def test_sequential_gradient_handles_empty_and_nan(self) -> None:
        from dashboard.theme import ACCENT, sequential_gradient

        assert sequential_gradient([], ACCENT) == []
        assert sequential_gradient([float("nan")], ACCENT) == [""]

    def test_diverging_gradient_splits_on_sign(self) -> None:
        from dashboard.theme import diverging_gradient

        negative, _, positive = diverging_gradient([-0.8, 0.0, 0.8])
        assert "rgba(239,68,68" in negative
        assert "rgba(34,197,94" in positive

    def test_diverging_gradient_intensity_tracks_distance(self) -> None:
        from dashboard.theme import diverging_gradient

        near, far = diverging_gradient([0.1, 1.0])
        assert _alpha_of(near) < _alpha_of(far)

    def test_styler_accepts_the_helpers(self) -> None:
        """The end-to-end path that previously raised ImportError."""
        from dashboard.theme import ACCENT, NEGATIVE, sequential_gradient

        frame = pd.DataFrame({"PageRank": [0.1, 0.3], "Conflict %": [10.0, 90.0]})
        html = (
            frame.style
            .apply(sequential_gradient, hex_color=ACCENT, subset=["PageRank"])
            .apply(sequential_gradient, hex_color=NEGATIVE, subset=["Conflict %"])
            .to_html()
        )
        assert html.count("background-color") == 4

    def test_rgba_conversion(self) -> None:
        from dashboard.theme import rgba

        assert rgba("#00d4ff", 0.5) == "rgba(0,212,255,0.500)"


def _alpha_of(style: str) -> float:
    """Extract the alpha channel from a generated cell style.

    Parameters
    ----------
    style : str
        CSS declaration produced by a gradient helper.

    Returns
    -------
    float
        The alpha component.
    """
    return float(style.split("rgba(")[1].split(")")[0].split(",")[3])


class TestAppHelpers:
    """Application-shell helpers that do not require a Streamlit session."""

    def test_fingerprint_is_stable(self, events: pd.DataFrame) -> None:
        from dashboard.app import fingerprint

        assert fingerprint(events) == fingerprint(events.copy())

    def test_fingerprint_detects_filtering(self, events: pd.DataFrame) -> None:
        from dashboard.app import fingerprint

        subset = events[events["event_type"] == events["event_type"].iloc[0]]
        assert fingerprint(events) != fingerprint(subset)

    def test_fingerprint_detects_content_change_at_equal_length(
        self, events: pd.DataFrame
    ) -> None:
        """A length-only key would collide here; the content hash must not."""
        from dashboard.app import fingerprint

        mutated = events.copy()
        mutated.loc[mutated.index[0], "tone_norm"] = 0.999
        assert fingerprint(events) != fingerprint(mutated)

    def test_fingerprint_of_empty_frame(self) -> None:
        from dashboard.app import fingerprint

        assert fingerprint(pd.DataFrame()) == "empty"
