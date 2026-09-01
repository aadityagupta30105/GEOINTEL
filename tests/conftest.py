"""Shared pytest fixtures for the GeoIntel test suite."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.graph_builder import build_graph  # noqa: E402
from data.gdelt_collector import generate_mock_data, preprocess  # noqa: E402

WINDOW_START = datetime(2024, 1, 1)
WINDOW_END = datetime(2024, 3, 31)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return _ROOT


@pytest.fixture(scope="session")
def raw_events() -> pd.DataFrame:
    """Deterministic synthetic events, before preprocessing."""
    return generate_mock_data(WINDOW_START, WINDOW_END, n_events=1500, seed=1234)


@pytest.fixture(scope="session")
def events(raw_events: pd.DataFrame) -> pd.DataFrame:
    """Deterministic preprocessed event frame."""
    return preprocess(raw_events)


@pytest.fixture(scope="session")
def graph(events: pd.DataFrame) -> nx.DiGraph:
    """Static interaction graph over the deterministic event frame."""
    return build_graph(events)


@pytest.fixture
def tiny_events() -> pd.DataFrame:
    """A hand-built event frame with fully known aggregates.

    Two directed dyads are present. ``USA -> CHN`` carries three events (two
    conflictual, one cooperative) and ``CHN -> USA`` carries one cooperative
    event, so every derived edge attribute is checkable by hand.
    """
    return pd.DataFrame([
        {"Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
         "tone_norm": -0.50, "event_type": "Conflict", "NumMentions": 10,
         "date": "2024-01-05", "EventRootCode": "11", "QuadClass": 3},
        {"Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
         "tone_norm": -0.30, "event_type": "Military/Conflict", "NumMentions": 20,
         "date": "2024-01-06", "EventRootCode": "19", "QuadClass": 4},
        {"Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
         "tone_norm": 0.20, "event_type": "Cooperation", "NumMentions": 5,
         "date": "2024-02-07", "EventRootCode": "04", "QuadClass": 1},
        {"Actor1CountryCode": "CHN", "Actor2CountryCode": "USA",
         "tone_norm": 0.40, "event_type": "Trade/Aid", "NumMentions": 7,
         "date": "2024-02-08", "EventRootCode": "06", "QuadClass": 2},
    ])
