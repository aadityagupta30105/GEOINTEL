"""
GDELT Data Collector
====================
Fetches country-level geopolitical event data from GDELT 1.0 daily exports and
normalises it into the schema consumed by the graph layer. Falls back to a
synthetic generator whenever the remote service is unavailable or yields no
state-to-state events.

Schema notes (regression guards)
--------------------------------
1. ENCODING: GDELT export files are Latin-1, not UTF-8.
2. COLUMN COUNT: GDELT 1.0 has 58 columns, not 44. A truncated column list
   omits the 14 Actor sub-fields (KnownGroupCode, EthnicCode, Religion1/2Code,
   Type1/2/3Code) between Actor1 and Actor2, which shifts every name after
   index 7 and maps ``Actor2CountryCode`` onto index 10
   (``Actor1Religion1Code``). That field is always blank, yielding zero
   bilateral events.
3. ROW LIMITS: the file must be read in full before filtering. GDELT orders
   rows by event id, not actor type, so an ``nrows`` prefix is dominated by
   sub-national and non-state actors with blank country codes.
"""

from __future__ import annotations

import io
import random
from datetime import datetime, timedelta
from typing import Final

import numpy as np
import pandas as pd
import requests

from utils.logging_config import ARROW, ERROR, OK, WARN, get_logger

__all__ = [
    "GDELT_COLS",
    "QUAD_CLASS_MAP",
    "EVENT_ROOT_MAP",
    "REQUIRED_COLUMNS",
    "get_gdelt_url",
    "fetch_gdelt_day",
    "collect_gdelt_range",
    "generate_mock_data",
    "preprocess",
]

_log = get_logger(__name__)


# --- GDELT 1.0 complete 58-column schema ------------------------------------
GDELT_COLS: Final[list[str]] = [
    # 0-4
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    # 5-14  Actor1 (10 fields)
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    # 15-24  Actor2 (10 fields); Actor2CountryCode is index 17, not 10
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode",
    "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    # 25-34  Event
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    # 35-41  Actor1 Geo
    "Actor1Geo_Type", "Actor1Geo_Fullname", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    # 42-48  Actor2 Geo
    "Actor2Geo_Type", "Actor2Geo_Fullname", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    # 49-55  Action Geo
    "ActionGeo_Type", "ActionGeo_Fullname", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    # 56-57
    "DATEADDED", "SOURCEURL",
]
assert len(GDELT_COLS) == 58, f"Expected 58 GDELT columns, got {len(GDELT_COLS)}"

# CAMEO QuadClass mapping.
QUAD_CLASS_MAP: Final[dict[int, str]] = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict",
}

# CAMEO EventRootCode to human-readable label.
EVENT_ROOT_MAP: Final[dict[str, str]] = {
    "01": "Make Public Statement",
    "02": "Appeal",
    "03": "Express Intent to Cooperate",
    "04": "Consult",
    "05": "Engage in Diplomatic Cooperation",
    "06": "Engage in Material Cooperation",
    "07": "Provide Aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit Force Posture",
    "16": "Reduce Relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Use Unconventional Mass Violence",
}

# Columns the graph layer depends on. Missing entries are materialised as null.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Actor1CountryCode", "Actor2CountryCode", "AvgTone",
    "QuadClass", "EventRootCode", "date", "NumMentions", "GoldsteinScale",
)

# Event root codes that override the QuadClass-derived classification.
_TRADE_AID_CODES: Final[frozenset[str]] = frozenset({"06", "07"})
_MILITARY_CODES: Final[frozenset[str]] = frozenset({"18", "19", "20"})

_ISO3_LENGTH: Final[int] = 3
_TONE_CLIP: Final[float] = 20.0
_HTTP_TIMEOUT_SECONDS: Final[int] = 90
_STREAM_CHUNK_BYTES: Final[int] = 131_072
_SAMPLE_SEED: Final[int] = 42


def get_gdelt_url(date: datetime) -> str:
    """Build the GDELT 1.0 daily export URL for a calendar date.

    Parameters
    ----------
    date : datetime
        The day to address. Only the date component is used.

    Returns
    -------
    str
        Fully qualified URL of the zipped daily export.
    """
    return f"http://data.gdeltproject.org/events/{date.strftime('%Y%m%d')}.export.CSV.zip"


def _filter_bilateral(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain rows where both actors carry distinct 3-letter country codes.

    Parameters
    ----------
    frame : pandas.DataFrame
        Raw GDELT frame carrying the full 58-column schema.

    Returns
    -------
    pandas.DataFrame
        A copy containing only state-to-state events.
    """
    actor1 = frame["Actor1CountryCode"].fillna("").astype(str).str.strip()
    actor2 = frame["Actor2CountryCode"].fillna("").astype(str).str.strip()
    mask = (
        (actor1.str.len() == _ISO3_LENGTH)
        & (actor2.str.len() == _ISO3_LENGTH)
        & (actor1 != actor2)
    )
    return frame[mask].copy()


def fetch_gdelt_day(date: datetime, target_rows: int = 2000) -> pd.DataFrame | None:
    """Download one day of GDELT events and retain state-to-state interactions.

    The full file is parsed before filtering because GDELT orders rows by event
    identifier; a row prefix is not representative of bilateral activity.

    Parameters
    ----------
    date : datetime
        Day to download.
    target_rows : int, optional
        Upper bound on retained rows. Sampling is seeded for reproducibility.

    Returns
    -------
    pandas.DataFrame or None
        Bilateral events for the day, or ``None`` when the fetch failed or the
        export contained no state-to-state events.
    """
    url = get_gdelt_url(date)
    day = date.strftime("%Y-%m-%d")

    try:
        _log.info("Fetching GDELT export: %s", url)
        response = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()

        # Accumulate raw ZIP bytes without any intermediate text decoding.
        chunks: list[bytes] = [
            chunk
            for chunk in response.iter_content(chunk_size=_STREAM_CHUNK_BYTES)
            if chunk
        ]
        raw_bytes = b"".join(chunks)

        frame = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep="\t",
            header=None,
            names=GDELT_COLS,
            dtype=str,
            encoding="latin-1",
            on_bad_lines="skip",
            compression="zip",
        )

        bilateral = _filter_bilateral(frame)
        total = len(bilateral)

        if total == 0:
            _log.warning("%s %s: no bilateral events in export", WARN, day)
            return None

        if total > target_rows:
            bilateral = bilateral.sample(n=target_rows, random_state=_SAMPLE_SEED)

        _log.info(
            "%s %s: %s bilateral events (retaining %s)",
            OK, day, f"{total:,}", f"{len(bilateral):,}",
        )
        return bilateral

    except KeyboardInterrupt:
        _log.warning("%s GDELT fetch interrupted by operator", WARN)
        raise
    except requests.RequestException as exc:
        _log.error("%s GDELT transport failure (%s): %s", ERROR, day, exc)
        return None
    except (ValueError, OSError) as exc:
        _log.error("%s GDELT parse failure (%s): %s", ERROR, day, exc)
        return None


def collect_gdelt_range(
    start: datetime,
    end: datetime,
    target_rows_per_day: int = 2000,
) -> pd.DataFrame:
    """Collect GDELT events across an inclusive date range.

    Individual day failures are tolerated. When no day yields bilateral events
    the synthetic generator is used, so downstream stages always receive data.

    Parameters
    ----------
    start : datetime
        First day to collect, inclusive.
    end : datetime
        Last day to collect, inclusive.
    target_rows_per_day : int, optional
        Per-day retention cap forwarded to :func:`fetch_gdelt_day`.

    Returns
    -------
    pandas.DataFrame
        Concatenated raw events, or synthetic events on total failure.
    """
    frames: list[pd.DataFrame] = []
    current = start

    try:
        while current <= end:
            day_frame = fetch_gdelt_day(current, target_rows_per_day)
            if day_frame is not None:
                day_frame["date"] = current.strftime("%Y-%m-%d")
                frames.append(day_frame)
            current += timedelta(days=1)
    except KeyboardInterrupt:
        _log.warning("%s Collection interrupted; using days retrieved so far", WARN)

    if not frames:
        _log.warning(
            "%s No GDELT bilateral events retrieved %s synthetic fallback",
            WARN, ARROW,
        )
        return generate_mock_data(start, end)

    combined = pd.concat(frames, ignore_index=True)
    _log.info(
        "%s GDELT collection complete: %s events across %d day(s)",
        OK, f"{len(combined):,}", len(frames),
    )
    return combined


# --- Synthetic generator ----------------------------------------------------

# GDELT-compatible ISO-3 country codes used by the synthetic generator.
_MOCK_COUNTRIES: Final[tuple[str, ...]] = (
    # Major powers
    "USA", "CHN", "RUS", "DEU", "GBR", "FRA", "IND", "BRA", "JPN", "KOR",
    "ISR", "IRN", "SAU", "TUR", "PAK", "NGA", "ZAF", "EGY", "MEX", "ARG",
    "IDN", "AUS", "CAN", "ITA", "UKR", "POL", "NLD", "SWE", "NOR", "CHL",
    # Europe
    "ESP", "PRT", "BEL", "AUT", "CHE", "DNK", "FIN", "GRC", "CZE", "HUN",
    "ROU", "BGR", "HRV", "SVK", "SVN", "SRB", "BLR", "MDA", "ALB", "LTU",
    "LVA", "EST", "BIH", "MKD", "MNE", "IRL", "LUX", "XKX",
    # Middle East and Central Asia
    "IRQ", "SYR", "JOR", "LBN", "YEM", "OMN", "ARE", "QAT", "KWT", "BHR",
    "AFG", "KAZ", "UZB", "TKM", "TJK", "KGZ", "AZE", "ARM", "GEO",
    # Asia-Pacific
    "VNM", "THA", "MYS", "PHL", "SGP", "BGD", "LKA", "NPL", "MMR", "KHM",
    "LAO", "MNG", "NZL", "PNG", "TWN", "PRK",
    # Africa
    "ETH", "ERI", "TZA", "KEN", "GHA", "CIV", "AGO", "CMR", "MOZ", "MDG",
    "ZMB", "ZWE", "SEN", "MWI", "MLI", "BFA", "NER", "TCD", "SDN", "SSD",
    "LBY", "TUN", "DZA", "MAR", "COD", "UGA", "RWA", "SOM", "DJI", "GAB",
    "COG", "TGO", "BEN", "LBR", "SLE", "GIN", "MRT", "NAM", "BWA",
    # Americas
    "COL", "VEN", "PER", "ECU", "BOL", "PRY", "URY", "GTM", "HND", "SLV",
    "NIC", "CRI", "PAN", "CUB", "DOM", "HTI", "JAM", "TTO", "GUY", "SUR",
    # Oceania
    "FJI",
)

# Dyads biased toward negative tone and conflictual quad classes.
_MOCK_TENSIONS: Final[frozenset[tuple[str, str]]] = frozenset({
    ("USA", "CHN"), ("USA", "RUS"), ("USA", "IRN"), ("USA", "PRK"), ("USA", "CUB"),
    ("CHN", "IND"), ("CHN", "TWN"), ("CHN", "JPN"), ("CHN", "VNM"), ("CHN", "PHL"),
    ("RUS", "UKR"), ("RUS", "POL"), ("RUS", "LTU"), ("RUS", "LVA"), ("RUS", "EST"),
    ("RUS", "GEO"), ("RUS", "AZE"), ("ISR", "IRN"), ("ISR", "SYR"), ("ISR", "LBN"),
    ("IND", "PAK"), ("SAU", "IRN"), ("SAU", "YEM"), ("ETH", "ERI"), ("ETH", "SOM"),
    ("SDN", "SSD"), ("SYR", "TUR"), ("ARM", "AZE"), ("GRC", "TUR"), ("SRB", "XKX"),
    ("IRQ", "SYR"), ("AFG", "PAK"),
})

# Dyads biased toward positive tone and cooperative quad classes.
_MOCK_ALLIANCES: Final[frozenset[tuple[str, str]]] = frozenset({
    ("USA", "GBR"), ("USA", "DEU"), ("USA", "JPN"), ("USA", "AUS"), ("USA", "CAN"),
    ("USA", "KOR"), ("USA", "ISR"), ("USA", "FRA"), ("USA", "ITA"), ("USA", "ESP"),
    ("USA", "NLD"), ("USA", "POL"), ("USA", "NOR"), ("USA", "DNK"), ("USA", "BEL"),
    ("USA", "COL"), ("USA", "MEX"), ("USA", "BRA"), ("USA", "SAU"), ("USA", "ARE"),
    ("CHN", "RUS"), ("CHN", "PAK"), ("CHN", "KAZ"), ("CHN", "MNG"), ("CHN", "IRN"),
    ("CHN", "PRK"), ("CHN", "VNM"), ("CHN", "THA"), ("CHN", "IDN"), ("CHN", "SAU"),
    ("RUS", "BLR"), ("RUS", "KAZ"), ("RUS", "ARM"), ("RUS", "IRN"), ("RUS", "SYR"),
    ("DEU", "FRA"), ("DEU", "NLD"), ("DEU", "AUT"), ("DEU", "POL"), ("GBR", "AUS"),
    ("GBR", "CAN"), ("GBR", "IRL"), ("FRA", "ESP"), ("FRA", "ITA"), ("FRA", "BEL"),
    ("JPN", "KOR"), ("JPN", "AUS"), ("KOR", "AUS"), ("IND", "RUS"), ("IND", "ISR"),
    ("BRA", "ARG"), ("BRA", "COL"), ("ARE", "SAU"), ("ARE", "QAT"), ("EGY", "SAU"),
    ("NGA", "ZAF"), ("KEN", "ETH"), ("GHA", "NGA"),
})


def _dyad_bias(
    country_a: str,
    country_b: str,
    rng: random.Random,
) -> tuple[float, int]:
    """Sample tone and quad class conditioned on a dyad prior alignment.

    Parameters
    ----------
    country_a, country_b : str
        ISO-3 codes of the ordered dyad. Membership tests are direction-free.
    rng : random.Random
        Random source, injected so callers control reproducibility.

    Returns
    -------
    tuple of (float, int)
        Sampled average tone and CAMEO quad class.
    """
    forward = (country_a, country_b)
    reverse = (country_b, country_a)

    if forward in _MOCK_TENSIONS or reverse in _MOCK_TENSIONS:
        return rng.gauss(-3.5, 2.5), rng.choice([3, 3, 3, 4, 4])
    if forward in _MOCK_ALLIANCES or reverse in _MOCK_ALLIANCES:
        return rng.gauss(3.0, 2.0), rng.choice([1, 1, 2, 2])
    return rng.gauss(0.0, 3.0), rng.choice([1, 2, 3, 4])


def generate_mock_data(
    start: datetime,
    end: datetime,
    n_events: int = 5000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic geopolitical events for offline operation.

    Tone and quad class are conditioned on curated tension and alliance dyads
    so the resulting network reproduces realistic bloc structure.

    Parameters
    ----------
    start : datetime
        First day of the simulated window, inclusive.
    end : datetime
        Last day of the simulated window, inclusive.
    n_events : int, optional
        Number of events to synthesise.
    seed : int or None, optional
        Seed for the local random source. ``None`` draws from system entropy.

    Returns
    -------
    pandas.DataFrame
        Raw synthetic events in the schema accepted by :func:`preprocess`.
    """
    rng = random.Random(seed)
    span_days = max(1, (end - start).days + 1)
    event_codes = list(EVENT_ROOT_MAP.keys())

    records: list[dict[str, str | int]] = []
    for _ in range(n_events):
        actor1, actor2 = rng.sample(_MOCK_COUNTRIES, 2)
        tone, quad = _dyad_bias(actor1, actor2, rng)
        event_date = start + timedelta(days=rng.randint(0, span_days - 1))
        records.append({
            "Actor1CountryCode": actor1,
            "Actor2CountryCode": actor2,
            "EventRootCode": rng.choice(event_codes),
            "QuadClass": quad,
            "GoldsteinScale": str(round(rng.uniform(-10, 10), 1)),
            "AvgTone": str(round(tone, 2)),
            "NumMentions": str(rng.randint(1, 200)),
            "NumArticles": str(rng.randint(1, 100)),
            "date": event_date.strftime("%Y-%m-%d"),
            "SOURCEURL": f"https://synthetic.invalid/{rng.randint(10000, 99999)}",
        })

    frame = pd.DataFrame(records)
    _log.info(
        "%s Synthetic generator produced %s events across %d day(s)",
        OK, f"{len(frame):,}", span_days,
    )
    return frame


# --- Preprocessing ----------------------------------------------------------

def _normalise_actor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename case- and separator-variant actor columns to canonical names.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame whose columns may use variant spellings.

    Returns
    -------
    pandas.DataFrame
        Frame with canonical ``Actor1CountryCode`` and ``Actor2CountryCode``.
    """
    rename: dict[str, str] = {}
    for column in frame.columns:
        collapsed = column.lower().replace(" ", "").replace("_", "")
        if collapsed == "actor1countrycode":
            rename[column] = "Actor1CountryCode"
        elif collapsed == "actor2countrycode":
            rename[column] = "Actor2CountryCode"
    return frame.rename(columns=rename)


def _classify_event_types(frame: pd.DataFrame) -> pd.Series:
    """Derive the coarse event taxonomy used by the graph layer.

    Precedence is deliberate: material cooperation and aid codes first, then
    violent codes, then the CAMEO quad class. Rows matching nothing are
    treated as diplomatic.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame carrying a normalised ``EventRootCode`` and integer
        ``QuadClass``.

    Returns
    -------
    pandas.Series
        Event-type label per row, aligned to ``frame.index``.
    """
    root = frame["EventRootCode"].astype(str).str[:2]
    quad = frame["QuadClass"]

    conditions = [
        root.isin(_TRADE_AID_CODES),
        root.isin(_MILITARY_CODES),
        quad.isin((1, 2)),
        quad.isin((3, 4)),
    ]
    choices = ["Trade/Aid", "Military/Conflict", "Cooperation", "Conflict"]

    return pd.Series(
        np.select(conditions, choices, default="Diplomatic"),
        index=frame.index,
        dtype="object",
    )


def _pad_event_root_code(value: object) -> str:
    """Normalise a CAMEO root code to a two-character string.

    Parameters
    ----------
    value : object
        Raw root code from GDELT or the synthetic generator.

    Returns
    -------
    str
        Zero-padded two-digit code for numeric input, otherwise the first two
        characters of the stripped input.
    """
    text = str(value).strip()
    return text.zfill(2) if text.isdigit() else text[:2]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise raw events for graph construction.

    Applies actor-column normalisation, bilateral filtering, numeric coercion,
    tone normalisation to ``[-1, 1]`` and event-type classification.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw GDELT or synthetic events.

    Returns
    -------
    pandas.DataFrame
        Cleaned events with the added columns ``tone_norm``, ``event_label``
        and ``event_type``. Empty when no rows survive filtering.
    """
    frame = _normalise_actor_columns(df.copy())

    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    frame["Actor1CountryCode"] = (
        frame["Actor1CountryCode"].fillna("").astype(str).str.strip()
    )
    frame["Actor2CountryCode"] = (
        frame["Actor2CountryCode"].fillna("").astype(str).str.strip()
    )

    bilateral_mask = (
        (frame["Actor1CountryCode"].str.len() == _ISO3_LENGTH)
        & (frame["Actor2CountryCode"].str.len() == _ISO3_LENGTH)
        & (frame["Actor1CountryCode"] != frame["Actor2CountryCode"])
    )
    frame = frame[bilateral_mask]

    if frame.empty:
        _log.warning("%s No valid bilateral events survived filtering", WARN)
        return frame.reset_index(drop=True)

    frame = frame.copy()

    frame["AvgTone"] = pd.to_numeric(frame["AvgTone"], errors="coerce").fillna(0.0)
    frame["QuadClass"] = (
        pd.to_numeric(frame["QuadClass"], errors="coerce").fillna(1).astype(int)
    )
    frame["NumMentions"] = pd.to_numeric(frame["NumMentions"], errors="coerce").fillna(1)
    frame["GoldsteinScale"] = pd.to_numeric(
        frame["GoldsteinScale"], errors="coerce"
    ).fillna(0.0)

    frame["tone_norm"] = frame["AvgTone"].clip(-_TONE_CLIP, _TONE_CLIP) / _TONE_CLIP

    frame["EventRootCode"] = frame["EventRootCode"].fillna("01").map(_pad_event_root_code)
    frame["event_label"] = frame["EventRootCode"].map(EVENT_ROOT_MAP).fillna("Unknown")
    frame["event_type"] = _classify_event_types(frame)

    _log.info(
        "%s Preprocessed %s events | %d source countries | %d target countries",
        OK,
        f"{len(frame):,}",
        frame["Actor1CountryCode"].nunique(),
        frame["Actor2CountryCode"].nunique(),
    )
    return frame.reset_index(drop=True)
