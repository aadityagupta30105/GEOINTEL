"""
Country Reference Geography
===========================
Static reference data backing the map projections: representative centroid
coordinates, display names and GDP measured at purchasing power parity.

Units
-----
``gdp_ppp_billions`` is denominated in **billions of current international
dollars** (IMF, 2023 estimates). Presentation helpers convert to trillions;
callers must not format the raw value with a trillion suffix.
"""

from __future__ import annotations

from typing import Final, NamedTuple

__all__ = ["CountryGeo", "COUNTRY_GEO", "MAX_GDP_PPP_BILLIONS", "format_gdp", "marker_size"]


class CountryGeo(NamedTuple):
    """Reference geography and economic scale for one country.

    Attributes
    ----------
    lat, lon : float
        Representative centroid in decimal degrees.
    name : str
        Display name.
    gdp_ppp_billions : float
        GDP at purchasing power parity, in billions of international dollars.
    """

    lat: float
    lon: float
    name: str
    gdp_ppp_billions: float


COUNTRY_GEO: Final[dict[str, CountryGeo]] = {
    # --- Major powers -------------------------------------------------------
    "USA": CountryGeo(38.0, -97.0, "United States", 26900),
    "CHN": CountryGeo(35.0, 103.0, "China", 33000),
    "RUS": CountryGeo(62.0, 90.0, "Russia", 5300),
    "DEU": CountryGeo(51.0, 10.0, "Germany", 5500),
    "GBR": CountryGeo(54.0, -2.0, "United Kingdom", 3700),
    "FRA": CountryGeo(46.0, 2.0, "France", 3900),
    "IND": CountryGeo(21.0, 78.0, "India", 13100),
    "BRA": CountryGeo(-10.0, -55.0, "Brazil", 3900),
    "JPN": CountryGeo(36.0, 138.0, "Japan", 6500),
    "KOR": CountryGeo(36.0, 128.0, "South Korea", 2700),
    "ISR": CountryGeo(31.5, 35.0, "Israel", 520),
    "IRN": CountryGeo(32.0, 53.0, "Iran", 1600),
    "SAU": CountryGeo(24.0, 45.0, "Saudi Arabia", 2000),
    "TUR": CountryGeo(39.0, 35.0, "Turkey", 3600),
    "PAK": CountryGeo(30.0, 70.0, "Pakistan", 1500),
    "NGA": CountryGeo(9.0, 8.0, "Nigeria", 1300),
    "ZAF": CountryGeo(-29.0, 25.0, "South Africa", 900),
    "EGY": CountryGeo(27.0, 30.0, "Egypt", 1900),
    "MEX": CountryGeo(24.0, -102.0, "Mexico", 3100),
    "ARG": CountryGeo(-34.0, -64.0, "Argentina", 1200),
    "IDN": CountryGeo(-2.0, 118.0, "Indonesia", 4400),
    "AUS": CountryGeo(-25.0, 134.0, "Australia", 1700),
    "CAN": CountryGeo(56.0, -96.0, "Canada", 2400),
    "ITA": CountryGeo(42.5, 12.5, "Italy", 3200),
    "UKR": CountryGeo(49.0, 31.0, "Ukraine", 600),
    "POL": CountryGeo(52.0, 20.0, "Poland", 1700),
    "NLD": CountryGeo(52.3, 5.3, "Netherlands", 1200),
    "SWE": CountryGeo(60.0, 15.0, "Sweden", 700),
    "NOR": CountryGeo(64.0, 13.0, "Norway", 600),
    "CHL": CountryGeo(-30.0, -71.0, "Chile", 600),
    # --- Europe -------------------------------------------------------------
    "ESP": CountryGeo(40.0, -4.0, "Spain", 2400),
    "PRT": CountryGeo(39.5, -8.0, "Portugal", 410),
    "BEL": CountryGeo(50.8, 4.4, "Belgium", 760),
    "AUT": CountryGeo(47.5, 14.5, "Austria", 620),
    "CHE": CountryGeo(47.0, 8.0, "Switzerland", 720),
    "DNK": CountryGeo(56.0, 10.0, "Denmark", 440),
    "FIN": CountryGeo(64.0, 26.0, "Finland", 360),
    "GRC": CountryGeo(39.0, 22.0, "Greece", 380),
    "CZE": CountryGeo(49.8, 15.5, "Czech Republic", 560),
    "HUN": CountryGeo(47.0, 19.0, "Hungary", 420),
    "ROU": CountryGeo(46.0, 25.0, "Romania", 730),
    "BGR": CountryGeo(42.7, 25.5, "Bulgaria", 230),
    "HRV": CountryGeo(45.0, 16.0, "Croatia", 160),
    "SVK": CountryGeo(48.7, 19.5, "Slovakia", 260),
    "SVN": CountryGeo(46.1, 14.8, "Slovenia", 110),
    "SRB": CountryGeo(44.0, 21.0, "Serbia", 190),
    "BLR": CountryGeo(53.5, 28.0, "Belarus", 220),
    "MDA": CountryGeo(47.0, 28.5, "Moldova", 40),
    "ALB": CountryGeo(41.0, 20.0, "Albania", 55),
    "LTU": CountryGeo(55.9, 23.9, "Lithuania", 160),
    "LVA": CountryGeo(57.0, 25.0, "Latvia", 80),
    "EST": CountryGeo(59.0, 25.0, "Estonia", 60),
    "BIH": CountryGeo(44.0, 17.5, "Bosnia and Herzegovina", 75),
    "MKD": CountryGeo(41.6, 21.7, "North Macedonia", 45),
    "MNE": CountryGeo(42.7, 19.4, "Montenegro", 18),
    "IRL": CountryGeo(53.0, -8.0, "Ireland", 620),
    "LUX": CountryGeo(49.8, 6.1, "Luxembourg", 105),
    "XKX": CountryGeo(42.6, 20.9, "Kosovo", 25),
    # --- Middle East and Central Asia --------------------------------------
    "IRQ": CountryGeo(33.0, 44.0, "Iraq", 520),
    "SYR": CountryGeo(35.0, 38.0, "Syria", 60),
    "JOR": CountryGeo(31.0, 36.0, "Jordan", 120),
    "LBN": CountryGeo(33.9, 35.5, "Lebanon", 70),
    "YEM": CountryGeo(15.5, 47.5, "Yemen", 55),
    "OMN": CountryGeo(22.0, 58.0, "Oman", 220),
    "ARE": CountryGeo(24.0, 54.0, "United Arab Emirates", 760),
    "QAT": CountryGeo(25.3, 51.2, "Qatar", 280),
    "KWT": CountryGeo(29.5, 47.8, "Kuwait", 240),
    "BHR": CountryGeo(26.0, 50.5, "Bahrain", 92),
    "AFG": CountryGeo(33.0, 65.0, "Afghanistan", 75),
    "KAZ": CountryGeo(48.0, 68.0, "Kazakhstan", 680),
    "UZB": CountryGeo(41.0, 64.0, "Uzbekistan", 280),
    "TKM": CountryGeo(39.0, 59.0, "Turkmenistan", 110),
    "TJK": CountryGeo(39.0, 71.0, "Tajikistan", 45),
    "KGZ": CountryGeo(41.0, 75.0, "Kyrgyzstan", 38),
    "AZE": CountryGeo(40.5, 47.5, "Azerbaijan", 200),
    "ARM": CountryGeo(40.0, 45.0, "Armenia", 65),
    "GEO": CountryGeo(42.0, 43.5, "Georgia", 80),
    # --- Asia-Pacific -------------------------------------------------------
    "VNM": CountryGeo(16.0, 108.0, "Vietnam", 1300),
    "THA": CountryGeo(15.0, 101.0, "Thailand", 1500),
    "MYS": CountryGeo(4.0, 109.0, "Malaysia", 1200),
    "PHL": CountryGeo(13.0, 122.0, "Philippines", 1100),
    "SGP": CountryGeo(1.35, 103.8, "Singapore", 720),
    "BGD": CountryGeo(24.0, 90.0, "Bangladesh", 1200),
    "LKA": CountryGeo(7.5, 80.7, "Sri Lanka", 280),
    "NPL": CountryGeo(28.0, 84.0, "Nepal", 140),
    "MMR": CountryGeo(17.0, 96.0, "Myanmar", 280),
    "KHM": CountryGeo(12.5, 105.0, "Cambodia", 95),
    "LAO": CountryGeo(18.0, 103.0, "Laos", 70),
    "MNG": CountryGeo(46.0, 105.0, "Mongolia", 55),
    "NZL": CountryGeo(-41.0, 174.0, "New Zealand", 260),
    "PNG": CountryGeo(-6.0, 147.0, "Papua New Guinea", 45),
    "TWN": CountryGeo(23.7, 121.0, "Taiwan", 1700),
    "PRK": CountryGeo(40.0, 127.0, "North Korea", 40),
    # --- Africa -------------------------------------------------------------
    "ETH": CountryGeo(9.0, 40.0, "Ethiopia", 380),
    "ERI": CountryGeo(15.2, 39.0, "Eritrea", 12),
    "TZA": CountryGeo(-6.0, 35.0, "Tanzania", 220),
    "KEN": CountryGeo(1.0, 38.0, "Kenya", 320),
    "GHA": CountryGeo(8.0, -2.0, "Ghana", 200),
    "CIV": CountryGeo(7.5, -5.5, "Cote d Ivoire", 180),
    "AGO": CountryGeo(-12.0, 18.5, "Angola", 280),
    "CMR": CountryGeo(6.0, 12.0, "Cameroon", 130),
    "MOZ": CountryGeo(-18.0, 35.0, "Mozambique", 55),
    "MDG": CountryGeo(-20.0, 47.0, "Madagascar", 50),
    "ZMB": CountryGeo(-15.0, 28.0, "Zambia", 75),
    "ZWE": CountryGeo(-20.0, 30.0, "Zimbabwe", 45),
    "SEN": CountryGeo(14.5, -14.0, "Senegal", 85),
    "MWI": CountryGeo(-13.5, 34.0, "Malawi", 35),
    "MLI": CountryGeo(17.0, -4.0, "Mali", 60),
    "BFA": CountryGeo(12.0, -2.0, "Burkina Faso", 55),
    "NER": CountryGeo(17.0, 8.0, "Niger", 35),
    "TCD": CountryGeo(15.0, 19.0, "Chad", 30),
    "SDN": CountryGeo(16.0, 30.0, "Sudan", 220),
    "SSD": CountryGeo(7.0, 30.0, "South Sudan", 25),
    "LBY": CountryGeo(27.0, 17.0, "Libya", 95),
    "TUN": CountryGeo(34.0, 9.0, "Tunisia", 170),
    "DZA": CountryGeo(28.0, 3.0, "Algeria", 650),
    "MAR": CountryGeo(32.0, -5.0, "Morocco", 400),
    "COD": CountryGeo(-2.0, 23.5, "DR Congo", 120),
    "UGA": CountryGeo(1.0, 32.0, "Uganda", 150),
    "RWA": CountryGeo(-2.0, 30.0, "Rwanda", 40),
    "SOM": CountryGeo(6.0, 46.0, "Somalia", 15),
    "DJI": CountryGeo(11.8, 42.5, "Djibouti", 5),
    "GAB": CountryGeo(-1.0, 11.7, "Gabon", 45),
    "COG": CountryGeo(-1.0, 15.0, "Republic of the Congo", 25),
    "TGO": CountryGeo(8.0, 1.0, "Togo", 22),
    "BEN": CountryGeo(9.3, 2.3, "Benin", 55),
    "LBR": CountryGeo(6.5, -9.5, "Liberia", 8),
    "SLE": CountryGeo(8.5, -11.8, "Sierra Leone", 14),
    "GIN": CountryGeo(11.0, -10.7, "Guinea", 32),
    "MRT": CountryGeo(21.0, -11.0, "Mauritania", 28),
    "NAM": CountryGeo(-22.0, 17.0, "Namibia", 30),
    "BWA": CountryGeo(-22.0, 24.0, "Botswana", 55),
    # --- Americas -----------------------------------------------------------
    "COL": CountryGeo(4.0, -73.0, "Colombia", 950),
    "VEN": CountryGeo(8.0, -66.0, "Venezuela", 220),
    "PER": CountryGeo(-10.0, -76.0, "Peru", 550),
    "ECU": CountryGeo(-2.0, -77.5, "Ecuador", 250),
    "BOL": CountryGeo(-17.0, -65.0, "Bolivia", 130),
    "PRY": CountryGeo(-23.0, -58.0, "Paraguay", 110),
    "URY": CountryGeo(-33.0, -56.0, "Uruguay", 120),
    "GTM": CountryGeo(15.5, -90.0, "Guatemala", 180),
    "HND": CountryGeo(15.0, -86.5, "Honduras", 80),
    "SLV": CountryGeo(13.7, -88.9, "El Salvador", 70),
    "NIC": CountryGeo(13.0, -85.0, "Nicaragua", 45),
    "CRI": CountryGeo(10.0, -84.0, "Costa Rica", 130),
    "PAN": CountryGeo(9.0, -80.0, "Panama", 170),
    "CUB": CountryGeo(22.0, -80.0, "Cuba", 140),
    "DOM": CountryGeo(19.0, -70.7, "Dominican Republic", 270),
    "HTI": CountryGeo(19.0, -72.3, "Haiti", 35),
    "JAM": CountryGeo(18.2, -77.4, "Jamaica", 30),
    "TTO": CountryGeo(10.7, -61.4, "Trinidad and Tobago", 50),
    "GUY": CountryGeo(5.0, -59.0, "Guyana", 40),
    "SUR": CountryGeo(4.0, -56.0, "Suriname", 12),
    # --- Oceania ------------------------------------------------------------
    "FJI": CountryGeo(-18.0, 178.0, "Fiji", 11),
}

MAX_GDP_PPP_BILLIONS: Final[float] = max(
    entry.gdp_ppp_billions for entry in COUNTRY_GEO.values()
)


def format_gdp(gdp_billions: float) -> str:
    """Render a GDP figure for display.

    Parameters
    ----------
    gdp_billions : float
        GDP at PPP in billions of international dollars.

    Returns
    -------
    str
        Trillions for values at or above one trillion, otherwise billions.
    """
    if gdp_billions >= 1000:
        return f"${gdp_billions / 1000:.1f}T"
    return f"${gdp_billions:.0f}B"


def marker_size(
    gdp_billions: float,
    base: float = 8.0,
    span: float = 52.0,
) -> float:
    """Scale a map marker by economic size.

    A square-root transform is applied so that small economies remain visible
    alongside the largest.

    Parameters
    ----------
    gdp_billions : float
        GDP at PPP in billions of international dollars.
    base : float, optional
        Minimum marker diameter in pixels.
    span : float, optional
        Additional diameter available to the largest economy.

    Returns
    -------
    float
        Marker diameter in pixels.
    """
    ratio = max(0.0, gdp_billions) / MAX_GDP_PPP_BILLIONS
    return base + (ratio ** 0.5) * span
