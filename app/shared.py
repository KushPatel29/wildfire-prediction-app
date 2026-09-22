"""
Loaders, colour scales and render helpers shared by every page.

Pages import this as `shared`: Streamlit puts the entry script's folder (app/) on
the import path, and the tests do the same.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

PUBLISHED = ROOT / "data" / "published"
BUNDLED_LIVE = ROOT / "data" / "live"
BUILT_LIVE = Path(tempfile.gettempdir()) / "canada-wildfire-live"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPO_URL = "https://github.com/KushPatel29/wildfire-prediction-app"
RELEASE_URL = f"{REPO_URL}/releases/download/live-forecast"
STALE_HOURS = 30
SEASON_YEAR = 2026
# Read from the grid, not typed: it was 771 until the fire-to-cell fix of 22 September
# 2026 moved every fire into its own cell and the grid became 753.
CELLS_IN_GRID = len(pd.read_parquet(MODELS / "cells.parquet", columns=["cell_id"]))
TOP_CELLS = math.ceil(0.10 * CELLS_IN_GRID)      # the day's riskiest tenth, as the evidence counts it

TEXT = "#E9E6E1"
MUTED = "#9AA3AD"
GRID = "rgba(233,230,225,0.09)"
SURFACE = "#1A1F26"
MODEL = "#F28C38"
LARGE_MODEL = "#E0645A"
CLIMATOLOGY = "#6FA8DC"
FWI_ONLY = "#C9CDD2"     # lighter than climatology's blue by 58%, so the pair reads without its legend
OUTSIDE = "#4A525C"
CAUSE_COLOURS = {"Lightning": "#F2B138", "Human": "#6FA8DC", "Unknown": "#6B737D"}
PROVINCE_SHORT = {
    "British Columbia": "BC", "Alberta": "AB", "Saskatchewan": "SK", "Manitoba": "MB", "Ontario": "ON",
    "Quebec": "QC", "New Brunswick": "NB", "Nova Scotia": "NS", "Prince Edward Island": "PE",
    "Newfoundland and Labrador": "NL", "Yukon": "YT", "Northwest Territories": "NT", "Nunavut": "NU",
    "Parks Canada": "Parks Canada", "Other": "Other",
}

# Five bands coloured like Canada's fire danger classes - blue, green, yellow, orange,
# red - so the map reads at a glance. The cut points are this model's, not the
# official danger rating's.
BAND_RGB = [(69, 117, 180), (102, 170, 92), (238, 196, 64), (240, 128, 40), (214, 47, 39)]
BAND_ALPHA = [72, 132, 180, 212, 240]
NO_DATA_RGBA = [120, 120, 120, 30]


def pct(value, digits: int | None = None) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    if digits is None:
        return f"{value * 100:.3g}%"
    return f"{value * 100:.{digits}f}%"


@dataclass(frozen=True)
class Scale:
    key: str
    title: str
    edges: tuple[float, ...]            # lower edge of each band
    kind: str                           # "pct", "ratio" or "index"
    help: str
    evidence_column: str | None = None  # replay column the bands are checked against

    def band(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        index = np.searchsorted(np.asarray(self.edges[1:]), values, side="right")
        return np.where(np.isnan(values), -1, index)

    def format(self, value: float) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "no data"
        if self.kind == "pct":
            return pct(value)
        if self.kind == "ratio":
            return f"{value:g}×"
        return f"{value:g}"

    def labels(self) -> list[str]:
        last = len(self.edges) - 1
        return [f"under {self.format(self.edges[1])}" if i == 0
                else f"{self.format(low)}+" if i == last
                else f"{self.format(low)}–{self.format(self.edges[i + 1])}"
                for i, low in enumerate(self.edges)]


SCALES = {
    "risk": Scale("risk", "New fire", (0, 0.01, 0.03, 0.07, 0.15), "pct",
                  "Calibrated probability that at least one new fire is reported in the cell that day.", "fires"),
    "large_risk": Scale("large_risk", "Fire over 200 ha", (0, 0.001, 0.005, 0.01, 0.025), "pct",
                        "Probability that a fire reported in the cell that day grows past 200 hectares.", "large_fires"),
    "vs_normal": Scale("vs_normal", "Against normal", (0, 0.5, 1, 2, 4), "ratio",
                       "The day's risk divided by the cell's own fire rate for that month in 2000–2016."),
    "fwi": Scale("fwi", "Fire Weather Index", (0, 5, 10, 20, 30), "index",
                 "The FWI System's fire intensity index, interpolated from the nearest stations.", "fires"),
}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_cells() -> pd.DataFrame:
    return pd.read_parquet(MODELS / "cells.parquet")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_hackathon() -> dict | None:
    """The 2024 model rebuilt and re-scored, if that pipeline has been run."""
    path = REPORTS / "hackathon_2024.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_published(name: str) -> pd.DataFrame:
    frame = pd.read_parquet(PUBLISHED / f"{name}.parquet")
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data(show_spinner=False)
def replay_day(day: pd.Timestamp) -> pd.DataFrame:
    return pd.read_parquet(PUBLISHED / "backtest_daily.parquet", filters=[("date", "==", pd.Timestamp(day))])


@st.cache_data(show_spinner=False)
def season_report(year: int = SEASON_YEAR) -> dict | None:
    path = PUBLISHED / f"season_{year}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def band_table(scale_key: str) -> pd.DataFrame | None:
    """How each band of a scale behaved in the 2020–2024 replay."""
    scale = SCALES[scale_key]
    if scale.evidence_column is None:
        return None
    frame = pd.read_parquet(PUBLISHED / "backtest_daily.parquet", columns=[scale.key, scale.evidence_column])
    band = scale.band(frame[scale.key])
    counts = frame[scale.evidence_column]
    total = max(float(counts.sum()), 1.0)
    rows = []
    for i in range(len(scale.edges)):
        part = counts[band == i]
        rows.append({"band": i, "cell_day_share": len(part) / len(frame),
                     "fire_day_rate": float((part > 0).mean()) if len(part) else float("nan"),
                     "fire_share": float(part.sum()) / total})
    return pd.DataFrame(rows)


@dataclass
class Live:
    forecast: pd.DataFrame
    hotspots: pd.DataFrame
    meta: dict
    origin: str


ORIGINS = {"scheduled": "the twice-daily GitHub Actions run", "built": "a rebuild on this server",
           "bundled": "the snapshot shipped with the app"}


@st.cache_data(ttl=1800, show_spinner=False)
def _remote_live() -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    if os.environ.get("WILDFIRE_OFFLINE"):
        return None
    try:
        meta = requests.get(f"{RELEASE_URL}/meta.json", timeout=15)
        if meta.status_code != 200:
            return None
        frames = []
        for name in ("forecast.parquet", "hotspots.parquet"):
            response = requests.get(f"{RELEASE_URL}/{name}", timeout=60)
            response.raise_for_status()
            frames.append(pd.read_parquet(io.BytesIO(response.content)))
        return frames[0], frames[1], meta.json()
    except (requests.RequestException, ValueError, OSError):
        return None


@st.cache_data(show_spinner=False)
def _local_live(folder: str, stamp: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = Path(folder)
    return (pd.read_parquet(path / "forecast.parquet"), pd.read_parquet(path / "hotspots.parquet"),
            json.loads((path / "meta.json").read_text(encoding="utf-8")))


def live_forecast() -> Live | None:
    """The newest forecast among the scheduled release, a rebuild on this server and
    the bundled snapshot."""
    options = []
    remote = _remote_live()
    if remote is not None:
        options.append(Live(*remote, origin="scheduled"))
    for folder, origin in ((BUILT_LIVE, "built"), (BUNDLED_LIVE, "bundled")):
        meta = folder / "meta.json"
        if meta.exists():
            try:
                options.append(Live(*_local_live(str(folder), meta.stat().st_mtime), origin=origin))
            except (OSError, ValueError):
                continue
    return max(options, key=lambda live: live.meta.get("generated_at", "")) if options else None


_BUILD_LOCK = threading.Lock()


def rebuild_live(progress) -> dict:
    from wildfire.forecast import build

    if not _BUILD_LOCK.acquire(blocking=False):
        raise RuntimeError("Another visitor started a rebuild moments ago. It takes about three minutes; "
                           "reload the page then.")
    try:
        return build(out=BUILT_LIVE, progress=progress)
    finally:
        _BUILD_LOCK.release()


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def hours_since(stamp: str) -> float:
    then = datetime.fromisoformat(stamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 3600


def age(stamp: str) -> str:
    hours = hours_since(stamp)
    if hours < 1:
        return f"{max(1, round(hours * 60))} minutes ago"
    if hours < 48:
        return f"{hours:.0f} hours ago"
    return f"{hours / 24:.0f} days ago"


def day_label(day, lead: int | None = None) -> str:
    ts = pd.Timestamp(day)
    text = f"{ts:%a %b} {ts.day}"
    if lead is None:
        return text
    return f"{text} · observed" if lead == 0 else f"{text} · +{lead} d"


def number(value: float, digits: int = 0) -> str:
    return "–" if value is None or not np.isfinite(value) else f"{value:,.{digits}f}"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def page_style() -> None:
    st.markdown(
        """<style>
        [data-testid="stAppDeployButton"] {display: none}
        .legend {display: flex; flex-wrap: wrap; gap: .35rem 1.15rem; margin: .35rem 0 .2rem; font-size: .86rem}
        .legend .item {display: inline-flex; align-items: center; gap: .4rem; white-space: nowrap}
        .legend .swatch {width: 14px; height: 14px; border-radius: 3px; display: inline-block}
        .legend .note {color: #9AA3AD}
        .kicker {color: #9AA3AD; font-size: .95rem; margin-top: -.5rem; max-width: 62rem}
        </style>""",
        unsafe_allow_html=True,
    )


def kicker(text: str) -> None:
    st.markdown(f"<p class='kicker'>{text}</p>", unsafe_allow_html=True)


def legend(scale: Scale, notes: list[str] | None = None) -> None:
    items = []
    for i, label in enumerate(scale.labels()):
        r, g, b = BAND_RGB[i]
        note = f"<span class='note'>{notes[i]}</span>" if notes else ""
        items.append(f"<span class='item'><span class='swatch' style='background:rgb({r},{g},{b})'></span>"
                     f"{label}{note}</span>")
    st.markdown(f"<div class='legend'>{''.join(items)}</div>", unsafe_allow_html=True)


def band_notes(scale: Scale) -> tuple[list[str] | None, str]:
    table = band_table(scale.key)
    if table is None:
        return None, scale.help
    what = "a fire over 200 ha" if scale.key == "large_risk" else "at least one new fire"
    notes = [f" · {pct(rate, 1)}" for rate in table["fire_day_rate"]]
    return notes, (f"{scale.help} Beside each band: the share of cell-days in that band that saw {what} "
                   "in the 2020–2024 replay, which no part of the model was trained on.")


def _json_ready(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=[c for c in frame if pd.api.types.is_datetime64_any_dtype(frame[c])])
    for column in out:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].astype("float64")
        elif pd.api.types.is_integer_dtype(out[column]):
            out[column] = out[column].astype("int64")
        elif pd.api.types.is_string_dtype(out[column]) and column != "polygon":
            out[column] = out[column].astype(object)
    return out


def cell_polygon(lat: float, lon: float, half: float = 0.5) -> list[list[float]]:
    return [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half]]


def draw_map(cells: pd.DataFrame, scale: Scale, tooltip: str, points: pd.DataFrame | None = None,
             rings: pd.DataFrame | None = None, ring_column: str = "fires", height: int = 540) -> None:
    """Grid cells coloured by band, with optional hotspot dots and rings where fires started."""
    import pydeck as pdk

    data = cells.copy()
    band = scale.band(data[scale.key])
    data["fill"] = [NO_DATA_RGBA if b < 0 else [*BAND_RGB[b], BAND_ALPHA[b]] for b in band]
    data["polygon"] = [cell_polygon(lat, lon) for lat, lon in zip(data["lat"], data["lon"])]
    layers = [pdk.Layer(
        "PolygonLayer", data=_json_ready(data), get_polygon="polygon", get_fill_color="fill",
        get_line_color=[16, 19, 23, 150], line_width_min_pixels=0.5, stroked=True, filled=True,
        pickable=True, auto_highlight=True)]
    if rings is not None and len(rings):
        ringed = rings[["lat", "lon", ring_column]].copy()
        ringed["radius"] = 14000 + 9000 * np.sqrt(ringed[ring_column].astype(float))
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=_json_ready(ringed), get_position=["lon", "lat"], get_radius="radius",
            stroked=True, filled=False, get_line_color=[240, 238, 232, 240], line_width_min_pixels=1.5))
    if points is not None and len(points):
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=_json_ready(points[["lat", "lon"]]), get_position=["lon", "lat"],
            get_radius=2500, radius_min_pixels=1.5, radius_max_pixels=5, get_fill_color=[255, 236, 179, 230]))
    deck = pdk.Deck(
        layers=layers, initial_view_state=pdk.ViewState(latitude=59.5, longitude=-97.0, zoom=2.9),
        map_provider="carto", map_style="dark",
        tooltip={"html": tooltip, "style": {"backgroundColor": SURFACE, "color": TEXT, "fontSize": "12px"}})
    st.pydeck_chart(deck, height=height)


def show(fig, height: int = 340, **layout) -> None:
    # t=66 leaves the title its own line above the legend; automargin keeps tick
    # labels from being clipped by the tight side margins.
    base = dict(height=height, margin=dict(l=8, r=8, t=66, b=8), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=12),
                title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=14)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
                hoverlabel=dict(bgcolor=SURFACE, font=dict(color=TEXT)))
    base.update(layout)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def day_marker(fig, day) -> None:
    """A vertical line at `day`. Not add_vline: with an annotation, plotly averages
    the x values, which pandas 3 refuses to do for Timestamps."""
    fig.add_shape(type="line", x0=day, x1=day, y0=0, y1=1, yref="paper",
                  line=dict(color=TEXT, width=1, dash="dot"))


def sources_footer() -> None:
    st.divider()
    st.caption(
        "Fire records: Canadian National Fire Database, Natural Resources Canada. Station weather, FWI System "
        "codes and satellite hotspots: Canadian Wildland Fire Information System (CWFIS). Contains information "
        "licensed under the Open Government Licence – Canada. Forecast weather by "
        "[Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). This is an independent model, not an official "
        "fire danger rating: for decisions, follow [CWFIS](https://cwfis.cfs.nrcan.gc.ca/) and your provincial "
        f"or territorial wildfire service. [Source code]({REPO_URL})."
    )
