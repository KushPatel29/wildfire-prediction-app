"""
Hold the generated Power BI project to the data it actually reads.

Every failure mode here is one Power BI does not report. A measure naming a
column that no longer exists gets `state: SemanticError`, every measure that
calls it inherits `DependencyError`, and the bound visual renders "Something's
wrong with one or more fields" - at runtime, on a project that opened cleanly. A
relationship whose keys do not match loads, filters nothing, and quietly adds a
blank member to the dimension. A column left out of the M query's type list
arrives as text, still binds, still draws, and sorts alphabetically.

None of that shows in a diff, so it is asserted here. Three of these tests were
written because the failure had already happened: the apostrophe in a measure
name that stopped the project opening at all, the month abbreviation typed as a
date that made every row of `dim_date` an error, and the hotspot timestamp that
typed as a datetime and matched no day in the calendar.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from powerbi import build_pbip
from powerbi.model_spec import MEASURES, RELATIONSHIPS, SORT_BY, TABLES
from powerbi.report_spec import PAGES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "powerbi" / "data"
PBIP = ROOT / "powerbi" / "pbip"
MODEL = PBIP / "CanadaWildfireRisk.SemanticModel" / "definition"
REPORT = PBIP / "CanadaWildfireRisk.Report" / "definition"

MEASURE_NAMES = {name for name, *_ in MEASURES}
COLUMN_REF = re.compile(r"(?:'([^']+)'|\b([A-Za-z_]\w*))\[([^\]]+)\]")
MEASURE_REF = re.compile(r"(?<![\w'\]])\[([^\]\[]+)\]")


@pytest.fixture(scope="module")
def headers() -> dict[str, list[str]]:
    """The columns each CSV actually has."""
    return {name: list(pd.read_csv(DATA / f"{name}.csv", nrows=1).columns) for name in TABLES}


@pytest.fixture(scope="module")
def visuals() -> list[dict]:
    """Every drawn visual. A visual *group* - the filter panel's container - has
    no `visual` body of its own; it is a box the panel's slicers sit in."""
    documents = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(REPORT.glob("pages/*/visuals/*/visual.json"))]
    return [document for document in documents if "visual" in document]


@pytest.fixture(scope="module")
def groups() -> list[dict]:
    documents = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(REPORT.glob("pages/*/visuals/*/visual.json"))]
    return [document for document in documents if "visualGroup" in document]


# --- the model against its data -------------------------------------------

def test_every_table_reads_a_csv_that_exists(headers):
    for name in TABLES:
        assert (DATA / f"{name}.csv").is_file(), name
        assert headers[name], name


def test_every_bound_column_is_a_column_the_csv_has(headers):
    """`sourceColumn` is what the model binds; the CSV header is what arrives."""
    missing = []
    for name in TABLES:
        tmdl = (MODEL / "tables" / f"{name}.tmdl").read_text(encoding="utf-8")
        for column in re.findall(r"^\t\tsourceColumn: (.+)$", tmdl, re.M):
            if column.strip() not in headers[name]:
                missing.append(f"{name}[{column.strip()}]")
    assert not missing, missing


def test_every_bound_column_is_also_typed(headers):
    """A column bound but left out of TransformColumnTypes arrives as text."""
    for name in TABLES:
        tmdl = (MODEL / "tables" / f"{name}.tmdl").read_text(encoding="utf-8")
        bound = {c.strip() for c in re.findall(r"^\t\tsourceColumn: (.+)$", tmdl, re.M)}
        typed = set(re.findall(r'\{"([^"]+)",\s*(?:Int64\.Type|type [\w.]+)\}', tmdl))
        assert bound <= typed, (name, sorted(bound - typed))


def test_every_measure_names_columns_that_exist(headers):
    dangling = []
    for name, dax, *_ in MEASURES:
        for quoted, bare, column in COLUMN_REF.findall(dax):
            table = quoted or bare
            if table in headers and column not in headers[table]:
                dangling.append(f"{name}: {table}[{column}]")
    assert not dangling, dangling


def test_every_measure_that_calls_another_calls_one_that_exists():
    dangling = []
    for name, dax, *_ in MEASURES:
        for called in MEASURE_REF.findall(dax):
            if called not in MEASURE_NAMES and not called.startswith("@"):
                dangling.append(f"{name} -> [{called}]")
    assert not dangling, dangling


def test_no_name_carries_an_apostrophe():
    """TMDL wraps a name in single quotes, so one inside a name ends it early.

    Power BI reports that as "Invalid indentation was detected" on a line three
    below the one at fault, and refuses to open the project at all."""
    build_pbip.check_names([name for name, *_ in MEASURES] + list(TABLES))


# --- relationships --------------------------------------------------------

def test_every_relationship_joins_columns_that_exist(headers):
    for from_table, from_column, to_table, to_column in RELATIONSHIPS:
        assert from_column in headers[from_table], (from_table, from_column)
        assert to_column in headers[to_table], (to_table, to_column)


def test_every_relationship_key_matches_on_both_sides():
    """The failure this exists for: a join that loads, filters nothing, and adds
    a blank member to the dimension rather than an error anywhere."""
    orphans = []
    for from_table, from_column, to_table, to_column in RELATIONSHIPS:
        many = pd.read_csv(DATA / f"{from_table}.csv", usecols=[from_column])[from_column]
        one = pd.read_csv(DATA / f"{to_table}.csv", usecols=[to_column])[to_column]
        missing = set(many.dropna().astype(str)) - set(one.dropna().astype(str))
        if missing:
            orphans.append(f"{from_table}[{from_column}] -> {to_table}[{to_column}]: "
                           f"{len(missing)} unmatched, e.g. {sorted(missing)[:3]}")
    assert not orphans, orphans


def test_the_one_side_of_every_relationship_is_unique():
    """Power BI rejects a many-to-many it was not asked for, and the report then
    opens with the relationship missing rather than wrong."""
    duplicated = []
    for _from_table, _from_column, to_table, to_column in RELATIONSHIPS:
        values = pd.read_csv(DATA / f"{to_table}.csv", usecols=[to_column])[to_column]
        if values.duplicated().any():
            duplicated.append(f"{to_table}[{to_column}]")
    assert not duplicated, duplicated


# --- typing ---------------------------------------------------------------

def test_a_date_column_is_a_date_in_the_csv_as_well_as_the_model(headers):
    """`type date` on "Apr" is an error on every row, and the table loads with a
    column of them. `type date` on a timestamp is the same error."""
    for name in TABLES:
        tmdl = (MODEL / "tables" / f"{name}.tmdl").read_text(encoding="utf-8")
        for column in re.findall(r'\{"([^"]+)",\s*type date\}', tmdl):
            values = pd.read_csv(DATA / f"{name}.csv", usecols=[column])[column]
            assert build_pbip.looks_like_dates(values), f"{name}[{column}] is not a date"
            assert not build_pbip.has_time(values), f"{name}[{column}] carries a time"


def test_every_sort_column_exists_and_is_numeric(headers):
    """A sort key read as text sorts 1, 10, 11, 2 - and nothing says so."""
    for table, pairs in SORT_BY.items():
        frame = pd.read_csv(DATA / f"{table}.csv")
        for column, sort_column in pairs.items():
            assert column in frame, (table, column)
            assert sort_column in frame, (table, sort_column)
            assert pd.api.types.is_numeric_dtype(frame[sort_column]), (table, sort_column)


# --- the report -----------------------------------------------------------

def test_every_visual_binds_a_field_that_resolves(visuals, headers):
    """A visual bound to a field the model does not have renders empty."""
    ui_measures = {name for name, _dax, _image in
                   build_pbip.ui_measures(PAGES, {name: fmt for name, _d, fmt, _f in MEASURES})}
    known = MEASURE_NAMES | ui_measures
    dangling = []
    for visual in visuals:
        body = json.dumps(visual["visual"].get("query", {}))
        for entity, prop in re.findall(r'"Entity":\s*"([^"]+)"[^}]*}[^}]*},\s*"Property":\s*"([^"]+)"', body):
            if entity in headers and prop not in headers[entity] and prop not in known:
                dangling.append(f"{visual['name']}: {entity}[{prop}]")
    assert not dangling, dangling


def test_every_visual_has_alt_text(visuals):
    """A dashboard nobody can read with a screen reader is a dashboard with a
    quarter of its audience removed."""
    missing = [v["name"] for v in visuals
               if "altText" not in json.dumps(v["visual"].get("visualContainerObjects", {}))]
    assert not missing, missing
    assert len(visuals) > 80, "the glob found almost nothing, so this proves nothing"


def test_nothing_is_drawn_outside_the_canvas(visuals, groups):
    """The canvas is 1280x720 at FitToPage. Power BI renders what hangs over the
    edge and the PDF export cuts it off."""
    outside = []
    for visual in visuals + groups:
        if visual.get("parentGroupName"):
            continue                      # positioned relative to its group
        pos = visual["position"]
        if (pos["x"] < 0 or pos["y"] < 0
                or pos["x"] + pos["width"] > 1280 or pos["y"] + pos["height"] > 720):
            outside.append(visual["name"])
    assert not outside, outside


def test_every_schema_version_is_one_microsoft_published(visuals):
    """An invented version number validates against nothing: the schema sets
    additionalProperties false, which is the one cheap way to catch a mistyped
    property, and a 404 gives that up while looking like you still have it."""
    published = set(build_pbip.SCHEMA.values())
    for visual in visuals:
        assert visual["$schema"] in published, visual["name"]


def test_the_totals_are_off_where_the_values_are_averages(visuals):
    """A total under a column of ROC-AUCs is an average of averages. Power BI
    prints one by default, and it reads as a headline."""
    for visual in visuals:
        title = json.dumps(visual["visual"].get("visualContainerObjects", {}).get("title", []))
        if "scoreboard" in title or "all three rankings" in title or "Both panels" in title:
            objects = json.dumps(visual["visual"].get("objects", {}))
            assert "subTotals" in objects or "total" in objects, visual["name"]


def test_each_headline_names_the_slice_it_shows():
    """A card with no row context averages every method and both targets: the
    capture rate showed 38.6% where the model's own figure was 46.3%."""
    for name, dax, *_ in MEASURES:
        if name.endswith(", this model"):
            assert 'method] = "model"' in dax or 'method] = "model"' in dax.replace(" ", ""), name


def test_the_forecast_verdict_names_every_province_at_the_peak(headers):
    """Calibrated risk comes in plateaus, so the day's peak is usually shared.
    On the 16 September forecast six cells in Alberta, British Columbia and
    Ontario sat at 11.4%; the verdict took one cell with TOPN(1), broke the tie
    by cell id, and said Ontario. It also called 119 cells "the day's" riskiest
    when that was eight days' worth: a day's tenth is 78."""
    dax = {name: expression for name, expression, *_ in MEASURES}["Forecast verdict"]
    assert "TOPN(1, VALUES(fact_forecast[cell_id])" not in dax
    assert "[Peak risk] = vPeak" in dax, "the cells at the peak are all of them"
    assert "CONCATENATEX" in dax and "VALUES(fact_forecast[province])" in dax
    assert "on at least one of these" in dax, "several days are not one day"

    forecast = pd.read_csv(DATA / "fact_forecast.csv", usecols=["date", "risk"])
    assert forecast["risk"].nunique() < len(forecast) / 10, (
        "isotonic calibration should leave plateaus; if it no longer does, "
        "this test's premise has changed")


def test_the_committed_project_matches_the_spec(tmp_path):
    """`python -m powerbi.build_pbip` is the only way the project is written; a
    hand-edit in Desktop would be silently overwritten by the next build."""
    build_pbip.build(tmp_path, ROOT / "powerbi")
    drift = build_pbip.differences(tmp_path, PBIP)
    assert not drift, drift[:20]


# --- the screenshots --------------------------------------------------------

SHOTS = ROOT / "powerbi" / "screenshots"


def test_there_is_one_screenshot_per_page_and_the_readme_shows_them_all():
    """A capture that never moved between pages ships eight copies of page one,
    and every number in them is still correct. So: as many images as pages, all
    different, all referenced."""
    import hashlib

    pages = json.loads((REPORT / "pages" / "pages.json").read_text(encoding="utf-8"))["pageOrder"]
    shots = sorted(SHOTS.glob("*.png"))
    assert len(shots) == len(pages)
    digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in shots}
    assert len(digests) == len(shots), "two screenshots are the same image"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for path in shots:
        assert f"powerbi/screenshots/{path.name}" in readme, path.name


def test_the_screenshots_are_the_canvas_and_nothing_else():
    """16:9, as the canvas is. A crop that caught Desktop's own banner comes out
    shorter than the page, which is how the first export of this report was
    caught."""
    import struct

    for path in sorted(SHOTS.glob("*.png")):
        width, height = struct.unpack(">II", path.read_bytes()[16:24])
        assert abs(width / height - 1280 / 720) < 0.01, (path.name, width, height)
        assert width >= 1600, (path.name, width)
