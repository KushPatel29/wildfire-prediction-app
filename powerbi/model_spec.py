"""
What the semantic model contains: tables, relationships and measures.

Kept apart from the writer in :mod:`powerbi.build_pbip` so the *shape* of the
model can be read and asserted on without wading through TMDL. The tests import
this module directly.

Nothing here recomputes a model score. Every measure either aggregates a column
`pipelines/powerbi_export.py` wrote, or divides two such aggregates. A DAX
expression that re-derived, say, the share of fires inside the day's riskiest
tenth would be a second implementation of a definition that already has tests in
`wildfire.evaluation` - and the two would drift the first time either changed.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Tables. `source` is the directory the CSV lives in, relative to DataPath.
# --------------------------------------------------------------------------

TABLES: dict[str, dict] = {
    # Dimensions
    "dim_cell": {"source": "data", "kind": "dimension"},
    "dim_province": {"source": "data", "kind": "dimension"},
    "dim_date": {"source": "data", "kind": "dimension"},
    "dim_season": {"source": "data", "kind": "dimension"},

    # The live forecast
    "fact_forecast": {"source": "data", "kind": "fact"},
    "fact_forecast_run": {"source": "data", "kind": "fact"},
    "fact_hotspot": {"source": "data", "kind": "fact"},

    # The out-of-time backtest, 2020-2024
    "fact_backtest_day": {"source": "data", "kind": "fact"},
    "fact_backtest_cell": {"source": "data", "kind": "fact"},

    # What the fire record says
    "fact_fires_year": {"source": "data", "kind": "fact"},
    "fact_fires_month": {"source": "data", "kind": "fact"},

    # How the model scored, and where
    "fact_model_scores": {"source": "data", "kind": "analysis"},
    "fact_model_year": {"source": "data", "kind": "analysis"},
    "fact_model_province": {"source": "data", "kind": "analysis"},
    "fact_reliability": {"source": "data", "kind": "analysis"},
    "fact_feature_gain": {"source": "data", "kind": "analysis"},

    # The current season, graded against satellites
    "fact_season_scores": {"source": "data", "kind": "analysis"},
    "fact_season_month": {"source": "data", "kind": "analysis"},
    "fact_season_day": {"source": "data", "kind": "analysis"},
    "fact_season_cell": {"source": "data", "kind": "analysis"},

    # The 2024 hackathon model, rebuilt and re-scored
    "fact_hackathon_scores": {"source": "data", "kind": "analysis"},
    "fact_hackathon_month": {"source": "data", "kind": "analysis"},
}


# --------------------------------------------------------------------------
# Sort orders. A text column sorts alphabetically on every axis until it is
# told otherwise, which puts April after August and "Test" before "Train".
# --------------------------------------------------------------------------

SORT_BY: dict[str, dict[str, str]] = {
    "dim_date": {"month_name": "month_index"},
    "fact_fires_month": {"month_name": "month"},
    "fact_hackathon_month": {"month_name": "month"},
    "fact_season_month": {"month_name": "month"},
    "fact_hackathon_scores": {"scoring_label": "sort_order"},
}


# --------------------------------------------------------------------------
# Relationships, many-to-one, single direction.
#
# `dim_cell` deliberately does NOT relate to `dim_province`: every fact that
# needs a province carries one, and a second path from a fact to the province
# through the cell grid would make the filter ambiguous.
# --------------------------------------------------------------------------

RELATIONSHIPS: list[tuple[str, str, str, str]] = [
    ("fact_forecast", "cell_id", "dim_cell", "cell_id"),
    ("fact_forecast", "date", "dim_date", "date"),
    ("fact_forecast", "province", "dim_province", "province"),
    ("fact_hotspot", "date", "dim_date", "date"),
    ("fact_backtest_day", "date", "dim_date", "date"),
    ("fact_backtest_cell", "cell_id", "dim_cell", "cell_id"),
    ("fact_backtest_cell", "year", "dim_season", "year"),
    ("fact_backtest_cell", "province", "dim_province", "province"),
    ("fact_fires_year", "year", "dim_season", "year"),
    ("fact_fires_year", "province", "dim_province", "province"),
    ("fact_fires_month", "province", "dim_province", "province"),
    ("fact_model_year", "year", "dim_season", "year"),
    ("fact_model_province", "province", "dim_province", "province"),
    ("fact_season_day", "date", "dim_date", "date"),
    ("fact_season_cell", "cell_id", "dim_cell", "cell_id"),
    ("fact_season_cell", "province", "dim_province", "province"),
    ("fact_hackathon_month", "year", "dim_season", "year"),
]


# No what-if parameters and no field parameters. A what-if slider implies the
# reader can change an input and watch the answer move; every number on this
# report is a score that was computed once, out of time, and changing a slider
# would only move which rows are shown.
WHATIF_PARAMETERS: tuple[tuple[str, str, float, float, float, str, str], ...] = ()
FIELD_PARAMETERS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = ()


def field_parameter_columns() -> dict[str, set[str]]:
    return {}


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------

MEASURES: list[tuple[str, str, str, str]] = [
    # --- the live forecast -------------------------------------------------
    ("Cells forecast", "DISTINCTCOUNT(fact_forecast[cell_id])", "#,0", "01 Forecast"),
    ("Mean risk", "AVERAGE(fact_forecast[risk])", "0.0%", "01 Forecast"),
    ("Peak risk", "MAX(fact_forecast[risk])", "0.0%", "01 Forecast"),
    ("Peak large-fire risk", "MAX(fact_forecast[large_risk])", "0.00%", "01 Forecast"),
    ("Cells in the riskiest tenth",
     "CALCULATE(DISTINCTCOUNT(fact_forecast[cell_id]), fact_forecast[in_top_decile] = 1)",
     "#,0", "01 Forecast"),
    ("Risk against normal",
     "DIVIDE(AVERAGE(fact_forecast[risk]), AVERAGE(fact_forecast[clim_month_rate]))",
     "0.00", "01 Forecast"),
    ("Mean FWI", "AVERAGE(fact_forecast[fwi])", "0.0", "01 Forecast"),
    ("Hotspots in the last two days", "COUNTROWS(fact_hotspot)", "#,0", "01 Forecast"),
    ("Stations behind the forecast", "MAX(fact_forecast_run[stations_used])", "#,0", "01 Forecast"),
    ("Forecast built", "MAX(fact_forecast_run[generated_at])", "", "01 Forecast"),
    # Formatted here rather than concatenated raw: a date pasted into a string
    # takes the viewer's locale with it, and the same tile then reads
    # 2026-09-15 on one machine and 15/09/2026 on the next.
    ("Starts from", "FORMAT(MAX(fact_forecast_run[as_of]), \"yyyy-mm-dd\")", "", "01 Forecast"),

    # --- the fire record ---------------------------------------------------
    ("Fires recorded", "SUM(fact_fires_year[fires])", "#,0", "02 Fire record"),
    ("Area burned", "SUM(fact_fires_year[area_ha])", "#,0", "02 Fire record"),
    ("Area burned (million ha)", "DIVIDE(SUM(fact_fires_year[area_ha]), 1000000)",
     "0.0", "02 Fire record"),
    ("Large fires recorded", "SUM(fact_fires_year[large_fires])", "#,0", "02 Fire record"),
    ("Lightning share of fires",
     "DIVIDE(CALCULATE([Fires recorded], fact_fires_year[cause_label] = \"Lightning\"), [Fires recorded])",
     "0.0%", "02 Fire record"),
    ("Lightning share of area",
     "DIVIDE(CALCULATE([Area burned], fact_fires_year[cause_label] = \"Lightning\"), [Area burned])",
     "0.0%", "02 Fire record"),
    ("Fires in the month", "SUM(fact_fires_month[fires])", "#,0", "02 Fire record"),
    ("Share of the season",
     "DIVIDE([Fires in the month], CALCULATE([Fires in the month], ALL(fact_fires_month[month], fact_fires_month[month_name])))",
     "0.0%", "02 Fire record"),

    # --- the out-of-time backtest ------------------------------------------
    ("Fires in the test seasons", "SUM(fact_backtest_day[fires])", "#,0", "03 Backtest"),
    ("Fires in the riskiest tenth", "SUM(fact_backtest_day[fires_in_top_decile])", "#,0", "03 Backtest"),
    ("Same-day capture",
     "DIVIDE([Fires in the riskiest tenth], [Fires in the test seasons])", "0.0%", "03 Backtest"),
    ("Large fires in the test seasons", "SUM(fact_backtest_day[large_fires])", "#,0", "03 Backtest"),
    ("Large fires in the riskiest tenth", "SUM(fact_backtest_day[large_in_top_decile])",
     "#,0", "03 Backtest"),
    ("Same-day capture, large fires",
     "DIVIDE([Large fires in the riskiest tenth], [Large fires in the test seasons])",
     "0.0%", "03 Backtest"),
    ("Days scored", "DISTINCTCOUNT(fact_backtest_day[date])", "#,0", "03 Backtest"),
    ("Cells ranked each day", "AVERAGE(fact_backtest_day[cells])", "#,0", "03 Backtest"),
    ("Cell fires", "SUM(fact_backtest_cell[fires])", "#,0", "03 Backtest"),
    ("Cell fires caught", "SUM(fact_backtest_cell[fires_caught])", "#,0", "03 Backtest"),
    ("Capture by cell", "DIVIDE([Cell fires caught], [Cell fires])", "0.0%", "03 Backtest"),
    ("Days a cell was flagged", "SUM(fact_backtest_cell[days_flagged])", "#,0", "03 Backtest"),
    ("Mean risk in the backtest", "AVERAGE(fact_backtest_cell[mean_risk])", "0.0%", "03 Backtest"),

    # --- how it scored -----------------------------------------------------
    ("ROC-AUC", "AVERAGE(fact_model_scores[roc_auc])", "0.000", "04 Scores"),
    ("PR-AUC", "AVERAGE(fact_model_scores[pr_auc])", "0.000", "04 Scores"),
    ("Fires in the top decile", "AVERAGE(fact_model_scores[share_of_fires_in_top_decile])",
     "0.0%", "04 Scores"),
    ("Lift", "AVERAGE(fact_model_scores[lift_top_decile])", "0.0", "04 Scores"),
    ("Brier score", "AVERAGE(fact_model_scores[brier])", "0.0000", "04 Scores"),
    ("Model ROC-AUC out of time",
     "CALCULATE([ROC-AUC], fact_model_scores[method] = \"model\", fact_model_scores[split] = \"Test\","
     " fact_model_scores[target] = \"has_fire\")", "0.000", "04 Scores"),
    ("Climatology ROC-AUC out of time",
     "CALCULATE([ROC-AUC], fact_model_scores[method] = \"climatology\", fact_model_scores[split] = \"Test\","
     " fact_model_scores[target] = \"has_fire\")", "0.000", "04 Scores"),
    ("Ahead of climatology",
     "[Model ROC-AUC out of time] - [Climatology ROC-AUC out of time]", "0.000", "04 Scores"),
    ("ROC-AUC by year", "AVERAGE(fact_model_year[roc_auc])", "0.000", "04 Scores"),
    ("Capture by year", "AVERAGE(fact_model_year[share_of_fires_in_top_decile])", "0.0%", "04 Scores"),
    ("ROC-AUC by province", "AVERAGE(fact_model_province[roc_auc])", "0.000", "04 Scores"),
    ("Capture by province", "AVERAGE(fact_model_province[share_of_fires_in_top_decile])",
     "0.0%", "04 Scores"),
    ("Cell-days behind the province scores", "SUM(fact_model_province[rows])", "#,0", "04 Scores"),
    ("Predicted", "AVERAGE(fact_reliability[predicted])", "0.0%", "04 Scores"),
    ("Observed", "AVERAGE(fact_reliability[observed])", "0.0%", "04 Scores"),
    ("Share of the tree gain", "SUM(fact_feature_gain[gain])", "0.0%", "04 Scores"),

    # The report writer has no per-visual filter, and that is deliberate: a
    # filter that lives in a visual's JSON is invisible in the model and
    # invisible in the field well, so the number on the page cannot be
    # explained by reading either. Where a chart needs one slice of the scores,
    # the slice is named here, in a measure, where it can be read and tested.
    ("ROC-AUC out of time",
     "CALCULATE([ROC-AUC], fact_model_scores[split] = \"Test\")", "0.000", "04 Scores"),
    ("PR-AUC out of time",
     "CALCULATE([PR-AUC], fact_model_scores[split] = \"Test\")", "0.000", "04 Scores"),
    ("Top-decile capture out of time",
     "CALCULATE([Fires in the top decile], fact_model_scores[split] = \"Test\")", "0.0%", "04 Scores"),
    ("Lift out of time",
     "CALCULATE([Lift], fact_model_scores[split] = \"Test\")", "0.0", "04 Scores"),
    # A card with no row context averages every method and both targets, which is
    # how a report ends up showing 38.6% for a capture rate that is 46.3% for the
    # model and 23.1% for the baseline. Each headline names its own slice.
    ("Capture out of time, this model",
     "CALCULATE([Fires in the top decile], fact_model_scores[method] = \"model\","
     " fact_model_scores[split] = \"Test\", fact_model_scores[target] = \"has_fire\")",
     "0.0%", "04 Scores"),
    ("Lift out of time, this model",
     "CALCULATE([Lift], fact_model_scores[method] = \"model\","
     " fact_model_scores[split] = \"Test\", fact_model_scores[target] = \"has_fire\")",
     "0.0", "04 Scores"),
    ("Brier score, this model",
     "CALCULATE([Brier score], fact_model_scores[method] = \"model\","
     " fact_model_scores[split] = \"Test\", fact_model_scores[target] = \"has_fire\")",
     "0.0000", "04 Scores"),
    ("ROC-AUC out of time, any fire",
     "CALCULATE([ROC-AUC], fact_model_scores[split] = \"Test\", fact_model_scores[target] = \"has_fire\")",
     "0.000", "04 Scores"),
    ("Top-decile capture out of time, any fire",
     "CALCULATE([Fires in the top decile], fact_model_scores[split] = \"Test\","
     " fact_model_scores[target] = \"has_fire\")", "0.0%", "04 Scores"),
    ("ROC-AUC by year, any fire",
     "CALCULATE([ROC-AUC by year], fact_model_year[target] = \"has_fire\")", "0.000", "04 Scores"),
    ("Capture by year, any fire",
     "CALCULATE([Capture by year], fact_model_year[target] = \"has_fire\")", "0.0%", "04 Scores"),
    ("ROC-AUC by province, any fire",
     "CALCULATE([ROC-AUC by province], fact_model_province[target] = \"has_fire\")", "0.000", "04 Scores"),
    ("Capture by province, any fire",
     "CALCULATE([Capture by province], fact_model_province[target] = \"has_fire\")", "0.0%", "04 Scores"),
    ("Predicted, any fire",
     "CALCULATE([Predicted], fact_reliability[target] = \"has_fire\")", "0.0%", "04 Scores"),
    ("Observed, any fire",
     "CALCULATE([Observed], fact_reliability[target] = \"has_fire\")", "0.0%", "04 Scores"),
    ("Share of the tree gain, any fire",
     "CALCULATE([Share of the tree gain], fact_feature_gain[target] = \"has_fire\")",
     "0.0%", "04 Scores"),

    # --- the current season, graded by satellite ---------------------------
    ("New detections", "SUM(fact_season_day[detections])", "#,0", "05 This season"),
    ("Detections in the riskiest tenth", "SUM(fact_season_day[detections_in_top_decile])",
     "#,0", "05 This season"),
    ("Season capture", "DIVIDE([Detections in the riskiest tenth], [New detections])",
     "0.0%", "05 This season"),
    ("Season ROC-AUC", "AVERAGE(fact_season_scores[roc_auc])", "0.000", "05 This season"),
    ("Season same-day capture", "AVERAGE(fact_season_scores[same_day_top_decile])",
     "0.0%", "05 This season"),
    ("Season ROC-AUC, this model",
     "CALCULATE([Season ROC-AUC], fact_season_scores[method] = \"model\")",
     "0.000", "05 This season"),
    ("Season ROC-AUC by month", "AVERAGE(fact_season_month[model_roc_auc])", "0.000", "05 This season"),
    ("Season cell-days", "SUM(fact_season_month[cell_days])", "#,0", "05 This season"),
    ("Cell detections", "SUM(fact_season_cell[detections])", "#,0", "05 This season"),

    # --- the 2024 model ----------------------------------------------------
    ("R squared", "AVERAGE(fact_hackathon_scores[r2])", "0.000", "06 The 2024 model"),
    ("Mean error, hectares", "AVERAGE(fact_hackathon_scores[mae])", "#,0", "06 The 2024 model"),
    ("Months scored", "AVERAGE(fact_hackathon_scores[rows_test])", "#,0", "06 The 2024 model"),
    ("Hectares burned", "SUM(fact_hackathon_month[actual_ha])", "#,0", "06 The 2024 model"),
    ("Hectares the 2024 model forecast",
     "SUM(fact_hackathon_month[predicted_ha_without_count])", "#,0", "06 The 2024 model"),
    ("How wrong the 2024 model was",
     "DIVIDE([Hectares the 2024 model forecast] - [Hectares burned], [Hectares burned])",
     "0.0%", "06 The 2024 model"),
    ("R squared, the longer panel",
     "CALCULATE([R squared], fact_hackathon_scores[panel] = \"nfdb\")", "0.000", "06 The 2024 model"),
    ("Mean error, the longer panel",
     "CALCULATE([Mean error, hectares], fact_hackathon_scores[panel] = \"nfdb\")",
     "#,0", "06 The 2024 model"),
    ("R squared as scored in 2024",
     "CALCULATE([R squared], fact_hackathon_scores[scoring] = \"as_scored_in_2024\","
     " fact_hackathon_scores[panel] = \"nfdb\")", "0.000", "06 The 2024 model"),
    ("R squared out of time",
     "CALCULATE([R squared], fact_hackathon_scores[scoring] = \"out_of_time\","
     " fact_hackathon_scores[panel] = \"nfdb\")", "0.000", "06 The 2024 model"),
    ("What the random split was worth",
     "[R squared as scored in 2024] - [R squared out of time]", "0.000", "06 The 2024 model"),

    # --- reference lines under the tiles -----------------------------------
    ("Forecast reference",
     "\"Built \" & [Forecast built] & \"  ·  from station day \" & [Starts from]",
     "", "07 Reference"),
    ("Capture reference",
     "FORMAT([Fires in the riskiest tenth], \"#,0\") & \" of \" & FORMAT([Fires in the test seasons], \"#,0\")"
     " & \" fires, \" & FORMAT([Days scored], \"#,0\") & \" days\"",
     "", "07 Reference"),
    ("Baseline reference",
     "\"Climatology \" & FORMAT([Climatology ROC-AUC out of time], \"0.000\") & \"  ·  ahead by \""
     " & FORMAT([Ahead of climatology], \"0.000\")",
     "", "07 Reference"),
    ("Season reference",
     "FORMAT([New detections], \"#,0\") & \" new detections over \" & FORMAT([Season cell-days], \"#,0\")"
     " & \" cell-days\"",
     "", "07 Reference"),
    ("Fire record reference",
     "FORMAT([Lightning share of fires], \"0%\") & \" of starts are lightning, \""
     " & FORMAT([Lightning share of area], \"0%\") & \" of the area\"",
     "", "07 Reference"),
    ("Riskiest tenth reference",
     "FORMAT([Cells in the riskiest tenth], \"#,0\") & \" of \" & FORMAT([Cells forecast], \"#,0\")"
     " & \" cells, \" & FORMAT([Risk against normal], \"0.0\") & \"x normal\"",
     "", "07 Reference"),
    ("The 2024 model reference",
     "\"Random split \" & FORMAT([R squared as scored in 2024], \"0.000\") & \"  ·  out of time \""
     " & FORMAT([R squared out of time], \"0.000\")",
     "", "07 Reference"),

    # --- the verdict each page opens with ----------------------------------
    # Calibrated risk comes in plateaus, so the peak is usually shared: on the
    # 16 September forecast six cells in three provinces sat at 11.4%, and a
    # TOPN(1) broken by cell id named Ontario alone. Every province at the
    # peak is named. And the riskiest tenth is a tenth of the cells a day; over a week it
    # is however many cells made it on any day, which is what gets said.
    ("Forecast verdict",
     "VAR vCells = [Cells in the riskiest tenth]\n"
     "VAR vDays = DISTINCTCOUNT(fact_forecast[date])\n"
     "VAR vPeak = [Peak risk]\n"
     "VAR vNormal = [Risk against normal]\n"
     "VAR vAtPeak = FILTER(VALUES(fact_forecast[cell_id]), [Peak risk] = vPeak)\n"
     "VAR vPeakCells = COUNTROWS(vAtPeak)\n"
     "VAR vPlaces = CALCULATETABLE(VALUES(fact_forecast[province]), vAtPeak)\n"
     "VAR vPlaceCount = COUNTROWS(vPlaces)\n"
     "VAR vFirst = CONCATENATEX(TOPN(1, vPlaces, fact_forecast[province], ASC), fact_forecast[province])\n"
     "VAR vLast = CONCATENATEX(TOPN(1, vPlaces, fact_forecast[province], DESC), fact_forecast[province])\n"
     "VAR vMiddle = CONCATENATEX(FILTER(vPlaces, fact_forecast[province] <> vFirst"
     " && fact_forecast[province] <> vLast), fact_forecast[province], \", \", fact_forecast[province], ASC)\n"
     "VAR vWhere = SWITCH(TRUE(),\n"
     "    vPlaceCount = 1, vFirst,\n"
     "    vPlaceCount = 2, vFirst & \" and \" & vLast,\n"
     "    vPlaceCount = 3, vFirst & \", \" & vMiddle & \" and \" & vLast,\n"
     "    FORMAT(vPlaceCount, \"0\") & \" provinces and territories\")\n"
     "RETURN\n"
     "    IF(ISBLANK(vPeak), BLANK(),\n"
     "        IF(vDays = 1,\n"
     "            FORMAT(vCells, \"#,0\") & \" cells make the day's riskiest tenth\",\n"
     "            FORMAT(vCells, \"#,0\") & \" cells make the riskiest tenth on at least one of these \""
     " & FORMAT(vDays, \"0\") & \" days\")\n"
     "        & \". The highest chance of a new fire is \" & FORMAT(vPeak, \"0.0%\")\n"
     "        & IF(vPeakCells > 1, \", shared by \" & FORMAT(vPeakCells, \"0\") & \" cells in \", \" in \")"
     " & vWhere\n"
     "        & \". Across the grid the forecast runs at \" & FORMAT(vNormal, \"0.0\")"
     " & \"x the normal rate for this month.\")",
     "", "08 Verdict"),
    ("Backtest verdict",
     "\"Ranked fresh every morning, the riskiest tenth of cells held \" & FORMAT([Same-day capture], \"0.0%\")"
     " & \" of the \" & FORMAT([Fires in the test seasons], \"#,0\") & \" fires reported over \""
     " & FORMAT([Days scored], \"#,0\") & \" days of the 2020-2024 seasons, and \""
     " & FORMAT([Same-day capture, large fires], \"0.0%\") & \" of the fires that grew past 200 hectares.\"",
     "", "08 Verdict"),
    ("Scores verdict",
     "\"Out of time, the model scores \" & FORMAT([Model ROC-AUC out of time], \"0.000\")"
     " & \" against \" & FORMAT([Climatology ROC-AUC out of time], \"0.000\") & \" for each cell's own rate\""
     " & \" for the month - \" & FORMAT([Ahead of climatology], \"0.000\") & \" ahead of the baseline that\""
     " & \" is hardest to beat, and the one a fire agency already has.\"",
     "", "08 Verdict"),
]
