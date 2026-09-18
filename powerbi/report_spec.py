"""
What the report contains: eight pages, and every visual on them.

Declarative on purpose. A PBIR report is one JSON file per visual, and typing
them by hand is how a report ends up with two ``visualContainer`` schema
versions, a mistyped property that renders nothing, and no way to check either.
Here the pages are data, the writer is one function, and the tests assert on this
module rather than on the JSON it produces.

The order of the pages is the argument the report makes: what the model says
about the next seven days, what the fire record it learned from looks like, how
it scored on seasons it never saw, where that holds and where it does not, how
the current season is going against satellites, and what the 2024 hackathon model
- the ancestor of all of it - scores when it is re-scored the same way.

Visual shorthand::

    card(field, subtitle=measure)
    bar / column / stacked_column / line / area / scatter / table / matrix / slicer

Fields are written ``table[column]`` for a column and ``[Measure]`` for a
measure, which is the notation Desktop shows in the field well.
"""

from __future__ import annotations

# Canvas is 1280x720 at FitToPage. A card row sits at y=20 with height 118 - a
# card carrying a reference subtitle needs at least 118px or the label is
# clipped, and clipping is silent.
CARD_Y = 20
CARD_H = 118
ROW1_Y = 152
ROW2_Y = 442
FULL_W = 1240

PAGES: list[dict] = [
    # ----------------------------------------------------------------- 1
    {
        "name": "section_forecast",
        "display": "Seven-day fire risk",
        "visuals": [
            {"type": "card", "field": "[Cells in the riskiest tenth]",
             "subtitle": "[Riskiest tenth reference]", "pos": (20, CARD_Y, 320, CARD_H),
             "alt": "Card. How many cells sit in the day's riskiest tenth, out of the "
                    "771 in the grid, and how that compares with the normal rate for "
                    "the month."},
            {"type": "card", "field": "[Peak risk]", "pos": (350, CARD_Y, 290, CARD_H),
             "alt": "Card. The highest probability of a new fire in any cell over the "
                    "forecast week."},
            {"type": "card", "field": "[Peak large-fire risk]", "pos": (650, CARD_Y, 290, CARD_H),
             "alt": "Card. The highest probability that a fire reported in a cell grows "
                    "past 200 hectares."},
            {"type": "card", "field": "[Hotspots in the last two days]",
             "subtitle": "[Forecast reference]", "pos": (950, CARD_Y, 310, CARD_H),
             "alt": "Card. Satellite hotspots CWFIS published in the last two days, with "
                    "when the forecast was built and which station day it starts from."},

            {"type": "scatter", "category": "dim_cell[cell_id]", "x": "[Mean FWI]",
             "y": ["[Mean risk]"], "size": "[Peak risk]",
             "title": "Every cell: fire weather against modelled risk",
             "pos": (20, ROW1_Y, 620, 256),
             "alt": "Scatter chart titled Every cell: fire weather against modelled risk. "
                    "One point per grid cell, plotting mean risk against mean Fire Weather "
                    "Index over the forecast week, sized by the cell's peak risk."},
            {"type": "line", "x": "fact_forecast[lead_days]", "y": ["[Peak risk]", "[Mean risk]"],
             "title": "The week ahead, day by day",
             "pos": (654, ROW1_Y, 606, 256),
             "alt": "Line chart titled The week ahead day by day. Plots the peak and the "
                    "mean chance of a new fire by lead day, from the latest observed station "
                    "day through seven forecast days. Both are probabilities, so they share "
                    "an axis; the Fire Weather Index is on the chart beside it."},

            {"type": "column", "x": "dim_province[province_short]", "y": ["[Mean risk]"],
             "title": "Where the risk sits this week",
             "pos": (20, ROW2_Y - 16, 500, 164),
             "alt": "Bar chart titled Where the risk sits this week. Plots mean risk by "
                    "province or territory."},
            {"type": "table",
             "columns": ["dim_cell[cell_id]", "dim_cell[province_short]", "[Peak risk]",
                         "[Peak large-fire risk]", "[Mean FWI]", "[Risk against normal]"],
             "sort": ("[Peak risk]", "Descending"),
             "totals": False,
             "title": "The cells to watch",
             "pos": (534, ROW2_Y - 16, 726, 164),
             "alt": "Table titled The cells to watch. Lists each grid cell with its peak "
                    "new-fire risk, peak large-fire risk, mean Fire Weather Index and how "
                    "that risk compares with the cell's normal rate for the month."},

            {"type": "narrative", "field": "[Forecast verdict]",
             "title": "What the forecast is saying",
             # Two lines: the sentence names every province at the peak, and
             # three long names do not fit on one.
             "pos": (20, ROW2_Y + 158, FULL_W, 116),
             "alt": "Narrative. A sentence generated from the measures on this page: how "
                    "many cells are in the day's riskiest tenth, the peak risk and where it "
                    "is, and how the week compares with the normal rate for the month."},
            {"type": "slicer", "field": "dim_province[province]", "title": "Province or territory",
             "pos": (1068, ROW2_Y - 16, 192, 76),
             "alt": "Slicer. Filters the page by province or territory."},
            {"type": "slicer", "field": "fact_forecast[lead_days]", "title": "Lead day",
             "pos": (1068, ROW2_Y + 70, 192, 76),
             "alt": "Slicer. Filters the page by lead day, 0 for the latest observed day."},
        ],
    },
    # ----------------------------------------------------------------- 2
    {
        "name": "section_record",
        "display": "Canada’s fires since 2000",
        "visuals": [
            {"type": "card", "field": "[Fires recorded]", "subtitle": "[Fire record reference]",
             "pos": (20, CARD_Y, 320, CARD_H),
             "alt": "Card. Fires in the National Fire Database point layer, with the "
                    "lightning share of starts and of area burned."},
            {"type": "card", "field": "[Area burned (million ha)]", "pos": (350, CARD_Y, 290, CARD_H),
             "alt": "Card. Total area burned, in millions of hectares."},
            {"type": "card", "field": "[Large fires recorded]", "pos": (650, CARD_Y, 290, CARD_H),
             "alt": "Card. Fires that grew past 200 hectares."},
            {"type": "card", "field": "[Lightning share of area]", "pos": (950, CARD_Y, 310, CARD_H),
             "alt": "Card. The share of area burned that lightning started."},

            {"type": "stacked_column", "x": "dim_season[year]", "y": ["[Fires recorded]"],
             "series": "fact_fires_year[cause_label]",
             "title": "Fires reported each year, by cause",
             "pos": (20, ROW1_Y, 620, 272),
             "alt": "Stacked column chart titled Fires reported each year by cause. Plots "
                    "the count of fires by year, split into lightning, human and unknown."},
            {"type": "stacked_column", "x": "dim_season[year]", "y": ["[Area burned (million ha)]"],
             "series": "fact_fires_year[cause_label]",
             "title": "Area burned each year, by cause",
             "pos": (654, ROW1_Y, 606, 272),
             "alt": "Stacked column chart titled Area burned each year by cause. Plots "
                    "millions of hectares by year, split into lightning, human and unknown."},

            {"type": "bar", "x": "dim_province[province_short]", "y": ["[Fires recorded]"],
             "title": "Which provinces report the fires",
             "pos": (20, ROW2_Y, 500, 258),
             "alt": "Bar chart titled Which provinces report the fires. Plots the count of "
                    "fires by province or territory."},
            {"type": "column", "x": "fact_fires_month[month_name]", "y": ["[Fires in the month]"],
             "title": "When the season runs",
             "pos": (534, ROW2_Y, 500, 258),
             "alt": "Column chart titled When the season runs. Plots the count of fires by "
                    "calendar month across the whole record."},

            {"type": "slicer", "field": "dim_province[province]", "title": "Province or territory",
             "pos": (1068, ROW2_Y, 192, 76),
             "alt": "Slicer. Filters the page by province or territory."},
            {"type": "slicer", "field": "fact_fires_year[cause_label]", "title": "Cause",
             "pos": (1068, ROW2_Y + 86, 192, 76),
             "alt": "Slicer. Filters the page by reported cause."},
            {"type": "slicer", "field": "dim_season[decade]", "title": "Decade",
             "pos": (1068, ROW2_Y + 172, 192, 76),
             "alt": "Slicer. Filters the page by decade."},
        ],
    },
    # ----------------------------------------------------------------- 3
    {
        "name": "section_backtest",
        "display": "Out of time: 2020-2024",
        "visuals": [
            {"type": "card", "field": "[Same-day capture]", "subtitle": "[Capture reference]",
             "pos": (20, CARD_Y, 320, CARD_H),
             "alt": "Card. The share of fires that started inside the day's riskiest tenth "
                    "of cells, with the counts behind it."},
            {"type": "card", "field": "[Same-day capture, large fires]",
             "pos": (350, CARD_Y, 290, CARD_H),
             "alt": "Card. The same measure for fires that grew past 200 hectares."},
            {"type": "card", "field": "[Fires in the test seasons]", "pos": (650, CARD_Y, 290, CARD_H),
             "alt": "Card. Fires reported across the 2020 to 2024 seasons."},
            {"type": "card", "field": "[Days scored]", "pos": (950, CARD_Y, 310, CARD_H),
             "alt": "Card. Days of the test seasons on which every cell was ranked."},

            {"type": "line", "x": "dim_date[date]",
             "y": ["[Fires in the test seasons]", "[Fires in the riskiest tenth]"],
             "title": "Every day of the test seasons: fires, and fires inside the riskiest tenth",
             "pos": (20, ROW1_Y, FULL_W, 272),
             "alt": "Line chart titled Every day of the test seasons. Plots the fires "
                    "reported each day and how many of them started inside the day's "
                    "riskiest tenth of cells."},

            {"type": "column", "x": "dim_season[year]", "y": ["[Capture by year, any fire]"],
             "title": "Capture, season by season",
             "pos": (20, ROW2_Y, 500, 258),
             "alt": "Column chart titled Capture season by season. Plots the share of fires "
                    "inside the riskiest tenth for each test season."},
            {"type": "table",
             "columns": ["dim_date[date]", "[Fires in the test seasons]",
                         "[Fires in the riskiest tenth]", "[Same-day capture]"],
             "sort": ("[Fires in the test seasons]", "Descending"),
             "title": "The busiest days, and what the ranking caught",
             "pos": (534, ROW2_Y, 726, 258),
             "alt": "Table titled The busiest days and what the ranking caught. Lists the "
                    "days with the most fires, how many started inside the riskiest tenth, "
                    "and the resulting capture rate."},

            {"type": "slicer", "field": "dim_season[year]", "title": "Season",
             "pos": (1068, ROW2_Y, 192, 76),
             "alt": "Slicer. Filters the page by test season."},
        ],
    },
    # ----------------------------------------------------------------- 4
    {
        "name": "section_scores",
        "display": "What it scored, and against what",
        "visuals": [
            {"type": "card", "field": "[Model ROC-AUC out of time]",
             "subtitle": "[Baseline reference]", "pos": (20, CARD_Y, 320, CARD_H),
             "alt": "Card. The model's ROC-AUC on the test seasons, with climatology's "
                    "score and the gap between them."},
            {"type": "card", "field": "[Capture out of time, this model]",
             "pos": (350, CARD_Y, 290, CARD_H),
             "alt": "Card. The share of fires inside the riskiest tenth of all test "
                    "cell-days, ranked once rather than per day."},
            {"type": "card", "field": "[Lift out of time, this model]", "pos": (650, CARD_Y, 290, CARD_H),
             "alt": "Card. How many times more likely a fire is in the riskiest tenth than "
                    "in an average cell-day."},
            {"type": "card", "field": "[Brier score, this model]", "pos": (950, CARD_Y, 310, CARD_H),
             "alt": "Card. Brier score: the mean squared error of the calibrated "
                    "probability, where lower is better."},

            {"type": "bar", "x": "fact_model_scores[method_label]",
             "y": ["[ROC-AUC out of time, any fire]"],
             "title": "The model against the baselines it has to beat",
             "pos": (20, ROW1_Y, 620, 272),
             "alt": "Bar chart titled The model against the baselines it has to beat. Plots "
                    "ROC-AUC on the test seasons for the model, for each cell's normal rate "
                    "for the month, and for a logistic regression on the Fire Weather Index."},
            {"type": "line", "x": "dim_season[year]", "y": ["[ROC-AUC by year, any fire]"],
             "title": "Season by season, out of time",
             "pos": (654, ROW1_Y, 606, 272),
             "alt": "Line chart titled Season by season out of time. Plots the model's "
                    "ROC-AUC for each test season."},

            {"type": "matrix", "rows": "fact_model_scores[method_label]",
             "columns_by": "fact_model_scores[target_label]",
             # Two measures, not three: six columns of numbers under two target
             # headings do not fit the half-width tile, and a cross-tab that has
             # to be scrolled sideways to be read is a table nobody reads.
             "values": ["[ROC-AUC out of time]", "[Top-decile capture out of time]"],
             "totals": False,
             "title": "Both questions, all three rankings",
             "pos": (20, ROW2_Y, 620, 258),
             "alt": "Matrix titled Both questions all three rankings. Rows are the model "
                    "and its two baselines, columns are the two targets, values are "
                    "ROC-AUC, PR-AUC and the share of fires in the top decile."},
            {"type": "scatter", "category": "fact_reliability[decile]",
             "x": "[Predicted, any fire]", "y": ["[Observed, any fire]"],
             "title": "Calibration: what it predicted against what happened",
             "pos": (654, ROW2_Y, 606, 258),
             "alt": "Scatter chart titled Calibration: what it predicted against what "
                    "happened. One point per decile of predicted risk, plotting the "
                    "observed fire rate against the predicted probability."},
        ],
    },
    # ----------------------------------------------------------------- 5
    {
        "name": "section_where",
        "display": "Where it works, and where it does not",
        "visuals": [
            {"type": "card", "field": "[Capture out of time, this model]",
             "subtitle": "[Baseline reference]", "pos": (20, CARD_Y, 320, CARD_H),
             "alt": "Card. The share of fires inside the riskiest tenth, for whatever is "
                    "selected."},
            {"type": "card", "field": "[Model ROC-AUC out of time]",
             "pos": (350, CARD_Y, 290, CARD_H),
             "alt": "Card. ROC-AUC for the selected provinces on the test seasons."},
            {"type": "card", "field": "[Cell-days behind the province scores]",
             "pos": (650, CARD_Y, 290, CARD_H),
             "alt": "Card. How many scored cell-days sit behind the province figures."},
            {"type": "card", "field": "[Cells forecast]", "pos": (950, CARD_Y, 310, CARD_H),
             "alt": "Card. Cells in the grid."},

            {"type": "bar", "x": "fact_model_province[province_short]",
             "y": ["[ROC-AUC by province, any fire]"],
             "sort": ("[ROC-AUC by province, any fire]", "Descending"),
             "title": "ROC-AUC by province, on the test seasons",
             "pos": (20, ROW1_Y, 620, 272),
             "alt": "Bar chart titled ROC-AUC by province on the test seasons. Plots the "
                    "model's ROC-AUC for each province with at least thirty fire days."},
            {"type": "bar", "x": "fact_feature_gain[feature]",
             "y": ["[Share of the tree gain, any fire]"],
             "sort": ("[Share of the tree gain, any fire]", "Descending"),
             "title": "What the trees split on",
             "pos": (654, ROW1_Y, 606, 272),
             "alt": "Bar chart titled What the trees split on. Plots each feature's share "
                    "of the boosted trees' total gain, for the any-fire model."},

            {"type": "scatter", "category": "dim_cell[cell_id]", "x": "[Mean risk in the backtest]",
             "y": ["[Capture by cell]"], "size": "[Cell fires]",
             "title": "Every cell over 2020-2024: how risky, and how much it caught",
             "pos": (20, ROW2_Y, 620, 258),
             "alt": "Scatter chart titled Every cell over 2020 to 2024. Plots each cell's "
                    "capture rate against its mean modelled risk, sized by the fires it "
                    "actually reported."},
            {"type": "table",
             "columns": ["dim_province[province_short]", "[Cell fires]", "[Cell fires caught]",
                         "[Capture by cell]", "[Days a cell was flagged]"],
             "sort": ("[Cell fires]", "Descending"),
             "title": "Province by province",
             "pos": (654, ROW2_Y, 606, 258),
             "alt": "Table titled Province by province. Lists fires reported, fires caught "
                    "inside the riskiest tenth, the capture rate, and how many cell-days "
                    "were flagged."},

            {"type": "slicer", "field": "dim_season[year]", "title": "Season",
             "pos": (1068, ROW2_Y, 192, 76),
             "alt": "Slicer. Filters the page by test season."},
        ],
    },
    # ----------------------------------------------------------------- 6
    {
        "name": "section_season",
        "display": "This season, graded by satellite",
        "visuals": [
            {"type": "card", "field": "[New detections]", "subtitle": "[Season reference]",
             "pos": (20, CARD_Y, 320, CARD_H),
             "alt": "Card. Cell-days on which satellites detected new fire activity this "
                    "season, over the cell-days scored."},
            {"type": "card", "field": "[Season capture]", "pos": (350, CARD_Y, 290, CARD_H),
             "alt": "Card. The share of those detections that fell inside the day's "
                    "riskiest tenth of cells."},
            {"type": "card", "field": "[Season ROC-AUC, this model]", "pos": (650, CARD_Y, 290, CARD_H),
             "alt": "Card. ROC-AUC against satellite detections for whichever ranking is "
                    "selected."},
            {"type": "card", "field": "[Cells forecast]", "pos": (950, CARD_Y, 310, CARD_H),
             "alt": "Card. Cells in the grid."},

            {"type": "bar", "x": "fact_season_scores[method_label]", "y": ["[Season ROC-AUC]"],
             "sort": ("[Season ROC-AUC]", "Descending"),
             "title": "Against satellite detections: the models, the index and the baseline",
             "pos": (20, ROW1_Y, 620, 272),
             "alt": "Bar chart titled Against satellite detections. Plots ROC-AUC for the "
                    "any-fire model, the large-fire model, the Fire Weather Index alone and "
                    "each cell's normal rate for the month."},
            {"type": "line", "x": "dim_date[date]",
             "y": ["[New detections]", "[Detections in the riskiest tenth]"],
             "title": "New fire activity each day of the season",
             "pos": (654, ROW1_Y, 606, 272),
             "alt": "Line chart titled New fire activity each day of the season. Plots "
                    "satellite detections of new activity and how many fell inside the "
                    "day's riskiest tenth."},

            {"type": "column", "x": "fact_season_month[month_name]", "y": ["[Season ROC-AUC by month]"],
             "sort": ("fact_season_month[month_name]", "Ascending"),
             "title": "Month by month",
             "pos": (20, ROW2_Y, 500, 258),
             "alt": "Column chart titled Month by month. Plots the any-fire model's ROC-AUC "
                    "against satellite detections for each month of the season."},
            {"type": "table",
             "columns": ["fact_season_scores[method_label]", "[Season ROC-AUC]",
                         "[Season same-day capture]"],
             "sort": ("[Season ROC-AUC]", "Descending"),
             "totals": False,
             "title": "The season's scoreboard",
             "pos": (534, ROW2_Y, 726, 258),
             "alt": "Table titled The season's scoreboard. Lists each ranking with its "
                    "ROC-AUC against satellite detections and its same-day capture rate."},
        ],
    },
    # ----------------------------------------------------------------- 7
    {
        "name": "section_2024",
        "display": "The 2024 model, re-scored",
        "visuals": [
            {"type": "card", "field": "[R squared as scored in 2024]",
             "subtitle": "[The 2024 model reference]", "pos": (20, CARD_Y, 320, CARD_H),
             "alt": "Card. The hackathon model's R squared on a random 80/20 split over "
                    "months, the way it was scored in 2024."},
            {"type": "card", "field": "[R squared out of time]", "pos": (350, CARD_Y, 290, CARD_H),
             "alt": "Card. The same model's R squared when it is fitted to months up to "
                    "2016 and scored on 2020 to 2024."},
            {"type": "card", "field": "[What the random split was worth]",
             "pos": (650, CARD_Y, 290, CARD_H),
             "alt": "Card. The difference between the two, which is what the random split "
                    "was adding."},
            {"type": "card", "field": "[Mean error, the longer panel]",
             "pos": (950, CARD_Y, 310, CARD_H),
             "alt": "Card. The mean absolute error in hectares for the selected scoring."},

            {"type": "bar", "x": "fact_hackathon_scores[scoring_label]",
             "y": ["[R squared, the longer panel]"],
             "sort": ("fact_hackathon_scores[scoring_label]", "Ascending"),
             "title": "The same forest, scored three ways",
             "pos": (20, ROW1_Y, 620, 272),
             "alt": "Bar chart titled The same forest scored three ways. Plots R squared "
                    "for the random split, for the out-of-time split, and for the "
                    "out-of-time split without the month's own fire count."},
            {"type": "line", "x": "fact_hackathon_month[month_name]",
             "y": ["[Hectares burned]", "[Hectares the 2024 model forecast]"],
             "title": "Hectares burned each month, against what it forecast",
             "pos": (654, ROW1_Y, 606, 272),
             "alt": "Line chart titled Hectares burned each month against what it forecast. "
                    "Plots the hectares that actually burned and the hackathon model's "
                    "forecast, by calendar month."},

            {"type": "column", "x": "dim_season[year]",
             "y": ["[Hectares burned]", "[Hectares the 2024 model forecast]"],
             "title": "Year by year, over the test window",
             "pos": (20, ROW2_Y, 620, 258),
             "alt": "Column chart titled Year by year over the test window. Plots hectares "
                    "burned and the hackathon model's forecast for each of 2020 to 2024."},
            {"type": "table",
             "columns": ["fact_hackathon_scores[panel]", "fact_hackathon_scores[scoring_label]",
                         "[R squared]", "[Mean error, hectares]", "[Months scored]"],
             "totals": False,
             "title": "Both panels, and what each was scored on",
             "pos": (654, ROW2_Y, 606, 258),
             "alt": "Table titled Both panels and what each was scored on. Lists the "
                    "National Fire Database monthly tables the 2024 project read and the "
                    "point layer this repository builds, each with its three scorings."},
        ],
    },
    # ----------------------------------------------------------------- 8
    {
        "name": "section_verdict",
        "display": "What the evidence says",
        "visuals": [
            {"type": "card", "field": "[Same-day capture]", "subtitle": "[Capture reference]",
             "pos": (20, CARD_Y, 320, CARD_H),
             "alt": "Card. The operational number: the share of fires that started inside "
                    "the day's riskiest tenth of cells across the test seasons."},
            {"type": "card", "field": "[Model ROC-AUC out of time]",
             "subtitle": "[Baseline reference]", "pos": (350, CARD_Y, 290, CARD_H),
             "alt": "Card. ROC-AUC out of time, with the climatology baseline beneath it."},
            {"type": "card", "field": "[Season capture]", "subtitle": "[Season reference]",
             "pos": (650, CARD_Y, 290, CARD_H),
             "alt": "Card. This season's capture rate against satellite detections."},
            {"type": "card", "field": "[R squared out of time]",
             "subtitle": "[The 2024 model reference]", "pos": (950, CARD_Y, 310, CARD_H),
             "alt": "Card. What the 2024 hackathon model scores when it is re-scored out "
                    "of time."},

            {"type": "column", "x": "dim_season[year]",
             "y": ["[Capture by year, any fire]"],
             "title": "Every test season: the share of fires in the riskiest tenth",
             "pos": (20, ROW1_Y, 620, 272),
             "alt": "Column chart titled Every test season. Plots the share of fires that "
                    "started inside the riskiest tenth of cell-days for each season from "
                    "2020 to 2024."},
            {"type": "bar", "x": "fact_model_scores[method_label]",
             "y": ["[Top-decile capture out of time, any fire]"],
             "title": "What the alternatives would have caught",
             "pos": (654, ROW1_Y, 606, 272),
             "alt": "Bar chart titled What the alternatives would have caught. Plots the "
                    "share of fires in the riskiest tenth for the model and for both "
                    "baselines on the test seasons."},

            {"type": "narrative", "field": "[Backtest verdict]",
             "title": "What the backtest says",
             "pos": (20, ROW2_Y, 620, 120),
             "alt": "Narrative. A sentence generated from the backtest measures: the share "
                    "of fires that started inside the day's riskiest tenth across the test "
                    "seasons, and the same figure for fires that grew past 200 hectares."},
            {"type": "narrative", "field": "[Scores verdict]",
             "title": "What the scores say",
             "pos": (654, ROW2_Y, 606, 120),
             "alt": "Narrative. A sentence generated from the score measures: the model's "
                    "ROC-AUC out of time against climatology's, and the gap between them."},
            {"type": "table",
             "columns": ["fact_model_scores[target_label]", "fact_model_scores[method_label]",
                         "[ROC-AUC out of time]", "[Top-decile capture out of time]",
                         "[Lift out of time]"],
             "sort": ("[ROC-AUC out of time]", "Descending"),
             "totals": False,
             "title": "The scoreboard, out of time",
             "pos": (20, ROW2_Y + 130, FULL_W, 128),
             "alt": "Table titled The scoreboard out of time. Lists both targets and all "
                    "three rankings with their ROC-AUC, the share of fires in the riskiest "
                    "tenth, and the lift over an average cell-day."},
        ],
    },
]

VISUAL_TYPES: dict[str, str] = {
    "card": "card",
    # A card bound to a text measure rather than a figure: the page's verdict,
    # written in DAX so it cannot drift from the numbers beside it.
    "narrative": "card",
    "bar": "clusteredBarChart",
    "column": "clusteredColumnChart",
    "stacked_column": "columnChart",
    "line": "lineChart",
    "area": "areaChart",
    "scatter": "scatterChart",
    "donut": "donutChart",
    "treemap": "treemap",
    "waterfall": "waterfallChart",
    "table": "tableEx",
    "matrix": "pivotTable",
    "slicer": "slicer",
    "gauge": "gauge",
    # The chrome report_chrome adds to every page. A card is still written
    # "card" above and generated as an image tile; see build_pbip.visual_json.
    "page_header": "image",
    "nav": "actionButton",
    "filters_button": "actionButton",
    "panel_close": "actionButton",
    "panel_clear": "actionButton",
    "panel_background": "shape",
    "panel_title": "textbox",
}

from powerbi.report_chrome import add_chrome  # noqa: E402 -- applied to the pages above

PAGES = add_chrome(PAGES)
