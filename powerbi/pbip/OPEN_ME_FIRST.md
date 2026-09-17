# How to open this Power BI project

Everything is here: the semantic model **and** the report, both authored as text
(TMDL and PBIR) and both generated from the specs in `powerbi/`. There is no
`.pbix` — a binary would make the whole thing unreviewable.

## Open it

1. Power BI Desktop, **File → Open → Browse this device**, and pick
   `CanadaWildfireRisk.pbip`.
2. **Home → Refresh.** The model reads the CSVs in `powerbi/data/`, whose paths
   come from the `DataPath` parameter — which is written with an absolute path by
   the generator, so if you cloned this somewhere else, change it under
   **Transform data → Manage parameters** or simply regenerate:

   ```
   python pipelines/powerbi_export.py     # write powerbi/data/*.csv
   python -m powerbi.build_pbip           # write powerbi/pbip
   ```

3. If a page looks empty, it is almost always the refresh: the report ships with
   no cached data.

## Do not hand-edit it

The project is generated. `python -m powerbi.build_pbip --check` regenerates it
into a temporary directory and diffs, and CI fails if the committed project and
the spec have drifted apart — so an edit made in Desktop is silently reverted by
the next build. Change `powerbi/model_spec.py` or `powerbi/report_spec.py`
instead, and rebuild.

## What is on it

| Page | What it answers |
|---|---|
| Seven-day fire risk | Which cells are riskiest over the next week, and how that compares with normal |
| Canada's fires since 2000 | What the model learned from: 164,707 fires by year, cause, province and month |
| Out of time: 2020–2024 | What the ranking would have caught, day by day, on seasons it never saw |
| What it scored, and against what | ROC-AUC, PR-AUC and capture against both baselines, and the calibration curve |
| Where it works, and where it does not | Province by province, cell by cell, and what the trees split on |
| This season, graded by satellite | 2026 against CWFIS hotspot detections, with FWI alone as a competitor |
| The 2024 model, re-scored | The hackathon random forest, scored the way this repository scores everything |
| What the evidence says | The four headline numbers and the scoreboard, in one page |

The buttons on each page (**Next ›**, **‹ Previous**, **Filters**) are page
navigation. In Power BI Desktop a button needs **Ctrl+click**; in the Power BI
service and in reading view a plain click works.
