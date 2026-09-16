"""Where this started: the hackathon project, and what changed."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import shared as sh

sh.page_style()
st.title("From a hackathon win to a live forecast")
sh.kicker("This began as <b>Wildfire Prevention Strategy Using Technology</b>, the first-prize project of team "
          "1904 Coders - Mrityunjay Gupta, Siddharth Alashi and Kush Patel - at a 2024 hackathon.")

st.markdown(
    """
### The original project

- **A Power BI dashboard** over the National Forestry Database's summary tables - fires and area burned by
  jurisdiction, cause, size class and month since 1990 - with filters for jurisdiction and year and a
  predictive view for British Columbia.
- **A predictive notebook** that joined about 12,500 daily weather records to area burned. Its random forest
  explained 67% of the variance in area burned on a random hold-out (97% on its own training rows); the deck
  compared logistic regression, SVM and random forests and reported 48.72% accuracy for predicting occurrence.
- **A sensor prototype**: a DHT22 temperature and humidity sensor and an LM393 wind-speed sensor feeding a
  Django and PostgreSQL web service, with notifications to a website and a phone.

The deck's closing slide named the next step itself: more data, and more models for finer, more accurate
predictions. That is this project.
"""
)

st.markdown("### What changed")
st.markdown(
    """
| | 2024 hackathon | Now |
|---|---|---|
| **Question** | How much area burns? | Where will new fires be reported in the next seven days? |
| **Fire data** | Summary tables by jurisdiction | Every fire in the NFDB point layer, 2000–2024 |
| **Weather** | Daily weather records joined to area burned | Noon observations from about 3,200 CWFIS stations, plus a live forecast |
| **Fire science** | None | The Canadian FWI System, implemented and checked against CWFIS's own codes |
| **Unit** | One weather record | One 1° cell on one day, 771 cells |
| **Validation** | Random 80/20 split | Seasons held out in time, two baselines, and the 2026 season against satellite detections |
| **Delivery** | A .pbix file and a slide deck | This app, rebuilt twice a day from public data |
"""
)

report = sh.load_hackathon()
if report is not None:
    panel = report["panels"]["nfdb"]
    scorings = panel["scorings"]
    st.subheader("The 2024 model, rebuilt and re-scored")
    st.markdown(
        f"""
The winning notebook is rebuilt in [`pipelines/hackathon_2024.py`]({sh.REPO_URL}/blob/main/pipelines/hackathon_2024.py) -
the same random forest, the same eight features, the same target: **hectares burned in a month, nationally**.
Only the scoring changes. Weather is rebuilt from the CWFIS station archives, because the notebook's own
`Weather_area.csv` is not in the repository, so these numbers are this rebuild's rather than the deck's.
"""
    )
    rows = [
        ("As it was scored in 2024", "random 80/20 split over months", "as_scored_in_2024"),
        ("Scored out of time", "fit to 2016, scored on 2020–2024", "out_of_time"),
        ("Scored as a forecast", "out of time, without the month's own fire count",
         "out_of_time_without_the_fire_count"),
    ]
    table = pd.DataFrame([
        {"Scoring": label, "How": how, "R²": f"{scorings[key]['r2']:+.3f}",
         "Mean error": f"{scorings[key]['mae']:,.0f} ha",
         "Months scored": scorings[key]["rows_test"]}
        for label, how, key in rows
    ])
    st.dataframe(table, hide_index=True, width="stretch")
    drop = scorings["as_scored_in_2024"]["r2"] - scorings["out_of_time"]["r2"]
    st.markdown(
        f"""
**The split was doing the work.** Splitting months at random puts June 2015 in training and June 2016 in the
hold-out, and a fire season is strongly autocorrelated - so the score measures interpolation into a year the
model has already seen. Scored the way this repository scores everything, in time, the same forest loses
**{drop:.3f} of R²**. Taking away the month's own fire count - a number no forecast can have, because it counts
the fires whose area is being predicted - costs it more.

None of that makes the 2024 project wrong. It makes it a first pass, and it is the reason the model on the other
pages answers a different question: not how many hectares will burn somewhere in Canada this month, but which
cells will report a fire tomorrow - scored on cells and days, against baselines, on seasons it has never seen.
"""
    )
    months = sh.load_published("hackathon_2024")
    months["when"] = pd.to_datetime(dict(year=months["Year"], month=months["Month"], day=1))
    fig = go.Figure()
    fig.add_bar(x=months["when"], y=months["area_ha"] / 1e6, name="Actually burned",
                marker_color=sh.CAUSE_COLOURS["Unknown"], hovertemplate="%{x|%b %Y}: %{y:,.2f}M ha<extra></extra>")
    fig.add_scatter(x=months["when"], y=months["predicted_ha_without_count"] / 1e6, name="The 2024 model, forecasting",
                    mode="lines+markers", line=dict(color=sh.CAUSE_COLOURS["Lightning"], width=2),
                    hovertemplate="%{x|%b %Y}: %{y:,.2f}M ha<extra></extra>")
    sh.show(fig, height=330, title="Hectares burned each month, 2020–2024, against the rebuilt 2024 model",
            yaxis=dict(title="Million ha"))
    st.caption("2023 is the year the monthly model cannot see coming: 17.6 million hectares, more than the other "
               "four test years together, and nothing in a month's average national weather says so in advance.")

st.markdown(
    f"""
### Kept for the record

The original notebooks, Power BI file, deck and source tables are in the repository, untouched, under
[`legacy/2024-hackathon`]({sh.REPO_URL}/tree/main/legacy/2024-hackathon).
"""
)
sh.sources_footer()
