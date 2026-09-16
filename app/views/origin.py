"""Where this started: the hackathon project, and what changed."""

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
| **Weather** | Daily weather records joined to area burned | Noon observations from about 2,850 CWFIS stations, plus a live forecast |
| **Fire science** | None | The Canadian FWI System, implemented and checked against CWFIS's own codes |
| **Unit** | One weather record | One 1° cell on one day, 771 cells |
| **Validation** | Random 80/20 split | Seasons held out in time, two baselines, and the 2026 season against satellite detections |
| **Delivery** | A .pbix file and a slide deck | This app, rebuilt twice a day from public data |
"""
)

st.markdown(
    f"""
### Kept for the record

The original notebooks, Power BI file, deck and source tables are in the repository, untouched, under
[`legacy/2024-hackathon`]({sh.REPO_URL}/tree/main/legacy/2024-hackathon).
"""
)
sh.sources_footer()
