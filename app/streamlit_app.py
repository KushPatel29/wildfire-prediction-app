"""Canada wildfire risk - entry point and navigation."""

import streamlit as st

st.set_page_config(page_title="Canada Wildfire Risk", page_icon="🔥", layout="wide")

pages = {
    "Forecast": [
        st.Page("views/forecast.py", title="Seven-day risk", icon=":material/local_fire_department:", default=True),
    ],
    "Evidence": [
        st.Page("views/season.py", title="2026 season check", icon=":material/satellite_alt:"),
        st.Page("views/replay.py", title="Replay 2020–2024", icon=":material/replay:"),
        st.Page("views/model_card.py", title="Model card", icon=":material/fact_check:"),
    ],
    "Context": [
        st.Page("views/history.py", title="Fires since 2000", icon=":material/bar_chart:"),
        st.Page("views/origin.py", title="From the hackathon", icon=":material/emoji_events:"),
    ],
}
st.navigation(pages).run()
