"""Seven-day risk: where new fires are most likely to start."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import shared as sh

sh.page_style()
st.title("Where new fires are likely this week")

live = sh.live_forecast()


def rebuild_panel(expanded: bool) -> None:
    with st.expander("Rebuild from live data now", expanded=expanded):
        st.write("Reads the last 30 days of CWFIS station observations, Open-Meteo's forecast for every cell and "
                 f"the latest satellite hotspots, then scores all {sh.CELLS_IN_GRID} cells. It takes about three minutes; the "
                 "scheduled run does the same twice a day.")
        if st.button("Rebuild the forecast", type="primary"):
            with st.status("Rebuilding the forecast", expanded=True) as status:
                try:
                    result = sh.rebuild_live(progress=st.write)
                except Exception as exc:  # network, rate limits, CWFIS gaps: show them, do not crash the page
                    status.update(label="Rebuild failed", state="error")
                    st.error(f"{type(exc).__name__}: {exc}")
                else:
                    status.update(label=f"Rebuilt in {result['seconds']:.0f} s", state="complete")
                    st.rerun()


if live is None:
    st.error("No forecast is available yet. Build one from live data below.")
    rebuild_panel(expanded=True)
    sh.sources_footer()
    st.stop()

meta = live.meta
if not meta.get("in_season"):
    sh.kicker(f"Checked {sh.age(meta['generated_at'])}.")
    st.info("The model covers the fire season, April to October, when 98% of Canada's recorded fires start. "
            "Outside it the forecast pauses rather than stretch the model to months it never saw. The 2026 "
            "season check and the 2020–2024 replay show how it performs.")
    rebuild_panel(expanded=False)
    sh.sources_footer()
    st.stop()

sh.kicker(
    f"Built {sh.age(meta['generated_at'])} from {meta['stations_used']:,} CWFIS weather stations reporting through "
    f"{sh.day_label(meta['as_of'])}, carried forward with Open-Meteo's forecast to {sh.day_label(meta['last_day'])}. "
    f"Source: {sh.ORIGINS[live.origin]}.")
if sh.hours_since(meta["generated_at"]) > sh.STALE_HOURS:
    st.warning(f"This forecast was built {sh.age(meta['generated_at'])}, so the scheduled refresh has been missed. "
               "Rebuild it from live data at the bottom of the page.")

forecast = live.forecast.copy()
forecast["date"] = pd.to_datetime(forecast["date"])
forecast["vs_normal"] = forecast["risk"] / forecast["clim_month_rate"].clip(lower=0.001)
days = [pd.Timestamp(d) for d in sorted(forecast["date"].unique())]
leads = {pd.Timestamp(d): int(lead) for d, lead in forecast.drop_duplicates("date")[["date", "lead_days"]].itertuples(index=False)}

left, right = st.columns([3, 2], vertical_alignment="bottom")
with left:
    day = st.select_slider("Day", options=days, value=days[min(1, len(days) - 1)],
                           format_func=lambda d: sh.day_label(d, leads[pd.Timestamp(d)]))
with right:
    scale_key = st.segmented_control("Colour cells by", options=list(sh.SCALES), default="risk", required=True,
                                     format_func=lambda k: sh.SCALES[k].title)
scale = sh.SCALES[scale_key or "risk"]

today = forecast[forecast["date"] == day].copy()
expected = float(today["risk"].sum())
high = int((today["risk"] >= 0.07).sum())
by_province = today.groupby("province")["risk"].sum().sort_values(ascending=False)
bands = sh.band_table("risk")
top_bands = bands[bands["band"] >= 3] if bands is not None else None

cols = st.columns(4)
cols[0].metric("Cells expected to report a fire", sh.number(expected), border=True,
               help=f"The sum of every cell's probability: how many of the {sh.CELLS_IN_GRID} cells the model expects to "
                    "report at least one new fire that day.")
cols[1].metric("Cells at 7% risk or more", f"{high}", border=True,
               help=None if top_bands is None else
               f"The top two bands. In the 2020–2024 replay they covered {sh.pct(top_bands['cell_day_share'].sum(), 0)} "
               f"of cell-days and held {sh.pct(top_bands['fire_share'].sum(), 0)} of the fires that started.")
cols[2].metric("Most fires expected in", sh.PROVINCE_SHORT.get(by_province.index[0], by_province.index[0]), border=True,
               help=" · ".join(f"{name}: {value:.1f}" for name, value in by_province.head(4).items()))
cols[3].metric("Satellite hotspots, last 48 h", f"{len(live.hotspots):,}", border=True,
               help="Fire detections in Canada from CWFIS's satellite hotspot feed. Context only: they are not a "
                    "model input.")

show_spots = st.toggle("Show satellite hotspots from the last 48 hours", value=True)
view = today.assign(
    risk_text=today["risk"].map(lambda v: sh.pct(v, 1)),
    large_text=today["large_risk"].map(lambda v: sh.pct(v, 2)),
    normal_text=today["vs_normal"].map(lambda v: f"{v:.1f}×"),
    fwi_text=today["fwi"].map(lambda v: sh.number(v)),
    weather_text=[f"{t:.0f}°C, RH {h:.0f}%, wind {w:.0f} km/h, rain {p:.1f} mm" if np.isfinite(t) else "no station data"
                  for t, h, w, p in zip(today["temp"], today["rh"], today["ws"], today["precip"])],
)
sh.draw_map(view, scale,
            "<b>{province}</b> · cell {cell_id}<br/>New fire: <b>{risk_text}</b> ({normal_text} normal)<br/>"
            "Over 200 ha: {large_text}<br/>FWI {fwi_text} · {weather_text}",
            points=live.hotspots if show_spots else None)
notes, caption = sh.band_notes(scale)
sh.legend(scale, notes)
st.caption(caption + (" Pale dots: satellite hotspots." if show_spots else ""))

st.subheader("The week by province")
week = forecast.groupby(["province", "date"])["risk"].sum().unstack("date")
week = week.loc[week.sum(axis=1).sort_values(ascending=True).index]
labels = [sh.day_label(d, leads[pd.Timestamp(d)]) for d in week.columns]
fig = go.Figure(go.Heatmap(
    z=week.to_numpy(), x=labels, y=[sh.PROVINCE_SHORT.get(p, p) for p in week.index],
    text=np.vectorize(lambda v: f"{v:.1f}")(week.to_numpy()), texttemplate="%{text}",
    colorscale=[[0, sh.SURFACE], [0.35, "#7A4A22"], [0.7, sh.MODEL], [1, "#FDE3C4"]],
    hovertemplate="%{y}, %{x}<br>%{z:.1f} cells expected to report a fire<extra></extra>", showscale=False))
sh.show(fig, height=60 + 34 * len(week), xaxis=dict(side="top"))
st.caption("Each number is the sum of that province's cell probabilities: the cells the model expects to report at "
           "least one new fire. Forecast weather gets less certain with every day ahead.")

st.subheader(f"Riskiest cells, {sh.day_label(day)}")
top = today.nlargest(15, "risk")[["province", "lat", "lon", "risk", "large_risk", "vs_normal", "fwi", "isi", "bui",
                                  "temp", "rh", "ws", "precip"]]
st.dataframe(top, hide_index=True, column_config={
    "province": "Province",
    "lat": st.column_config.NumberColumn("Lat", format="%.1f"),
    "lon": st.column_config.NumberColumn("Lon", format="%.1f"),
    "risk": st.column_config.ProgressColumn("New fire", format="percent", min_value=0.0,
                                            max_value=max(0.3, float(top["risk"].max()))),
    "large_risk": st.column_config.NumberColumn("Over 200 ha", format="percent"),
    "vs_normal": st.column_config.NumberColumn("vs normal", format="%.1f×"),
    "fwi": st.column_config.NumberColumn("FWI", format="%.0f"),
    "isi": st.column_config.NumberColumn("ISI", format="%.1f"),
    "bui": st.column_config.NumberColumn("BUI", format="%.0f"),
    "temp": st.column_config.NumberColumn("Temp °C", format="%.0f"),
    "rh": st.column_config.NumberColumn("RH %", format="%.0f"),
    "ws": st.column_config.NumberColumn("Wind km/h", format="%.0f"),
    "precip": st.column_config.NumberColumn("Rain mm", format="%.1f"),
})

with st.expander("How this forecast is built, and what it cannot see"):
    st.markdown(
        """
**Built.** Canada is cut into 1° cells, keeping the CELLS_IN_GRID that recorded at least ten fires in 2000–2016. For each
cell the last month of noon weather and FWI System codes are interpolated from the nearest CWFIS stations
(up to four within 200 km). Open-Meteo's forecast for the cell centre then steps the Fine Fuel, Duff and Drought
codes forward one day at a time, with the same FWI implementation that reproduces CWFIS's published codes. Each
day's row is scored by two gradient-boosted models, one for any new fire and one for a fire that grows past
200 ha, both calibrated on 2017–2019.

**Cannot see.** Lightning, which starts most of the area burned, is not an input: a storm over a low-risk cell
is invisible until fires are reported. Nor are people, fuel breaks, suppression, or the state of the vegetation
beyond what the moisture codes carry. Forecast weather adds error with each day ahead, and a 1° cell is
around 110 km by 70 km, so the map says where to look, not where a fire will be.
""".replace("CELLS_IN_GRID", str(sh.CELLS_IN_GRID))
    )

rebuild_panel(expanded=False)
sh.sources_footer()
