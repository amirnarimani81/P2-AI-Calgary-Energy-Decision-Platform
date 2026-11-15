import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.seasonal import STL

import calgary_utils as cu

cu.page_header("Seasonal & Time-Series Patterns", "Section 2 · Explore", """
When is energy used? Daily, weekly, monthly and seasonal patterns, year-over-year trend and a
**trend / seasonal / residual decomposition** of daily consumption.
""")
raw, mapping, data = cu.require_data()
u = cu.unit_label(mapping)
daily = cu.to_daily(data)
hourly_data = cu.freq_label(data) == "hourly"
results = {}

# -------------------------------------------------------
# SEASONAL KPIs
# -------------------------------------------------------
st.subheader("Seasons")
order = ["Winter", "Spring", "Summer", "Fall"]
season = daily.groupby("season").agg(avg_per_day=("consumption", "mean"), peak_day=("consumption", "max"),
                                     days=("consumption", "size")).reindex(order).dropna()
if "temperature" in daily:
    season["avg_temp_c"] = daily.groupby("season")["temperature"].mean().reindex(season.index)
if "price" in data:
    season["avg_price"] = data.groupby("season")["price"].mean().reindex(season.index)
season["vs_annual_avg_%"] = (season["avg_per_day"] / daily["consumption"].mean() - 1) * 100
st.dataframe(season.round(1))
results["seasons"] = season.round(2).to_dict("index")

# -------------------------------------------------------
# DAILY & WEEKLY PROFILES
# -------------------------------------------------------
c1, c2 = st.columns(2)
if hourly_data:
    prof = data.groupby(["season", "hour"])["consumption"].mean().reset_index()
    c1.plotly_chart(px.line(prof, x="hour", y="consumption", color="season", markers=True,
                            category_orders={"season": order},
                            title=f"Average daily profile by season ({u}/reading)"))
    heat = data.pivot_table(index="weekday", columns="hour", values="consumption", aggfunc="mean")
    heat.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(heat)]
    c2.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Viridis",
                              title="Weekday × hour heatmap"))
    hp = data.groupby("hour")["consumption"].mean()
    results["peak_hour"], results["lowest_hour"] = int(hp.idxmax()), int(hp.idxmin())
    results["daily_swing_pct"] = round(float((hp.max() / hp.min() - 1) * 100), 1)
    if results["daily_swing_pct"] < 10:
        st.warning(f"The average daily profile varies only {results['daily_swing_pct']}% between hours. Real "
                   "buildings and grids usually swing 20-60% between night and day; check the source data.")
else:
    wk = daily.groupby("weekday")["consumption"].mean()
    c1.plotly_chart(px.bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(wk)], y=wk.values,
                           title=f"Average by weekday ({u}/day)"))

wd = daily.groupby("is_weekend")["consumption"].mean()
if len(wd) == 2:
    results["weekend_vs_weekday_pct"] = round(float((wd[1] / wd[0] - 1) * 100), 2)
    st.metric("Weekend vs weekday consumption", f"{results['weekend_vs_weekday_pct']:+.1f}%")

# -------------------------------------------------------
# TREND
# -------------------------------------------------------
st.subheader("Trend")
c1, c2 = st.columns(2)
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily["consumption"], name="Daily", line=dict(width=0.6), opacity=0.5))
fig.add_trace(go.Scatter(x=daily.index, y=daily["consumption"].rolling(30, min_periods=10).mean(),
                         name="30-day average", line=dict(width=2)))
fig.update_layout(title=f"Daily consumption ({u}/day)", height=380)
c1.plotly_chart(fig)
my = daily.groupby(["year", "month"])["consumption"].mean().reset_index()
c2.plotly_chart(px.line(my, x="month", y="consumption", color=my["year"].astype(str), markers=True,
                        title="Monthly average by year (year-over-year)"))
yearly = daily.groupby("year")["consumption"].agg(["mean", "size"])
yearly = yearly[yearly["size"] >= 200]
if len(yearly) >= 2:
    yoy = (yearly["mean"].pct_change() * 100).round(2).dropna()
    st.caption("Year-over-year change in average daily consumption (years with ≥200 days): "
               + ", ".join(f"{y}: {v:+.1f}%" for y, v in yoy.items()))
    results["yoy_change_pct"] = yoy.to_dict()

# -------------------------------------------------------
# DECOMPOSITION
# -------------------------------------------------------
st.subheader("Decomposition of daily consumption (STL)")
period = st.radio("Seasonal period", [7, 365], format_func=lambda p: "Weekly (7 days)" if p == 7 else "Annual (365 days)",
                  horizontal=True)
series = daily["consumption"].asfreq("D").interpolate(limit=7).dropna()
if len(series) < 2 * period + 10:
    st.info("Not enough continuous daily data for this period.")
else:
    stl = STL(series, period=period, robust=True).fit()
    comp = pd.DataFrame({"observed": series, "trend": stl.trend, "seasonal": stl.seasonal, "residual": stl.resid})
    fig = px.line(comp.reset_index().melt(id_vars="datetime"), x="datetime", y="value", facet_row="variable",
                  height=700)
    fig.update_yaxes(matches=None)
    st.plotly_chart(fig)
    strength = max(0.0, 1 - comp["residual"].var() / (comp["seasonal"] + comp["residual"]).var())
    trend_strength = max(0.0, 1 - comp["residual"].var() / (comp["trend"] + comp["residual"]).var())
    k = st.columns(2)
    k[0].metric("Seasonal strength (0-1)", f"{strength:.2f}")
    k[1].metric("Trend strength (0-1)", f"{trend_strength:.2f}")
    results[f"stl_period_{period}"] = {"seasonal_strength": round(strength, 3), "trend_strength": round(trend_strength, 3)}

if "price" in data and hourly_data:
    st.subheader("Electricity price by hour and season")
    pp = data.groupby(["season", "hour"])["price"].mean().reset_index()
    st.plotly_chart(px.line(pp, x="hour", y="price", color="season", category_orders={"season": order}, markers=True))
    hp = data.groupby("hour")["price"].mean()
    results["most_expensive_hours"] = hp.nlargest(3).index.tolist()
    results["cheapest_hours"] = hp.nsmallest(3).index.tolist()

cu.ai_interpretation("Seasonal & time-series patterns", results,
                     "1) Seasonal and daily usage patterns 2) Trend over the years 3) Peak periods and "
                     "their cost implications 4) Operational opportunities (scheduling, setbacks, load shifting).")
