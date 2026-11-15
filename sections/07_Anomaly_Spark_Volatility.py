import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import calgary_utils as cu

cu.page_header("Anomalies, Spark Spread & Volatility", "Section 3 · Market & Risk", """
Market risk for a Calgary energy buyer or generator:

- **Price anomalies**: hours far from the usual price for that hour and season (robust z-score)
- **Spark spread**: electricity price minus gas fuel cost (gas price × heat rate), i.e. whether gas-fired
  generation or on-site CHP is profitable
- **Volatility & price risk**: coefficient of variation, rolling volatility, upper-tail **Value-at-Risk**
  and expected shortfall (what a buyer pays in the worst hours)
""")
raw, mapping, data = cu.require_data()
if "price" not in data:
    st.error("Map an electricity price column in the sidebar to use this page.")
    st.stop()
hourly = cu.freq_label(data) == "hourly"
results = {}

tab1, tab2, tab3, tab4 = st.tabs(["Price anomalies", "Spark spread", "Volatility & price risk", "Buy / sell timing"])

# -------------------------------------------------------
# ANOMALIES
# -------------------------------------------------------
with tab1:
    k_thr = st.slider("Anomaly threshold (robust z)", 2.5, 8.0, 4.0, 0.5)
    key = ["season", "hour"] if hourly else ["season"]
    med = data.groupby(key)["price"].transform("median")
    resid = data["price"] - med
    mad = (resid - resid.median()).abs().median()
    z = 0.6745 * (resid - resid.median()) / mad if mad else resid * 0
    anomalies = data.assign(expected=med, z=z)[z.abs() > k_thr]
    k = st.columns(4)
    k[0].metric("Anomalous readings", f"{len(anomalies):,}", f"{len(anomalies) / len(data) * 100:.2f}% of data",
                delta_color="off")
    k[1].metric("High-price anomalies", int((anomalies["z"] > 0).sum()))
    k[2].metric("Low-price anomalies", int((anomalies["z"] < 0).sum()))
    k[3].metric("Max anomalous price", f"{anomalies['price'].max():.1f}" if len(anomalies) else "-")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["price"], name="Price", line=dict(width=0.5)))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["price"], mode="markers", name="Anomaly",
                             marker=dict(size=6, symbol="x")))
    fig.update_layout(height=380, yaxis_title="$/MWh")
    st.plotly_chart(fig)
    st.caption("Expected price = median for the same season" + (" and hour" if hourly else "") +
               ", so normal evening peaks are not flagged as anomalies.")
    if len(anomalies):
        st.dataframe(anomalies[["price", "expected", "z"]].sort_values("z", key=abs, ascending=False).head(20).round(2))
        if hourly:
            st.plotly_chart(px.histogram(anomalies, x="hour", nbins=24, title="When anomalies happen (hour)"))
    results["anomalies"] = {"count": len(anomalies), "pct": round(len(anomalies) / len(data) * 100, 3),
                            "threshold_z": k_thr,
                            "common_hours": anomalies["hour"].value_counts().head(3).index.tolist() if hourly else []}

# -------------------------------------------------------
# SPARK SPREAD
# -------------------------------------------------------
with tab2:
    if "gas_price" not in data:
        st.info("Map a gas price column ($/GJ) in the sidebar to calculate the spark spread.")
    else:
        c1, c2 = st.columns(2)
        heat_rate = c1.number_input("Heat rate (GJ/MWh)", value=7.0, step=0.5,
                                    help="~6.5-7.5 for a combined-cycle plant, ~9-11 for a peaker or small CHP.")
        vom = c2.number_input("Variable O&M + carbon cost ($/MWh)", value=0.0, step=5.0,
                              help="Add carbon compliance and variable O&M for a net margin.")
        ss = data["price"] - data["gas_price"] * heat_rate - vom
        k = st.columns(4)
        k[0].metric("Average spark spread", f"{ss.mean():.2f} $/MWh")
        k[1].metric("Profitable readings", f"{(ss > 0).mean() * 100:.1f}%")
        k[2].metric("Min / max", f"{ss.min():.0f} / {ss.max():.0f}")
        k[3].metric("Avg fuel cost", f"{(data['gas_price'] * heat_rate).mean():.2f} $/MWh")
        monthly = ss.resample("MS").mean()
        st.plotly_chart(px.bar(monthly, title="Monthly average spark spread ($/MWh)", labels={"value": "$/MWh"}))
        if hourly:
            hs = ss.groupby(data["hour"]).mean()
            st.plotly_chart(px.bar(hs, title="Average spark spread by hour"))
            st.caption(f"Best hours to run gas generation / CHP: {', '.join(map(str, hs.nlargest(4).index))}. "
                       f"Worst: {', '.join(map(str, hs.nsmallest(4).index))}.")
        results["spark_spread"] = {"heat_rate": heat_rate, "avg": round(float(ss.mean()), 2),
                                   "profitable_pct": round(float((ss > 0).mean() * 100), 2),
                                   "min": round(float(ss.min()), 2), "max": round(float(ss.max()), 2)}

# -------------------------------------------------------
# VOLATILITY & PRICE RISK
# -------------------------------------------------------
with tab3:
    p = data["price"]
    cv = p.std() / p.mean() * 100
    level = "LOW" if cv < 15 else "MODERATE" if cv < 30 else "HIGH"
    var95, var99 = p.quantile(0.95), p.quantile(0.99)
    es95 = p[p >= var95].mean()
    k = st.columns(5)
    k[0].metric("Price CV", f"{cv:.1f}%", level, delta_color="off")
    k[1].metric("VaR 95% (price exceeded 5% of time)", f"{var95:.1f}")
    k[2].metric("VaR 99%", f"{var99:.1f}")
    k[3].metric("Expected shortfall 95%", f"{es95:.1f}", help="Average price in the worst 5% of readings")
    k[4].metric("Mean price", f"{p.mean():.1f}")
    st.caption("For a buyer, risk is high prices, so VaR here is the **upper** tail. Historical "
               "percentiles are used (no normality assumption).")
    win = "7D" if hourly else 30
    vol = p.rolling(win, min_periods=5).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol.index, y=vol, name="Rolling volatility", line=dict(width=1)))
    fig.update_layout(title=f"Rolling price volatility ({'7-day' if hourly else '30-reading'} std)", height=330)
    st.plotly_chart(fig)
    if "consumption" in data:
        cost = data["consumption"] * cu.to_mwh_factor(mapping) * p
        high = p >= var95
        share = cost[high].sum() / cost.sum() * 100
        st.metric("Share of energy cost incurred in the top-5% price readings", f"{share:.1f}%")
        results["cost_share_top5pct_hours"] = round(float(share), 2)
    results["volatility"] = {"cv_pct": round(float(cv), 2), "risk_level": level, "var95": round(float(var95), 2),
                             "var99": round(float(var99), 2), "expected_shortfall95": round(float(es95), 2)}

# -------------------------------------------------------
# BUY / SELL TIMING
# -------------------------------------------------------
with tab4:
    if not hourly:
        st.info("Needs hourly data.")
    else:
        hp = data.groupby("hour")["price"].mean()
        wk = data.groupby("is_weekend")["price"].mean()
        best_buy, best_sell = int(hp.idxmin()), int(hp.idxmax())
        k = st.columns(4)
        k[0].metric("Cheapest hour", f"{best_buy}:00", f"{hp.min():.1f} $/MWh", delta_color="off")
        k[1].metric("Most expensive hour", f"{best_sell}:00", f"{hp.max():.1f} $/MWh", delta_color="off")
        k[2].metric("Peak / off-peak spread", f"{hp.max() - hp.min():.1f} $/MWh")
        if len(wk) == 2:
            k[3].metric("Weekend discount", f"{(1 - wk[1] / wk[0]) * 100:.1f}%")
        st.plotly_chart(px.bar(hp, title="Average price by hour"))
        st.caption("A large peak/off-peak spread favours load shifting and storage (pages 14 and 15).")
        results["timing"] = {"cheapest_hour": best_buy, "most_expensive_hour": best_sell,
                             "spread": round(float(hp.max() - hp.min()), 2)}

cu.ai_interpretation("Market risk (anomalies, spark spread, volatility)", results,
                     "1) Market risk level and its cost impact 2) Spark spread / generation economics 3) "
                     "Procurement and hedging implications 4) Operational actions (load shifting, storage).")
