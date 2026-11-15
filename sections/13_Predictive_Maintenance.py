import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import calgary_utils as cu

cu.page_header("Predictive Maintenance", "Section 5 · Facility Decisions", """
Energy use is a condition-monitoring signal: degrading or faulty equipment (fouled coils, worn compressors,
stuck dampers, control faults, equipment left on) shows up as **consumption above what weather and
schedule explain**.

1. **Expected-consumption model** trained on a healthy reference period
2. **Condition indicators**: efficiency drift, anomalies (Isolation Forest), overload hours, ramp stress, base load
3. **Daily health index (0-100)**, risk level and **days until the high-risk threshold**
4. **Next-day risk classifier** (compared with a persistence benchmark) and recommended actions
""")
raw, mapping, data = cu.require_data()
u = cu.unit_label(mapping)
if cu.freq_label(data) == "monthly":
    st.error("Needs hourly or daily data.")
    st.stop()
if mapping.get("clean", "None") != "None":
    st.warning("Outlier cleaning is ON in the sidebar: spikes this page should detect may be removed. "
               "Set 'Consumption outliers' to None for maintenance analysis.")
spd = cu.steps_per_day(data)
hourly = spd >= 4
ppd = max(spd, 1)

c1, c2, c3, c4 = st.columns(4)
ref_pct = c1.slider("Healthy reference period (first % of data)", 20, 80, 50)
capacity = c2.number_input(f"Rated capacity per reading ({u})", value=round(float(data["consumption"].quantile(0.99) * 1.1), 1),
                           help="Default = 110% of the 99th percentile; use the real service / equipment rating.")
contamination = c3.slider("Expected anomaly share", 0.005, 0.05, 0.01, 0.005)
label_options = ["(none: health-index proxy)"] + [c for c in raw.columns if raw[c].dropna().nunique() == 2]
label_col = c4.selectbox("Real maintenance / failure label (0/1)", label_options)

# 1. expected consumption --------------------------------------------------
feats = [f for f in cu.base_features(data, include_price=False)]
cut = data.index[int(len(data) * ref_pct / 100)]
ref = data[data.index < cut]


@st.cache_resource(show_spinner="Training expected-consumption model...")
def expected_model(X, y):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42).fit(X, y)


m = expected_model(ref[feats].astype(float), ref["consumption"].astype(float))
d = data.copy()
d["expected"] = m.predict(d[feats].astype(float))
d["residual_pct"] = (d["consumption"] - d["expected"]) / d["expected"].clip(lower=1e-6) * 100
d["ramp"] = d["consumption"].diff().abs() / capacity

# 2. anomalies -------------------------------------------------------------
iso_cols = ["residual_pct", "ramp"] + (["hour"] if hourly else [])
iso = IsolationForest(contamination=contamination, random_state=42).fit(d.loc[d.index < cut, iso_cols].fillna(0))
d["anomaly"] = iso.predict(d[iso_cols].fillna(0)) == -1

# 3. daily health index ----------------------------------------------------
day = d.resample("D").agg({"consumption": "sum", "expected": "sum", "anomaly": "sum", "ramp": "max"})
day = day[day["expected"] > 0]
day["drift_pct"] = (day["consumption"] / day["expected"] - 1) * 100
day["drift_7d"] = day["drift_pct"].rolling(7, min_periods=3).mean()
day["overload"] = (d["consumption"] > 0.9 * capacity).resample("D").sum().reindex(day.index)
if hourly:
    night = d[d["hour"].between(0, 4)]["consumption"].resample("D").mean()
    day["base_load_ratio"] = (night / d["consumption"].resample("D").mean()).reindex(day.index)
pen = (np.clip(day["drift_7d"].fillna(0) / 15, 0, 1) * 40 + np.clip(day["anomaly"] / (0.25 * ppd), 0, 1) * 30
       + np.clip(day["overload"].fillna(0) / (0.2 * ppd), 0, 1) * 20 + np.clip(day["ramp"].fillna(0) / 0.5, 0, 1) * 10)
day["health_index"] = (100 - pen).round(1)
day["risk"] = pd.cut(day["health_index"], [-1, 60, 80, 101], labels=["High", "Medium", "Low"])

latest = day.iloc[-1]
trend = linregress(np.arange(min(60, len(day))), day["drift_pct"].tail(60).fillna(0))
k = st.columns(5)
k[0].metric("Latest health index", f"{latest['health_index']:.0f}/100", str(latest["risk"]), delta_color="off")
k[1].metric("7-day efficiency drift", f"{latest['drift_7d']:+.1f}%")
k[2].metric("Drift trend (60 d)", f"{trend.slope * 30:+.2f}%/month",
            "significant" if trend.pvalue < 0.05 else "not significant", delta_color="off")
k[3].metric("Anomalies (last 30 d)", int(day["anomaly"].tail(30).sum()))
k[4].metric("High-risk days (last 30 d)", int((day["risk"].tail(30) == "High").sum()))

fig = go.Figure()
fig.add_trace(go.Scatter(x=day.index, y=day["health_index"], name="Health index", line=dict(width=1)))
fig.add_hrect(y0=0, y1=60, opacity=0.08, line_width=0, fillcolor="red")
fig.add_hrect(y0=60, y1=80, opacity=0.08, line_width=0, fillcolor="orange")
fig.add_vline(x=cut, line_dash="dash")
fig.update_layout(title="Daily health index (dashed = end of healthy reference)", yaxis_range=[0, 102], height=360)
st.plotly_chart(fig)
c1, c2 = st.columns(2)
c1.plotly_chart(px.line(day[["drift_pct", "drift_7d"]], title="Efficiency drift: actual vs expected (%)"))
view = d.tail(int(ppd * 60))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=view.index, y=view["consumption"], name="Actual", line=dict(width=1)))
fig2.add_trace(go.Scatter(x=view.index, y=view["expected"], name="Expected", line=dict(width=1)))
an = view[view["anomaly"]]
fig2.add_trace(go.Scatter(x=an.index, y=an["consumption"], mode="markers", name="Anomaly", marker=dict(size=7, symbol="x")))
fig2.update_layout(title="Last 60 days: actual vs expected", height=380)
c2.plotly_chart(fig2)

# 4a. days to threshold ----------------------------------------------------
st.subheader("Degradation projection")
hi = day["health_index"].rolling(7, min_periods=3).mean().dropna().tail(60)
proj = linregress(np.arange(len(hi)), hi.values)
current = float(hi.iloc[-1])
if current < 60:
    days_left = 0
    st.error(f"Already in the High-risk zone (7-day health {current:.0f}): maintenance is due now.")
elif proj.slope < 0 and proj.pvalue < 0.05:
    days_left = int((current - 60) / -proj.slope)
    st.warning(f"Health falling {proj.slope * 7:.1f} points/week → High-risk threshold in about **{days_left} days** "
               f"(~{day.index[-1] + pd.Timedelta(days=days_left):%Y-%m-%d}).")
else:
    days_left = None
    st.success(f"No significant downward trend in the last 60 days (7-day health {current:.0f}).")

# 4b. classifier -----------------------------------------------------------
st.subheader("Next-day risk classifier")
X = day[["drift_pct", "drift_7d", "anomaly", "ramp", "overload", "health_index"] + (["base_load_ratio"] if hourly else [])].copy()
X["drift_slope_7d"] = day["drift_pct"].diff(7) / 7
X["anomaly_7d"] = day["anomaly"].rolling(7, min_periods=1).sum()
X = X.fillna(0)
if label_col.startswith("(none"):
    label = (day["risk"] == "High").astype(int)
    st.caption("No real maintenance records mapped: target = health-index rule (High risk tomorrow). This is an "
               "early-warning model; map a real work-order / failure column for a true failure model.")
else:
    lab = raw.copy()
    lab.index = cu._timestamps(lab, mapping)
    vals = sorted(lab[label_col].dropna().unique())
    label = (lab[label_col] == vals[-1]).astype(int).groupby(level=0).max().resample("D").max().reindex(day.index).fillna(0)
    st.caption(f"Target: real events from `{label_col}` (positive = {vals[-1]}).")
y_next = label.shift(-1)
Xm, ym = X.iloc[:-1], y_next.iloc[:-1].astype(int)
split = int(len(Xm) * 0.7)
X_tr, X_te, y_tr, y_te = Xm.iloc[:split], Xm.iloc[split:], ym.iloc[:split], ym.iloc[split:]
auc_model = auc_persist = tomorrow = None
if y_tr.nunique() < 2:
    st.warning("Too few risk days in the training period to train a classifier.")
else:
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(pd.DataFrame(classification_report(y_te, pred, output_dict=True, zero_division=0)).T.round(3))
        if y_te.nunique() == 2:
            auc_model, auc_persist = roc_auc_score(y_te, proba), roc_auc_score(y_te, -X_te["health_index"])
            a, b = st.columns(2)
            a.metric("ROC AUC classifier", f"{auc_model:.3f}")
            b.metric("ROC AUC persistence", f"{auc_persist:.3f}")
            if auc_model < auc_persist:
                st.warning("The classifier does not beat 'tomorrow looks like today'; treat its probability "
                           "with caution. The degradation projection above does not depend on it.")
    with c2:
        st.plotly_chart(px.imshow(confusion_matrix(y_te, pred, labels=[0, 1]), text_auto=True,
                                  x=["Pred OK", "Pred risk"], y=["Actual OK", "Actual risk"], title="Confusion matrix"))
    tomorrow = float(clf.predict_proba(X.iloc[[-1]])[:, 1][0])
    (st.error if tomorrow >= 0.5 else st.warning if tomorrow >= 0.25 else st.success)(
        f"Probability of a maintenance-risk day on {day.index[-1] + pd.Timedelta(days=1):%Y-%m-%d}: **{tomorrow:.0%}**")

# 5. actions ---------------------------------------------------------------
st.subheader("Recommended maintenance actions")
recent, acts = day.tail(14), []
if recent["drift_7d"].iloc[-1] > 5:
    acts.append(("High", f"Consumption {recent['drift_7d'].iloc[-1]:.1f}% above weather-normalized expectation",
                 "Inspect HVAC: filters, coils, refrigerant, boiler efficiency, economizer / dampers"))
if trend.pvalue < 0.05 and trend.slope > 0:
    acts.append(("Medium", f"Steady efficiency decline ({trend.slope * 30:+.2f}%/month)",
                 "Schedule preventive maintenance before the winter peak; trend motor and pump currents"))
if recent["anomaly"].sum() > 0.1 * ppd:
    acts.append(("Medium", f"{int(recent['anomaly'].sum())} anomalous readings in 14 days",
                 "Review BAS schedules and alarms; look for equipment left on or control faults"))
if recent["overload"].sum() > 0:
    acts.append(("High", f"{int(recent['overload'].sum())} readings above 90% of rated capacity",
                 "Thermal stress on transformer / feeder: IR scan, check loading, peak management"))
if hourly and recent["base_load_ratio"].mean() > day["base_load_ratio"].head(max(len(ref) // int(ppd), 1)).mean() * 1.1:
    acts.append(("Low", "Night-time base load up >10% vs reference", "Audit after-hours lighting, ventilation, process, plug loads"))
if not acts:
    acts.append(("Low", "No significant degradation signals", "Continue routine preventive maintenance"))
st.dataframe(pd.DataFrame(acts, columns=["Priority", "Signal", "Recommended action"]), hide_index=True)
st.subheader("Highest-risk days")
st.dataframe(day.sort_values("health_index").head(15)[["consumption", "expected", "drift_pct", "anomaly", "overload",
                                                        "health_index", "risk"]].round(2))

cu.ai_interpretation("Predictive maintenance", {
    "latest_health_index": latest["health_index"], "latest_risk": str(latest["risk"]),
    "drift_7d_pct": latest["drift_7d"], "drift_trend_pct_per_month": trend.slope * 30, "trend_p": trend.pvalue,
    "anomalies_last_30d": int(day["anomaly"].tail(30).sum()), "high_risk_days_last_30d": int((day["risk"].tail(30) == "High").sum()),
    "days_until_high_risk": days_left, "classifier_auc": auc_model, "persistence_auc": auc_persist,
    "tomorrow_risk_probability": tomorrow, "label_source": label_col, "actions": acts},
    "1) Current equipment condition 2) Likely root causes 3) Prioritised 30-day maintenance plan (Calgary "
    "winter context) 4) Cost of inaction (energy waste, failure risk).")
