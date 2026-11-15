import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import calgary_utils as cu

cu.page_header("Data Preview & Quality", "Section 1 · Data", """
What is in the dataset, how complete and regular it is, and whether it is plausible for a Calgary
energy system. Ends with a **data quality score** and an AI data-quality review.
""")
raw, mapping, data = cu.require_data()
view = cu.raw_view(raw, mapping, drop_engineered=False)

# -------------------------------------------------------
# OVERVIEW
# -------------------------------------------------------
k = st.columns(5)
k[0].metric("Rows (raw)", f"{len(raw):,}")
k[1].metric("Columns", raw.shape[1])
k[2].metric("Duplicate rows", int(raw.duplicated().sum()))
k[3].metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
k[4].metric("Rows used for analysis", f"{len(data):,}")

st.subheader("Preview")
st.dataframe(raw.head(20))

st.subheader("Columns")
prof = pd.DataFrame({"type": raw.dtypes.astype(str), "missing %": (raw.isna().mean() * 100).round(2),
                     "unique": raw.nunique()})
num = raw.select_dtypes("number")
prof.loc[num.columns, "min"] = num.min().round(2)
prof.loc[num.columns, "max"] = num.max().round(2)
role = {v: k for k, v in mapping.items() if k in ("date", "consumption", "temperature", "price", "gas_price") and v}
prof["mapped as"] = [role.get(c, "") for c in prof.index]
eng = [c for c in raw.columns if str(c).lower().startswith(cu.ENGINEERED_PREFIXES)]
prof["pre-engineered"] = [c in eng for c in prof.index]
st.dataframe(prof)
if eng:
    st.caption(f"{len(eng)} columns look pre-engineered (lags, rolling stats, changes, volatility). "
               "The platform builds its own leakage-safe versions, so these are ignored for modelling.")

# -------------------------------------------------------
# TIME COVERAGE
# -------------------------------------------------------
st.subheader("Time coverage")
steps = data.index.to_series().diff().dropna()
step_counts = steps.value_counts().head(6)
c1, c2 = st.columns(2)
with c1:
    st.write(f"**{data.index.min():%Y-%m-%d %H:%M} → {data.index.max():%Y-%m-%d %H:%M}**  "
             f"({(data.index.max() - data.index.min()).days / 365.25:.1f} years)")
    st.dataframe(pd.DataFrame({"time step": step_counts.index.astype(str), "count": step_counts.values}),
                 hide_index=True)
    expected = (data.index.max() - data.index.min()) / steps.min() + 1 if len(steps) else len(data)
    coverage = len(data) / expected * 100
    st.metric("Coverage vs a complete series at the smallest step", f"{coverage:.0f}%")
with c2:
    per_month = data["consumption"].resample("MS").count()
    st.plotly_chart(px.bar(per_month, title="Readings per month", labels={"value": "readings", "index": ""}))

# -------------------------------------------------------
# PLAUSIBILITY CHECKS
# -------------------------------------------------------
st.subheader("Plausibility checks")
checks = []
c = data["consumption"]
checks.append(("Consumption has no negative values", bool((c >= 0).all()), f"min {c.min():,.1f}"))
q1, q3 = c.quantile([0.25, 0.75])
iqr_out = int(((c < q1 - 3 * (q3 - q1)) | (c > q3 + 3 * (q3 - q1))).sum())
checks.append(("Consumption extreme outliers (3×IQR) below 0.5%", iqr_out / len(c) < 0.005,
               f"{iqr_out} readings ({iqr_out / len(c) * 100:.2f}%)"))
if cu.freq_label(data) == "hourly":
    hp = data.groupby("hour")["consumption"].mean()
    swing = (hp.max() / hp.min() - 1) * 100
    checks.append(("Daily load cycle visible (hour-of-day swing > 10%)", swing > 10, f"swing {swing:.1f}%"))
if "temperature" in data:
    t = data["temperature"]
    checks.append(("Temperatures within Calgary range (-45 to 37 °C)", bool(t.between(-45, 37).all()),
                   f"{t.min():.1f} to {t.max():.1f} °C"))
    within = data.groupby(data.index.to_period("M"))["temperature"].std().mean()
    checks.append(("Temperature varies within each month (std > 3 °C)", within > 3, f"avg monthly std {within:.1f} °C"))
    daily = cu.to_daily(data)
    r = daily["consumption"].corr(daily["HDD"])
    checks.append(("Colder days use more energy (corr with HDD > 0)", r > 0, f"corr {r:.2f}"))
if "price" in data:
    p = data["price"]
    checks.append(("Electricity price within -100..1000 $/MWh (Alberta pool limits)",
                   bool(p.between(-100, 1000).all()), f"{p.min():.1f} to {p.max():.1f}"))
chk = pd.DataFrame(checks, columns=["Check", "Pass", "Detail"])
st.dataframe(chk, hide_index=True)

# -------------------------------------------------------
# QUALITY SCORE
# -------------------------------------------------------
completeness = 100 - raw[[v for v in role]].isna().mean().mean() * 100 if role else 100
regularity = (steps == steps.median()).mean() * 100 if len(steps) else 100
duplicates = 100 - min(raw.duplicated().mean() * 100 * 10, 100)
plaus = chk["Pass"].mean() * 100
score = round(0.3 * completeness + 0.2 * min(coverage, 100) + 0.15 * regularity + 0.1 * duplicates + 0.25 * plaus)
st.subheader("Data quality score")
s = st.columns(6)
s[0].metric("Overall", f"{score}/100")
s[1].metric("Completeness", f"{completeness:.0f}")
s[2].metric("Coverage", f"{min(coverage, 100):.0f}")
s[3].metric("Regular steps", f"{regularity:.0f}")
s[4].metric("No duplicates", f"{duplicates:.0f}")
s[5].metric("Plausibility", f"{plaus:.0f}")
st.caption("Weights: completeness 30%, coverage 20%, regular steps 15%, duplicates 10%, plausibility 25%.")

# -------------------------------------------------------
# DISTRIBUTION
# -------------------------------------------------------
st.subheader("Distribution of a column")
numeric_cols = view.select_dtypes("number").columns.tolist()
default = mapping["consumption"] if mapping["consumption"] in numeric_cols else numeric_cols[0]
col = st.selectbox("Column", numeric_cols, index=numeric_cols.index(default))
c1, c2 = st.columns(2)
c1.plotly_chart(px.histogram(view, x=col, nbins=60, marginal="box"))
c2.plotly_chart(px.line(view[col].resample("D").mean(), title=f"Daily mean of {col}"))

cu.ai_interpretation("Data quality", {
    "rows": len(raw), "rows_used": len(data), "period": f"{data.index.min()} to {data.index.max()}",
    "time_steps": {str(k): int(v) for k, v in step_counts.items()}, "coverage_pct": round(coverage, 1),
    "quality_score": score, "checks": chk.to_dict("records"), "cleaning": data.attrs.get("cleaning", {}),
    "pre_engineered_columns": len(eng)},
    "1) Overall data quality and fitness for forecasting / M&V 2) The most serious issues 3) Specific "
    "cleaning or data-collection steps to fix them 4) Which analyses are safe to run now.")
