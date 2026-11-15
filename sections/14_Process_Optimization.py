import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import calgary_utils as cu

cu.page_header("Process Optimization", "Section 5 · Facility Decisions", """
Three ways to cut the cost of Calgary energy consumption:

1. **Load shifting**: a linear program moves flexible load from expensive to cheap hours within each day
   and shaves the monthly peak (energy cost + demand charge)
2. **Efficiency measures**: weather-normalized savings from lower base load and heating / cooling
   sensitivity, with payback and NPV
3. **Weather what-if**: how consumption and cost respond to a colder or warmer year
""")
raw, mapping, data = cu.require_data()
hourly = cu.freq_label(data) == "hourly"
to_mwh = cu.to_mwh_factor(mapping)

c1, c2, c3 = st.columns(3)
price_src = c1.radio("Energy price", (["Price column in data"] if "price" in data else []) + ["Time-of-use tariff"])
demand_charge = c2.number_input("Demand charge ($/kW-month)", value=10.0, min_value=0.0,
                                help="Charge on the monthly peak; check your utility / retailer bill.")
if price_src == "Time-of-use tariff":
    on_rate = c3.number_input("On-peak ($/MWh)", value=140.0)
    off_rate = c3.number_input("Off-peak ($/MWh)", value=70.0)
    on_start, on_end = st.slider("On-peak hours (weekdays)", 0, 23, (7, 21))
    is_on = ((data["hour"] >= on_start) & (data["hour"] <= on_end) & (data["is_weekend"] == 0)) if hourly \
        else pd.Series(True, index=data.index)
    data = data.assign(opt_price=np.where(is_on, on_rate, off_rate))
else:
    data = data.assign(opt_price=data["price"])
data = data.assign(mwh=data["consumption"] * to_mwh)
data = data.assign(cost=data["mwh"] * data["opt_price"])
avg_price = float(data["cost"].sum() / data["mwh"].sum())

tab1, tab2, tab3 = st.tabs(["Load shifting (LP)", "Efficiency measures & payback", "Weather what-if"])


def optimize_month(load, price, flex, up, dc_per_mw):
    """min Σ price·x + dc·P  s.t. daily energy conserved, x_h ≤ P, load(1-flex) ≤ x ≤ load(1+up)."""
    n = len(load)
    days = pd.Index(load.index.date)
    uniq = days.unique()
    A_eq = lil_matrix((len(uniq), n + 1))
    for i, dd in enumerate(uniq):
        A_eq[i, np.where(days == dd)[0]] = 1
    A_ub = lil_matrix((n, n + 1))
    A_ub[np.arange(n), np.arange(n)] = 1
    A_ub[:, n] = -1
    r = linprog(np.append(price.values, dc_per_mw), A_ub=A_ub.tocsr(), b_ub=np.zeros(n), A_eq=A_eq.tocsr(),
                b_eq=load.groupby(days).sum().reindex(uniq).values,
                bounds=[(l * (1 - flex), l * (1 + up)) for l in load.values] + [(0, None)], method="highs")
    return (pd.Series(r.x[:n], index=load.index), r.x[n]) if r.success else (None, None)


with tab1:
    if not hourly:
        st.info("Load shifting needs hourly data; use the Efficiency tab.")
    else:
        c1, c2, c3 = st.columns(3)
        flex = c1.slider("Flexible share of each hour's load (%)", 0, 50, 15,
                         help="Pre-heating/cooling, EV charging, pumps, batch processes.") / 100
        up = c2.slider("Max increase in any hour (%)", 0, 100, 30) / 100
        months = sorted(data.index.to_period("M").unique())
        sel = c3.select_slider("Months", options=[str(m) for m in months],
                               value=(str(months[max(0, len(months) - 12)]), str(months[-1])))
        if st.button("Run load-shifting optimization", type="primary"):
            dc_mw = demand_charge * 1000
            rows, new = [], []
            sel_m = [m for m in months if sel[0] <= str(m) <= sel[1]]
            prog = st.progress(0.0)
            for i, m in enumerate(sel_m):
                part = data[data.index.to_period("M") == m]
                x, peak = optimize_month(part["mwh"], part["opt_price"], flex, up, dc_mw)
                prog.progress((i + 1) / len(sel_m))
                if x is None:
                    continue
                new.append(x)
                rows.append({"Month": str(m), "Baseline energy $": float((part["mwh"] * part["opt_price"]).sum()),
                             "Optimized energy $": float((x * part["opt_price"]).sum()),
                             "Baseline peak MW": part["mwh"].max(), "Optimized peak MW": peak,
                             "Baseline demand $": part["mwh"].max() * dc_mw, "Optimized demand $": peak * dc_mw})
            prog.empty()
            st.session_state["shift_result"] = (pd.DataFrame(rows), pd.concat(new) if new else None, flex)
        sr = st.session_state.get("shift_result")
        if sr and sr[1] is not None:
            res, newload, flex_used = sr
            res["Baseline total $"] = res["Baseline energy $"] + res["Baseline demand $"]
            res["Optimized total $"] = res["Optimized energy $"] + res["Optimized demand $"]
            res["Savings $"] = res["Baseline total $"] - res["Optimized total $"]
            tb, to = res["Baseline total $"].sum(), res["Optimized total $"].sum()
            k = st.columns(4)
            k[0].metric("Baseline cost", f"${tb:,.0f}")
            k[1].metric("Optimized cost", f"${to:,.0f}")
            k[2].metric("Savings", f"${tb - to:,.0f}", f"{(tb - to) / tb * 100:.1f}%")
            k[3].metric("Avg peak reduction", f"{(1 - res['Optimized peak MW'].sum() / res['Baseline peak MW'].sum()) * 100:.1f}%")
            st.dataframe(res.round(2), hide_index=True)
            base = data.loc[newload.index, "mwh"]
            prof = pd.DataFrame({"Baseline": base.groupby(base.index.hour).mean(),
                                 "Optimized": newload.groupby(newload.index.hour).mean()})
            pr = data.loc[newload.index, "opt_price"]
            fig = go.Figure()
            for col in prof:
                fig.add_trace(go.Scatter(x=prof.index, y=prof[col], name=col, mode="lines+markers"))
            fig.add_trace(go.Bar(x=prof.index, y=pr.groupby(pr.index.hour).mean(), name="Avg price $/MWh",
                                 yaxis="y2", opacity=0.25))
            fig.update_layout(title="Average daily profile before / after", xaxis_title="Hour", yaxis_title="MWh per reading",
                              yaxis2=dict(overlaying="y", side="right", title="$/MWh"), height=400)
            st.plotly_chart(fig)
            st.caption("Totals cover the readings in the data; with gappy data they understate a full year.")
            st.session_state["opt_shift"] = {"flexible_share": flex_used, "baseline_cost": tb, "optimized_cost": to,
                                             "savings_pct": round((tb - to) / tb * 100, 2)}

daily = cu.to_daily(data)
daily["mwh"] = daily["consumption"] * to_mwh
model_cols = [c for c in ["HDD", "CDD", "is_weekend"] if c in daily]
ols = sm.OLS(daily["mwh"], sm.add_constant(daily[model_cols])).fit()

with tab2:
    st.markdown(f"Daily model: **MWh = base + b·HDD + c·CDD + d·weekend** (R² = {ols.rsquared:.3f}). "
                "Savings use the last 365 days of weather, so they are weather-normalized.")
    st.dataframe(pd.DataFrame({"coef": ols.params, "p_value": ols.pvalues}).round(4))
    last = daily.tail(365)
    c1, c2, c3 = st.columns(3)
    base_red = c1.slider("Base-load reduction (%)", 0, 40, 10, help="LED, scheduling, plug and process idle") / 100
    heat_red = c2.slider("Heating sensitivity reduction (%)", 0, 50, 15, help="Envelope, heat recovery, boiler/RTU") / 100
    cool_red = c3.slider("Cooling sensitivity reduction (%)", 0, 50, 10, help="Economizer, chiller/RTU, setpoints") / 100
    c4, c5, c6 = st.columns(3)
    capex = c4.number_input("Project cost ($)", value=150000.0, step=10000.0)
    life = c5.number_input("Measure life (years)", value=15, min_value=1)
    rate = c6.number_input("Discount rate (%)", value=8.0) / 100
    p = ols.params
    base_term = p["const"] + (p["is_weekend"] * last["is_weekend"] if "is_weekend" in p else 0)
    heat = heat_red * p.get("HDD", 0) * last.get("HDD", 0)
    cool = cool_red * p.get("CDD", 0) * last.get("CDD", 0)
    sav = np.clip(base_red * base_term + heat + cool, 0, None)
    annual_mwh = float(sav.sum()) * 365 / len(last)
    annual_total = float(last["mwh"].sum()) * 365 / len(last)
    annual_dollars = annual_mwh * avg_price
    payback = capex / annual_dollars if annual_dollars > 0 else float("nan")
    npv = sum(annual_dollars / (1 + rate) ** t for t in range(1, int(life) + 1)) - capex
    k = st.columns(4)
    k[0].metric("Annual savings", f"{annual_mwh:,.1f} MWh", f"{annual_mwh / annual_total * 100:.1f}% of use")
    k[1].metric("Annual $ savings", f"${annual_dollars:,.0f}", f"at ${avg_price:,.0f}/MWh", delta_color="off")
    k[2].metric("Simple payback", f"{payback:.1f} years")
    k[3].metric(f"NPV ({int(life)} y @ {rate:.0%})", f"${npv:,.0f}", "approve" if npv > 0 else "reject",
                delta_color="normal" if npv > 0 else "inverse")
    st.caption("NPV = Σ savings/(1+r)^t − cost. Electricity only; add gas savings separately for gas heating.")
    split = pd.DataFrame({"Base load": base_red * base_term, "Heating": heat, "Cooling": cool}, index=last.index)
    split = split.groupby(split.index.to_period("M")).sum().to_timestamp()
    st.plotly_chart(px.bar(split, title="Monthly savings by measure (MWh)"))
    eff = {"annual_savings_mwh": annual_mwh, "annual_savings_$": annual_dollars, "payback_years": payback,
           "npv": npv, "capex": capex, "model_r2": ols.rsquared}

with tab3:
    if "HDD" not in daily:
        st.info("Map a temperature column to run weather scenarios.")
    else:
        shift = st.slider("Temperature change vs last 12 months (°C)", -8.0, 8.0, -3.0, 0.5)
        last = daily.tail(365).copy()
        t_new = last["temperature"] + shift
        scen = last.assign(HDD=np.clip(cu.HDD_BASE_C - t_new, 0, None), CDD=np.clip(t_new - cu.CDD_BASE_C, 0, None))
        base_pred = ols.predict(sm.add_constant(last[model_cols], has_constant="add"))
        new_pred = ols.predict(sm.add_constant(scen[model_cols], has_constant="add"))
        source = "daily HDD/CDD regression"
        tuned = st.session_state.get("tuned_model")
        if tuned and not tuned["use_lags"] and hourly:
            h = data[data.index >= last.index[0]]
            h2 = h.assign(temperature=h["temperature"] + shift)
            h2 = h2.assign(HDD=np.clip(cu.HDD_BASE_C - h2["temperature"], 0, None),
                           CDD=np.clip(h2["temperature"] - cu.CDD_BASE_C, 0, None))
            f = tuned["features"]
            base_pred = pd.Series(tuned["model"].predict(h[f].astype(float)) * to_mwh, index=h.index).resample("D").sum()
            new_pred = pd.Series(tuned["model"].predict(h2[f].astype(float)) * to_mwh, index=h.index).resample("D").sum()
            source = f"tuned {tuned['model_name']} (page 9)"
        delta = float(new_pred.sum() - base_pred.sum())
        k = st.columns(2)
        k[0].metric("Annual consumption change", f"{delta:+,.1f} MWh", f"{delta / base_pred.sum() * 100:+.1f}%")
        k[1].metric("Annual cost change", f"${delta * avg_price:+,.0f}")
        st.caption(f"Model: {source}.")
        comp = pd.DataFrame({"Last 12 months": base_pred, f"{shift:+.1f} °C": new_pred})
        comp = comp.groupby(comp.index.to_period("M")).sum().to_timestamp()
        st.plotly_chart(px.line(comp, markers=True, title="Monthly MWh under the scenario"))

cu.ai_interpretation("Process optimization", {
    "load_shifting": st.session_state.get("opt_shift", "not run"), "efficiency": eff,
    "weather_model": {"r2": ols.rsquared, "coefficients": ols.params.to_dict()}, "avg_price": avg_price},
    "1) Best cost-reduction levers 2) Is the efficiency package worth it (payback / NPV) 3) Constraints and "
    "risks of load shifting 4) A 3-step action plan.")
