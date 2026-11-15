import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

import calgary_utils as cu

cu.page_header("Financial & Procurement Risk", "Section 5 · Facility Decisions", """
Turns consumption and prices into money decisions:

1. **Cost history & budget**: what energy cost, whether usage falls in expensive hours, and a 12-month budget
   with a P10-P90 range
2. **Fixed vs floating procurement**: Monte Carlo of next year's cost, cost **Value-at-Risk** and the best
   **hedge ratio** (share bought at a fixed price)
3. **Investment appraisal**: NPV, IRR, simple and discounted payback and sensitivity for any energy project
""")
raw, mapping, data = cu.require_data()
if "price" not in data:
    st.error("Map an electricity price column ($/MWh) in the sidebar.")
    st.stop()
to_mwh = cu.to_mwh_factor(mapping)
d = data.assign(mwh=data["consumption"] * to_mwh)
d = d.assign(cost=d["mwh"] * d["price"])
monthly = d.groupby(d.index.to_period("M")).agg(mwh=("mwh", "sum"), cost=("cost", "sum"), price_avg=("price", "mean"))
monthly["load_weighted_price"] = monthly["cost"] / monthly["mwh"]
lw = d["cost"].sum() / d["mwh"].sum()
tw = d["price"].mean()
rng = np.random.default_rng(42)
results = {}

# expected MWh per calendar month (scaled to a full month even if readings are gappy)
steps = d.index.to_series().diff().dropna()
step_h = steps[steps > pd.Timedelta(0)].min() / pd.Timedelta(hours=1) if len(steps) else 24
step_h = min(max(step_h, 1), 24 * 31)
per_reading = d["mwh"].groupby(d.index.month).mean()
hours = {m: pd.Period(f"2025-{m:02d}").days_in_month * 24 for m in range(1, 13)}
exp_mwh = pd.Series({m: per_reading.get(m, per_reading.mean()) / step_h * hours[m] for m in range(1, 13)})
lwp = monthly["load_weighted_price"]
by_month = {m: lwp[lwp.index.month == m].values for m in range(1, 13)}


def simulate_year(n, level=None):
    sims = np.array([[exp_mwh[m] * rng.choice(by_month[m] if len(by_month[m]) else lwp.values) for m in range(1, 13)]
                     for _ in range(n)])
    return sims if level is None else sims * level[:, None]


tab1, tab2, tab3 = st.tabs(["Cost history & budget", "Fixed vs floating (Monte Carlo)", "Investment appraisal"])

with tab1:
    k = st.columns(4)
    k[0].metric("Average price paid (load-weighted)", f"${lw:.2f}/MWh")
    k[1].metric("Average market price (time-weighted)", f"${tw:.2f}/MWh")
    k[2].metric("Load-shape premium", f"{(lw / tw - 1) * 100:+.2f}%",
                help="Positive = consumption is concentrated in more expensive hours.")
    k[3].metric("Energy cost in the recorded readings", f"${d['cost'].sum():,.0f}")
    if step_h > 1 or (steps != steps.median()).mean() > 0.1:
        st.caption("Readings are not every hour, so the recorded total covers only those hours. The budget below "
                   "scales each month to a full month.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly.index.to_timestamp(), y=monthly["cost"], name="Energy cost $ (recorded)"))
    fig.add_trace(go.Scatter(x=monthly.index.to_timestamp(), y=monthly["load_weighted_price"], name="$/MWh paid", yaxis="y2"))
    fig.update_layout(height=380, yaxis2=dict(overlaying="y", side="right", title="$/MWh"), title="Monthly energy cost")
    st.plotly_chart(fig)

    st.subheader("12-month budget")
    sims = simulate_year(3000)
    annual = sims.sum(axis=1)
    budget = pd.DataFrame({"Expected MWh": exp_mwh.values, "P10 $": np.percentile(sims, 10, axis=0),
                           "P50 $": np.percentile(sims, 50, axis=0), "P90 $": np.percentile(sims, 90, axis=0)},
                          index=[pd.Timestamp(f"2025-{m:02d}-01").strftime("%b") for m in range(1, 13)])
    k = st.columns(3)
    k[0].metric("Annual budget P50", f"${np.percentile(annual, 50):,.0f}")
    k[1].metric("P10 (good year)", f"${np.percentile(annual, 10):,.0f}")
    k[2].metric("P90 (bad year)", f"${np.percentile(annual, 90):,.0f}")
    st.dataframe(budget.round(0))
    st.caption("Consumption = average for each calendar month; price = that month's load-weighted price from a "
               "randomly chosen past year. This range only reflects past month-to-month price variation, not a "
               "change in next year's overall price level (see the Fixed vs floating tab).")
    results["budget"] = {"p10": np.percentile(annual, 10), "p50": np.percentile(annual, 50),
                         "p90": np.percentile(annual, 90), "expected_mwh": exp_mwh.sum(), "load_weighted_price": lw,
                         "time_weighted_price": tw, "load_shape_premium_pct": (lw / tw - 1) * 100}

with tab2:
    st.markdown("Each simulated year draws every month's price from a random past year, then scales the whole "
                "year by a market-level factor (uncertainty about next year's general price level).")
    c = st.columns(3)
    fixed_price = c[0].number_input("Fixed contract price ($/MWh)", value=float(round(lw * 1.05, 1)))
    level_sd = c[1].slider("Uncertainty of next year's price level (1 std, %)", 0, 50, 20) / 100
    n_sims = c[2].select_slider("Simulations", [1000, 3000, 10000], 3000)
    levels = rng.lognormal(mean=-0.5 * level_sd ** 2, sigma=level_sd, size=n_sims) if level_sd else np.ones(n_sims)
    float_cost = simulate_year(n_sims, levels).sum(axis=1)
    fixed_cost = exp_mwh.sum() * fixed_price
    rows = []
    for h in np.linspace(0, 1, 11):
        cost = h * fixed_cost + (1 - h) * float_cost
        var95 = np.percentile(cost, 95)
        rows.append({"Hedge % fixed": int(round(h * 100)), "Expected cost $": cost.mean(), "VaR 95% $": var95,
                     "CVaR 95% $": cost[cost >= var95].mean(), "Std $": cost.std()})
    hedge = pd.DataFrame(rows)
    risk_aversion = st.slider("Risk aversion (weight on bad-year cost)", 0.0, 1.0, 0.5, 0.05,
                              help="0 = minimize expected cost; 1 = minimize the cost of a bad (worst 5%) year.")
    hedge["Objective"] = (1 - risk_aversion) * hedge["Expected cost $"] + risk_aversion * hedge["CVaR 95% $"]
    best_h = hedge.loc[hedge["Objective"].idxmin()]
    k = st.columns(4)
    k[0].metric("Floating: expected cost", f"${float_cost.mean():,.0f}")
    k[1].metric("Floating: VaR 95%", f"${np.percentile(float_cost, 95):,.0f}", help="Cost exceeded in 5% of years")
    k[2].metric("Fixed contract cost", f"${fixed_cost:,.0f}")
    k[3].metric("Recommended hedge", f"{int(best_h['Hedge % fixed'])}% fixed")
    c1, c2 = st.columns(2)
    fig = px.histogram(x=float_cost, nbins=60, title="Floating-price annual cost", labels={"x": "annual cost $"})
    fig.add_vline(x=fixed_cost, line_dash="dash", annotation_text="fixed")
    c1.plotly_chart(fig)
    fig = go.Figure(go.Scatter(x=hedge["CVaR 95% $"], y=hedge["Expected cost $"], mode="lines+markers+text",
                               text=hedge["Hedge % fixed"].astype(str) + "%", textposition="top center"))
    fig.update_layout(title="Risk-cost frontier (each point = hedge ratio)", xaxis_title="Bad-year cost (CVaR 95%)",
                      yaxis_title="Expected cost", height=420)
    c2.plotly_chart(fig)
    st.dataframe(hedge.round(0), hide_index=True)
    st.caption("VaR/CVaR are on the cost's upper tail: the bad years for a buyer.")
    results["procurement"] = {"fixed_price": fixed_price, "floating_expected": float_cost.mean(),
                              "floating_var95": np.percentile(float_cost, 95), "fixed_cost": fixed_cost,
                              "recommended_hedge_pct": int(best_h["Hedge % fixed"]), "risk_aversion": risk_aversion}

with tab3:
    saved = cu.get_results()
    eff = (saved.get("Process optimization", {}).get("results", {}) or {}).get("efficiency") or {}
    des = saved.get("Design optimization & scale-up", {}).get("results", {}) or {}
    options = ["Manual"] + (["Efficiency package (page 14)"] if eff else []) + \
              (["PV + battery design (page 15)"] if des.get("capex") else [])
    source = st.radio("Pre-fill from", options, horizontal=True)
    if source.startswith("Efficiency"):
        cap0, sav0 = float(eff.get("capex", 150000)), float(eff.get("annual_savings_$", 30000))
    elif source.startswith("PV"):
        cap0, sav0 = float(des["capex"]), float(des["annual_saving"])
    else:
        cap0, sav0 = 150000.0, 30000.0
    c = st.columns(3)
    capex = c[0].number_input("Investment ($)", value=cap0, step=10000.0)
    saving = c[0].number_input("Year-1 savings ($/yr)", value=sav0, step=1000.0)
    om = c[1].number_input("Extra O&M ($/yr)", value=0.0, step=500.0)
    esc = c[1].number_input("Energy price escalation (%/yr)", value=2.0) / 100
    life = int(c[2].number_input("Life (years)", value=15, min_value=1))
    rate = c[2].number_input("Discount rate (%)", value=8.0) / 100

    def appraise(cap, sav, r=rate):
        cf = [-cap] + [sav * (1 + esc) ** (t - 1) - om for t in range(1, life + 1)]
        return cf, sum(v / (1 + r) ** t for t, v in enumerate(cf))

    cf, npv = appraise(capex, saving)
    try:
        irr = brentq(lambda r: sum(v / (1 + r) ** t for t, v in enumerate(cf)), -0.99, 10)
    except ValueError:
        irr = float("nan")
    cum = np.cumsum(cf)
    disc = np.cumsum([v / (1 + rate) ** t for t, v in enumerate(cf)])
    pay = next((t for t, v in enumerate(cum) if v >= 0), None)
    dpay = next((t for t, v in enumerate(disc) if v >= 0), None)
    k = st.columns(4)
    k[0].metric("NPV", f"${npv:,.0f}", "approve" if npv > 0 else "reject", delta_color="normal" if npv > 0 else "inverse")
    k[1].metric("IRR", "n/a" if np.isnan(irr) else f"{irr * 100:.1f}%")
    k[2].metric("Simple payback", f"{pay} yr" if pay is not None else "> life")
    k[3].metric("Discounted payback", f"{dpay} yr" if dpay is not None else "> life")
    fig = px.line(pd.DataFrame({"Cumulative cash flow": cum, "Discounted cumulative": disc}), markers=True,
                  title="Cumulative cash flow by year", labels={"index": "year", "value": "$"})
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig)
    tor = pd.DataFrame([
        {"Variable": "Investment ±20%", "Favourable": appraise(capex * 0.8, saving)[1] - npv,
         "Unfavourable": appraise(capex * 1.2, saving)[1] - npv},
        {"Variable": "Savings ±20%", "Favourable": appraise(capex, saving * 1.2)[1] - npv,
         "Unfavourable": appraise(capex, saving * 0.8)[1] - npv},
        {"Variable": "Discount rate ±2 pts", "Favourable": appraise(capex, saving, rate - 0.02)[1] - npv,
         "Unfavourable": appraise(capex, saving, rate + 0.02)[1] - npv}])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=tor["Variable"], x=tor["Favourable"], orientation="h", name="favourable"))
    fig.add_trace(go.Bar(y=tor["Variable"], x=tor["Unfavourable"], orientation="h", name="unfavourable"))
    fig.update_layout(barmode="overlay", title=f"NPV sensitivity (base ${npv:,.0f})", height=300, xaxis_title="Δ NPV $")
    st.plotly_chart(fig)
    results["investment"] = {"source": source, "capex": capex, "year1_savings": saving, "npv": npv, "irr": irr,
                             "payback_years": pay, "discounted_payback_years": dpay, "life": life, "rate": rate}

cu.ai_interpretation("Financial & procurement risk", results,
                     "1) Energy cost exposure and budget range 2) Recommended procurement / hedging strategy "
                     "3) Investment decision (NPV, IRR, payback, sensitivity) 4) Financial risks to communicate "
                     "to management.")
