import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy.stats import norm

import calgary_utils as cu

cu.page_header("AI Decision Engine & Report", "Section 6 · AI Decision Support", """
The decision layer. It combines a fresh analysis of the data with the **results saved by every page you ran
in this session** and produces:

| Output | What it is |
|---|---|
| **Engineering report** | Structured JSON of all results → LLM interpretation → PDF / Markdown |
| **Decision engine** | Insight, root cause, risk and ranked **next-best actions** |
| **Energy trial design** | The energy version of design of experiments: how many days a savings trial must run to prove the saving |

Without a running LLM, the report and actions fall back to transparent rule-based versions.
""")
raw, mapping, data = cu.require_data()
LLM_OK = cu.llm_available()
st.caption(f"Local LLM ({cu.LLM_MODEL}): {'connected' if LLM_OK else 'not running, rule-based fallbacks active'}")


@st.cache_data(show_spinner=False)
def core_analytics(df):
    daily = cu.to_daily(df)
    out = {"period": f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}", "readings": int(len(df)),
           "frequency": cu.freq_label(df), "mean_per_reading": float(df["consumption"].mean()),
           "peak_reading": float(df["consumption"].max()), "peak_time": str(df["consumption"].idxmax()),
           "est_annual_total": float(daily["consumption"].mean() * 365),
           "avg_daily_by_month": {int(k): round(float(v), 1) for k, v in
                                  daily["consumption"].groupby(daily.index.month).mean().items()}}
    if cu.freq_label(df) == "hourly":
        hp = df.groupby("hour")["consumption"].mean()
        out["hourly_profile_swing_pct"] = float((hp.max() / hp.min() - 1) * 100)
    model_cv, rho = None, 0.0
    if "HDD" in daily:
        ols = sm.OLS(daily["consumption"], sm.add_constant(daily[["HDD", "CDD", "is_weekend"]])).fit()
        out["daily_weather_model"] = {"R2": ols.rsquared, "base_per_day": ols.params["const"],
                                      "per_HDD": ols.params["HDD"], "per_CDD": ols.params["CDD"]}
        resid = ols.resid
        model_cv = float(np.sqrt((resid ** 2).sum() / (len(resid) - 4)) / daily["consumption"].mean())
        rho = float(np.clip(resid.autocorr(1), 0, 0.95)) if len(resid) > 10 else 0.0
        out["daily_model_cv_pct"], out["residual_autocorrelation"] = model_cv * 100, rho
        if ols.params["CDD"] > 0 and ols.params["HDD"] > 0:
            out["weather_note"] = ("Cooling sensitivity larger than heating (unusual for Calgary)"
                                   if ols.params["CDD"] > ols.params["HDD"] else "Heating-dominated (typical for Calgary)")
    z = daily["consumption"] - daily["consumption"].rolling(15, center=True, min_periods=5).median()
    mad = (z - z.median()).abs().median()
    out["abnormal_days"] = int((0.6745 * (z - z.median()).abs() / mad > 4).sum()) if mad else 0
    return cu.jsonable(out), model_cv, rho


eda, model_cv, rho = core_analytics(data)
results = cu.get_results()
payload = cu.jsonable({"site": "Calgary facility / energy system", "data_summary": eda,
                       "data_cleaning": data.attrs.get("cleaning", {}),
                       "module_results": {k: v["results"] for k, v in results.items()}})
ran = [k for k in results if k not in ("Decision engine", "Energy trial design")]
with st.expander(f"Inputs to the AI: {len(ran)} page result(s) saved this session"):
    st.caption("Saved: " + (", ".join(ran) if ran else "none yet. Run pages 1-16 first for a complete report."))
    st.json(payload, expanded=False)

tab1, tab2, tab3 = st.tabs(["Engineering report", "Decision engine & next-best actions", "Energy trial design"])

# =======================================================
# 1. ENGINEERING REPORT
# =======================================================
REPORT_PROMPT = """You are a senior energy engineer reviewing a Calgary energy system.

Structured results (JSON):
{payload}

Write an engineering report with these markdown sections:
1. Executive summary: 4-6 lines with the key numbers
2. Consumption performance: totals, profile, peaks, abnormal days
3. Weather and schedule drivers: heating / cooling sensitivity, weekday / weekend
4. Model reliability: R2, CV(RMSE), NMBE, forecast skill, uncertainty; are models fit for decisions?
5. Equipment condition: predictive-maintenance findings
6. Market and financial risk: price risk, budget range, hedging
7. Optimization opportunities: load shifting, efficiency, PV / battery; savings, payback, NPV
8. Risks and data limitations
9. Recommended actions: prioritised, quantified, with owner and timeframe

Rules: use only the numbers provided; write "not analysed" for modules that are missing; no filler."""


def fallback_report(p):
    e = p["data_summary"]
    lines = ["# Executive summary",
             f"- Period {e['period']}, {e['readings']:,} readings ({e['frequency']}).",
             f"- Estimated annual consumption {e['est_annual_total']:,.0f}; peak {e['peak_reading']:,.1f} at {e['peak_time']}.",
             f"- {e['abnormal_days']} abnormal days (robust z > 4 vs 15-day median)."]
    if "hourly_profile_swing_pct" in e:
        lines.append(f"- Average daily load swing between hours: {e['hourly_profile_swing_pct']:.1f}%.")
    if "daily_weather_model" in e:
        w = e["daily_weather_model"]
        lines += ["# Weather drivers",
                  f"- Daily model R2 {w['R2']:.3f}: base {w['base_per_day']:,.1f}/day, {w['per_HDD']:.2f} per heating "
                  f"degree-day, {w['per_CDD']:.2f} per cooling degree-day.", f"- {e.get('weather_note', '')}"]
    for mod, r in p["module_results"].items():
        lines.append(f"# {mod}")
        for k, v in (r.items() if isinstance(r, dict) else [("value", r)]):
            if not isinstance(v, (dict, list)):
                lines.append(f"- {k}: {round(v, 3) if isinstance(v, float) else v}")
    lines += ["# Note", "- Rule-based summary (LLM not running). Start Ollama for an interpreted report."]
    return "\n".join(lines)


with tab1:
    if st.button("Generate engineering report", type="primary"):
        with st.spinner("Writing report..."):
            st.session_state["eng_report"] = (
                cu.call_llm(REPORT_PROMPT.format(payload=json.dumps(payload, indent=1)[:14000]),
                            system="You are a precise, quantitative energy engineer.")
                if LLM_OK else fallback_report(payload))
    if "eng_report" in st.session_state:
        text = st.session_state["eng_report"]
        st.markdown(text)
        c1, c2 = st.columns(2)
        pdf = cu.to_pdf_bytes(text, "Calgary Energy Engineering Report")
        if pdf:
            c1.download_button("Download PDF", pdf, "calgary_energy_report.pdf", "application/pdf")
        else:
            c1.caption("Install `reportlab` for PDF export.")
        c2.download_button("Download Markdown", text, "calgary_energy_report.md")

# =======================================================
# 2. DECISION ENGINE
# =======================================================
DECISION_PROMPT = """You are the decision engine of a Calgary energy platform.
DATA (JSON):
{payload}

Return ONLY valid JSON:
{{"insight": "2-3 sentences with numbers",
  "root_cause": "most likely driver of current cost / consumption / condition",
  "risk": "LOW|MEDIUM|HIGH",
  "confidence": 0.0-1.0,
  "next_actions": [{{"action": "...", "category": "operations|maintenance|capital|procurement|data",
                    "expected_saving_pct": number or null, "priority": 1-5, "reason": "..."}}]}}
3-6 actions ranked by priority. Base every number on DATA."""


def rule_based(p):
    acts, mr, e = [], p["module_results"], p["data_summary"]

    def add(action, category, saving, priority, reason):
        acts.append({"action": action, "category": category, "expected_saving_pct": saving,
                     "priority": priority, "reason": reason})

    dq = mr.get("Data quality", {})
    if dq and dq.get("quality_score", 100) < 75:
        add("Fix data quality before relying on models (see page 1 checks)", "data", None, 1,
            f"Data quality score {dq['quality_score']}/100")
    if e.get("weather_note", "").startswith("Cooling"):
        add("Validate temperature data and cooling loads", "data", None, 2, e["weather_note"])
    pm = mr.get("Predictive maintenance", {})
    if pm and str(pm.get("latest_risk")) in ("High", "Medium"):
        add("Inspect HVAC / process equipment flagged by the health index", "maintenance",
            abs(pm.get("drift_7d_pct") or 3), 1, f"Risk {pm.get('latest_risk')}, 7-day drift {pm.get('drift_7d_pct')}%")
    po = mr.get("Process optimization", {})
    sh = po.get("load_shifting") if isinstance(po.get("load_shifting"), dict) else None
    if sh and sh.get("savings_pct", 0) > 1:
        add("Shift flexible loads to low-price hours", "operations", sh["savings_pct"], 2, "Dispatch LP savings")
    eff = po.get("efficiency") or {}
    if eff and (eff.get("npv") or 0) > 0:
        add("Fund the efficiency package (base load, heating, cooling)", "capital", None, 2,
            f"NPV ${eff['npv']:,.0f}, payback {eff.get('payback_years', 0):.1f} yr")
    de = mr.get("Design optimization & scale-up", {})
    if de and (de.get("npv") or 0) > 0:
        add(f"Install {de['optimal_pv_kw']:.0f} kW PV + {de['optimal_battery_kwh']:.0f} kWh battery", "capital", None, 3,
            f"NPV ${de['npv']:,.0f}, risk {de.get('investment_risk')}")
    fin = (mr.get("Financial & procurement risk", {}) or {}).get("procurement")
    if fin:
        add(f"Hedge about {fin['recommended_hedge_pct']}% of next year's volume at a fixed price", "procurement", None, 3,
            f"Floating VaR95 ${fin['floating_var95']:,.0f} vs fixed ${fin['fixed_cost']:,.0f}")
    if e.get("abnormal_days", 0) > 0:
        add(f"Investigate {e['abnormal_days']} abnormal days (schedules, faults, meter errors)", "operations", None, 3,
            "Robust anomaly screen on daily totals")
    if not acts:
        add("Run pages 1-16 to generate evidence for decisions", "data", None, 1, "No module results yet")
    return {"insight": "Rule-based decision (LLM not running).", "root_cause": "See the actions and their reasons.",
            "risk": "HIGH" if any(a["priority"] == 1 for a in acts if a["category"] == "maintenance") else "MEDIUM",
            "confidence": 0.5, "next_actions": sorted(acts, key=lambda a: a["priority"])}


with tab2:
    if st.button("Run decision engine", type="primary"):
        decision = None
        if LLM_OK:
            with st.spinner("Reasoning..."):
                txt = cu.call_llm(DECISION_PROMPT.format(payload=json.dumps(payload)[:14000]), json_mode=True)
            try:
                decision = json.loads(txt)
                assert isinstance(decision.get("next_actions"), list)
            except Exception:
                st.warning("The LLM did not return valid JSON; using the rule-based engine.")
                decision = None
        st.session_state["decision"] = decision or rule_based(payload)
    if "decision" in st.session_state:
        dsn = st.session_state["decision"]
        k = st.columns(3)
        k[0].metric("Risk", dsn.get("risk", "-"))
        k[1].metric("Confidence", f"{float(dsn.get('confidence') or 0):.0%}")
        k[2].metric("Actions", len(dsn.get("next_actions", [])))
        st.info(f"**Insight:** {dsn.get('insight', '')}\n\n**Root cause:** {dsn.get('root_cause', '')}")
        st.dataframe(pd.DataFrame(dsn.get("next_actions", [])), hide_index=True)
        cu.save_result("Decision engine", dsn)

# =======================================================
# 3. ENERGY TRIAL DESIGN
# =======================================================
def days_needed(saving_pct, cv, rho, alpha=0.05, power=0.8):
    """Days of trial data to detect a saving against a weather-normalized daily baseline:
    n = ((z_(1-a/2) + z_power) * CV / s)^2, inflated by (1+rho)/(1-rho) for autocorrelated residuals."""
    s = (saving_pct or 0) / 100
    if s <= 0 or not cv:
        return None
    return int(np.ceil(((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * cv / s) ** 2 * (1 + rho) / (1 - rho)))


with tab3:
    st.markdown("Before rolling out an operational change, **trial** it and measure the saving against the "
                "weather-normalized baseline (page 8). The noisier the baseline, the longer the trial must run.")
    if model_cv is None:
        st.warning("Needs a temperature column for the daily baseline model.")
    else:
        c = st.columns(3)
        cv_in = c[0].number_input("Daily baseline CV(RMSE) %", value=round(model_cv * 100, 2)) / 100
        conf = c[1].selectbox("Confidence", [0.90, 0.95], index=1)
        power = c[2].selectbox("Power (chance to detect a real saving)", [0.8, 0.9])
        trials = pd.DataFrame([
            ("Night / weekend HVAC setback", "operations", 5.0, "Heating season"),
            ("Optimal start (later warm-up)", "operations", 3.0, "Heating season"),
            ("Supply-air / hot-water temperature reset", "operations", 4.0, "Heating season"),
            ("After-hours base-load shutdown", "operations", 6.0, "Any"),
            ("Pre-heat before peak hours (load shift)", "operations", 2.0, "Heating season"),
            ("Economizer / free-cooling repair", "maintenance", 3.0, "Shoulder / cooling season"),
        ], columns=["Trial", "Category", "Expected saving %", "Best season"])
        edited = st.data_editor(trials, num_rows="dynamic", key="trial_editor")
        edited = edited.assign(**{"Days needed": [days_needed(s, cv_in, rho, 1 - conf, power) for s in edited["Expected saving %"]]})
        edited["Feasible in one season (≤ 90 d)"] = edited["Days needed"].apply(
            lambda n: "yes" if n and n <= 90 else "no: alternate ON/OFF days or use more sites")
        st.dataframe(edited, hide_index=True)
        st.caption(f"Residual day-to-day autocorrelation ρ = {rho:.2f} is included (correlated days carry less "
                   "information). Halving the expected saving quadruples the days needed. Best design: alternate the "
                   "measure ON/OFF by day or week so both periods see similar weather.")
        if LLM_OK and st.button("AI: write the trial plan"):
            with st.spinner("Designing the trial..."):
                st.markdown(cu.call_llm(
                    "You are an M&V engineer (IPMVP). Design measurement trials for a Calgary facility using this "
                    f"table:\n{edited.to_string()}\nData summary: {json.dumps(eda)}\nFor the 3 most feasible trials "
                    "give: hypothesis, ON/OFF schedule, duration (use 'Days needed'), measurement boundary, success "
                    "criterion, risks."))
        cu.save_result("Energy trial design", {"cv_pct": cv_in * 100, "rho": rho, "trials": edited.to_dict("records")})
