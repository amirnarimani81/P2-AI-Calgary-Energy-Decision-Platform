import pandas as pd
import streamlit as st

import calgary_utils as cu

def _find_pic(stem):
    """Return the image named `stem` in the app folder, whatever its extension (png / jpg / jpeg / webp)."""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        if (p := cu.ROOT / f"{stem}{ext}").exists():
            return p
    return cu.ROOT / f"{stem}.png"  # missing: .exists() is False, so the image is skipped


PIC1, PIC2 = _find_pic("pic1"), _find_pic("pic2")

# =====================================================
# HERO
# =====================================================
st.title("AI Calgary Energy Decision")
if PIC1.exists():
    st.image(str(PIC1), caption="Calgary Energy Decision Intelligence Platform")
st.subheader("AI Platform for Calgary Energy Consumption Analytics, Forecasting, Optimization and "
             "Executive Decision Support")
st.markdown("""
An end-to-end Artificial Intelligence platform that turns Calgary **energy consumption, weather and market
data** into actionable business intelligence: data quality → exploration → market risk → models and
forecasts → facility decisions (maintenance, optimization, investment) → AI reports and an AI consultant.
""")

# =====================================================
# STATUS + HOW TO USE
# =====================================================
c1, c2, c3 = st.columns(3)
loaded = cu.DATA_KEY in st.session_state
c1.metric("Data", "loaded" if loaded else "not loaded",
          st.session_state.get(cu.NAME_KEY, "use the sidebar") if loaded else "use the sidebar", delta_color="off")
c2.metric("Page results saved this session", len(cu.get_results()))
c3.metric("Local LLM (Ollama llama3.2)", "connected" if cu.llm_available() else "not running")

st.header("How to use")
st.markdown("""
1. **Load data once** in the sidebar: upload your Calgary file (e.g. `preprocessor.xlsx`) or click
   *Use synthetic demo data*. Open **Columns, units & cleaning** and check the mapping (date, consumption,
   temperature, price, gas price, unit, City filter, cleaning). Every page uses this one setup.
2. **Work through the sections in the sidebar, top to bottom.** Each page saves its key results.
3. **Finish with section 6.** The AI Decision Engine and the AI Consultant read the results of every page
   you ran in this session.

**Recommended path:** 1 → 2 → 3 → 4 → 8 → 9 → 12 → 13 → 14 → 15 → 16 → 18 → 19 (use 17 whenever you have documents)
""")

st.header("Platform sections")
sections = pd.DataFrame([
    ("1 · Data", "1. Data Preview & Quality", "What is in the data? Gaps, duplicates, outliers, quality score"),
    ("1 · Data", "2. Energy Database & SQL", "SQLite database, SQL analytics, add meter readings, reload as data source"),
    ("2 · Explore", "3. EDA & Market Statistics", "Consumption, price and gas statistics, distributions, drivers"),
    ("2 · Explore", "4. Seasonal & Time-Series Patterns", "Hourly / weekly / monthly / seasonal patterns, decomposition"),
    ("2 · Explore", "5. Statistical Analysis", "Correlation, regression, ANOVA, chi-square with effect sizes"),
    ("2 · Explore", "6. Feature Engineering", "Calendar, weather, lag and rolling features; spike flags; export"),
    ("3 · Market & Risk", "7. Anomalies, Spark Spread & Volatility", "Price anomalies, spark spread, volatility, price VaR"),
    ("4 · Models", "8. Weather-Normalized Regression", "HDD/CDD baseline, ASHRAE Guideline 14, avoided energy"),
    ("4 · Models", "9. ML Hyperparameter Tuning", "Random Forest / Gradient Boosting / XGBoost / Ridge, time-series CV"),
    ("4 · Models", "10. ML Forecasting", "Day-ahead / week-ahead ML forecast vs naive benchmark"),
    ("4 · Models", "11. Time-Series Forecasting", "ARIMA / SARIMA / Holt-Winters / optional LSTM on daily consumption"),
    ("4 · Models", "12. Consumption Prediction & What-If", "P10/P50/P90 prediction, what-if, temperature response"),
    ("5 · Facility", "13. Predictive Maintenance", "Health index, drift, anomalies, days to high risk, actions"),
    ("5 · Facility", "14. Process Optimization", "Load-shifting LP, efficiency measures, weather scenarios"),
    ("5 · Facility", "15. Design Optimization & Scale-Up", "Bayesian PV + battery sizing, uncertainty, portfolio"),
    ("5 · Facility", "16. Financial & Procurement Risk", "Cost budget, fixed vs floating price, VaR, NPV / IRR"),
    ("6 · AI", "17. AI EngInsight", "Document assistant: summary, deep analysis, chat with your reports"),
    ("6 · AI", "18. AI Decision Engine & Report", "Engineering report (PDF), next-best actions, savings-trial design"),
    ("6 · AI", "19. AI Energy Consultant", "Tool-calling agent: database, models, page results, documents"),
], columns=["Section", "Page", "Answers"])
st.dataframe(sections, hide_index=True)

st.divider()

# =====================================================
# PROJECT VISION
# =====================================================
st.header("Project vision")
st.write("""
Modern energy systems generate large amounts of data, but traditional dashboards only display it and still
need experts to interpret the results. This platform builds AI directly into the analytical workflow and
moves beyond prediction by:

- understanding analytical outputs
- explaining business impact
- detecting operational and equipment risks
- evaluating financial consequences
- generating optimization recommendations

The goal is an AI-powered energy intelligence system that supports faster, better-informed decisions for
Calgary energy consumers.
""")

col1, col2 = st.columns([2, 1])
with col1:
    st.header("Executive summary")
    st.write("""
The platform converts raw Calgary energy data into decision-ready intelligence:

- energy data processing and quality control
- SQL business intelligence (SQLite)
- statistical analytics
- machine-learning prediction and time-series forecasting
- market and financial risk evaluation
- predictive maintenance
- process and design optimization
- AI interpretation, reports and consultant
""")
with col2:
    st.header("Business impact")
    st.success("""
Improve consumption forecasting

Reduce operational uncertainty

Detect equipment and market risks early

Optimize energy costs and procurement

Improve operational efficiency

Support investment and strategic planning
""")

st.divider()

# =====================================================
# MODULES
# =====================================================
st.header("Platform intelligence modules")
modules = [
    ("Data Engineering", "ETL, mapping, cleaning, validation, quality score (pages 1-2)"),
    ("SQL Analytics", "SQLite energy database, KPI and market queries, data entry (page 2)"),
    ("EDA & Statistics", "Patterns, seasonality, correlation, ANOVA, chi-square (pages 3-6)"),
    ("Market & Risk", "Price anomalies, spark spread, volatility, VaR (page 7)"),
    ("Machine Learning", "Weather regression, tuned RF / GB / XGBoost, P10-P90 prediction (pages 8, 9, 12)"),
    ("Forecasting", "ML day/week-ahead, ARIMA, SARIMA, Holt-Winters, optional LSTM (pages 10-11)"),
    ("Facility Decisions", "Predictive maintenance, load shifting, PV + battery design (pages 13-15)"),
    ("Financial Intelligence", "Budget, fixed vs floating procurement, NPV / IRR (page 16)"),
    ("AI EngInsight", "Document summary, deep analysis and chat with your reports (page 17)"),
    ("AI Decision Engine", "LLM report, next-best actions, savings-trial design (page 18)"),
    ("AI Consultant", "Tool-calling agent over database, models, results and documents (page 19)"),
]
cols = st.columns(5)
for i, (title, text) in enumerate(modules):
    with cols[i % 5]:
        st.info(title)
        st.caption(text)

st.divider()

# =====================================================
# WORKFLOW
# =====================================================
st.header("End-to-end AI energy intelligence workflow")
st.markdown("Raw energy data → Data engineering → Analytics → Prediction → Risk analysis → Optimization → "
            "LLM interpretation → Executive decision support")
if PIC2.exists():
    st.image(str(PIC2), caption="AI Energy Intelligence Workflow Architecture")
workflow = pd.DataFrame({
    "Stage": ["1. Data ingestion", "2. Data preparation", "3. Analytics", "4. Predictive intelligence",
              "5. Decision intelligence", "6. AI interpretation"],
    "Description": ["Load consumption, weather, price and gas data (upload or SQLite)",
                    "Map columns, set units, remove duplicates and outliers, build features",
                    "EDA, SQL analytics, seasonal patterns, statistical tests, market risk",
                    "Weather-normalized regression, tuned ML, ML and time-series forecasts",
                    "Predictive maintenance, optimization, design and financial analysis",
                    "LLM report, next-best actions, trial design, AI consultant"],
    "Pages": ["1-2", "1, 6", "2-7", "8-12", "13-16", "17-19"]})
st.dataframe(workflow, hide_index=True)

st.header("Architecture")
st.code("""Calgary data (consumption, weather, price, gas)  ->  sidebar setup (mapping, units, cleaning)
        |                                                    |
        v                                                    v
  SQLite energy database  <------------------  shared prepared dataset (calgary_utils.prepare)
        |
        v
EDA / seasonal / statistics / features  ->  market risk (anomalies, spark spread, VaR)
        |
        v
Models: weather regression, tuned ML, ML + time-series forecasts, P10/P50/P90 prediction
        |
        v
Facility decisions: predictive maintenance, load shifting, PV + battery design, finance
        |
        v
Results registry  ->  AI Decision Engine (report, actions, trials)  ->  AI Consultant (tools + docs)
""", language="text")

st.divider()

# =====================================================
# SQL, LLM AND AGENT LAYERS
# =====================================================
st.header("Business SQL analytics engine")
s1, s2 = st.columns(2)
s1.info("""Database architecture

• SQLite energy database (calgary_energy.db)
• Table energy_readings with calendar and weather fields
• Validated form for new meter readings
• Database can be reloaded as the data source for every page""")
s2.info("""Analytical capabilities

• Consumption KPIs and peak days
• Hourly, weekday and seasonal profiles
• Temperature-band analysis
• Price: peak / cheap hours, weekend discount, peak index
• Monthly energy cost""")

st.header("Large Language Model (LLM) interpretation engine")
st.write("""
The LLM layer (local Llama 3.2 through Ollama) is the bridge between technical analytics and business
decisions. Every page can send its results to the LLM, which explains what happened, why, which risks exist,
the financial impact, and which actions to consider. Prompts contain only computed results, and the model is
instructed not to invent numbers. Without Ollama, the platform still runs, and pages 18-19 use transparent
rule-based fallbacks.
""")

st.header("AI agent architecture")
a1, a2, a3 = st.columns(3)
a1.info("""Analytics pages (1-16)

Data, SQL, statistics, forecasting, maintenance, optimization and finance. Each page saves its results to a
shared results registry.""")
a2.info("""Decision engine (page 18)

Reads every saved result and produces an engineering report (PDF), ranked next-best actions and a
savings-trial design.""")
a3.info("""AI consultant (page 19)

Tool-calling agent: SQL database, prediction model, saved page results, market summary, document search
and trial sizing. Quotes the numbers it used.""")

st.divider()

# =====================================================
# TECHNOLOGY STACK + SKILLS
# =====================================================
st.header("Technology stack")
st.dataframe(pd.DataFrame({
    "Layer": ["Programming", "Data analytics", "Machine learning", "Forecasting", "Optimization", "Database",
              "AI / LLM", "Reporting", "Deployment"],
    "Technology": ["Python", "Pandas, NumPy, SciPy, Statsmodels, Plotly", "Scikit-learn, XGBoost",
                   "ML forecasting, ARIMA, SARIMA, Holt-Winters, optional LSTM (TensorFlow)",
                   "Linear programming (SciPy HiGHS), Bayesian optimization", "SQLite",
                   "Ollama (Llama 3.2), tool calling, TF-IDF document retrieval", "ReportLab PDF", "Streamlit"]}),
    hide_index=True)

st.header("Technical skills & problem-solving approach")
skills = {
    "Data engineering": "Problem: energy datasets have gaps, duplicates, irregular timestamps and noise.\n\n"
                        "Solution: column mapping, unit handling, duplicate and outlier cleaning, time-based "
                        "features, a data-quality score and plausibility checks.",
    "SQL database engineering": "Problem: data must be stored, queried and extended reliably.\n\n"
                                "Solution: SQLite energy database, read-only SQL analytics, validated data entry "
                                "and a closed loop back into every model.",
    "Exploratory data analysis": "Problem: consumption hides daily, weekly and seasonal patterns.\n\n"
                                 "Solution: seasonal profiles, heatmaps, STL decomposition, correlation and "
                                 "significance tests with effect sizes.",
    "Machine learning": "Problem: consumption depends on weather, calendar and history.\n\n"
                        "Solution: weather-normalized regression (ASHRAE Guideline 14), tuned ML with time-series "
                        "cross-validation, P10/P50/P90 prediction, comparison with naive benchmarks.",
    "Time-series forecasting": "Problem: planning needs reliable forecasts.\n\n"
                               "Solution: leakage-free ML forecasts, ARIMA, SARIMA, Holt-Winters and optional LSTM, "
                               "each scored against a naive benchmark.",
    "Optimization & decision support": "Problem: organizations need actions, not only charts.\n\n"
                                       "Solution: load-shifting linear programs, Bayesian PV + battery sizing, "
                                       "predictive maintenance, procurement risk, NPV / IRR, AI decision engine.",
    "AI and LLM engineering": "Problem: results still need expert interpretation.\n\n"
                              "Solution: structured JSON prompts, rule-based fallbacks, a tool-calling consultant and "
                              "document retrieval.",
}
for title, text in skills.items():
    with st.expander(title):
        st.write(text)

st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Platform pages", "19")
k2.metric("Forecasting approaches", "6")
k3.metric("Optimization methods", "3", "LP, Bayesian, Monte Carlo", delta_color="off")
k4.metric("AI layers", "3", "interpretation, decisions, consultant", delta_color="off")

st.caption("Data note: results are only as reliable as the input data. Page 1 flags quality issues "
           "(irregular timestamps, implausible temperatures, missing daily load pattern) before modelling.")
