import pandas as pd
import plotly.express as px
import streamlit as st

import calgary_utils as cu

cu.page_header("Energy Database & SQL", "Section 1 · Data", """
Loads the cleaned data into a **SQLite energy database** (`calgary_energy.db`, table
`energy_readings`) and runs SQL analytics: consumption KPIs plus market queries such as peak and cheap
price hours, weekend discount and peak index. You can **add new meter readings** through a validated
form and reload the database as the data source for every page.
""")
raw, mapping, data = cu.require_data()
T = cu.DB_TABLE

# -------------------------------------------------------
# 1. LOAD
# -------------------------------------------------------
st.subheader("1. Load the cleaned dataset into SQLite")
if st.button("Write current dataset to database", type="primary"):
    n = cu.db_write_prepared(data, st.session_state.get(cu.NAME_KEY, "upload"))
    st.success(f"Wrote {n:,} rows to `{cu.DB_PATH.name}` (table `{T}`, existing table replaced).")
if not cu.db_exists():
    st.info("No database yet. Click **Write current dataset to database** to create it.")
    st.stop()

info = cu.db_query(f"SELECT COUNT(*) AS rows_, MIN(datetime) AS first, MAX(datetime) AS last, "
                   f"SUM(source LIKE 'manual%') AS manual FROM {T}").iloc[0]
k = st.columns(4)
k[0].metric("Rows", f"{int(info['rows_']):,}")
k[1].metric("First reading", str(info["first"])[:10])
k[2].metric("Last reading", str(info["last"])[:10])
k[3].metric("Manually added", int(info["manual"] or 0))

# -------------------------------------------------------
# 2. SQL ANALYTICS
# -------------------------------------------------------
st.subheader("2. SQL analytics")
cols = cu.db_query(f"PRAGMA table_info({T})")["name"].tolist()
Q = {
    "Top 10 highest-consumption days": f"""
SELECT DATE(datetime) AS day, ROUND(AVG(consumption), 1) AS avg_per_reading,
       ROUND(MAX(consumption), 1) AS peak, COUNT(*) AS readings
FROM {T} GROUP BY day ORDER BY avg_per_reading DESC LIMIT 10""",
    "Monthly consumption & weather": f"""
SELECT strftime('%Y-%m', datetime) AS month, ROUND(AVG(consumption), 1) AS avg_per_reading,
       ROUND(MAX(consumption), 1) AS peak{", ROUND(AVG(temperature), 1) AS avg_temp_c" if "temperature" in cols else ""}
FROM {T} GROUP BY month ORDER BY month""",
    "Average consumption by hour of day": f"""
SELECT hour, ROUND(AVG(consumption), 1) AS avg_consumption, COUNT(*) AS readings
FROM {T} GROUP BY hour ORDER BY hour""",
    "Weekday vs weekend consumption": f"""
SELECT CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
       ROUND(AVG(consumption), 1) AS avg_consumption, COUNT(*) AS readings
FROM {T} GROUP BY is_weekend""",
    "Seasonal summary": f"""
SELECT season, ROUND(AVG(consumption), 1) AS avg_consumption, ROUND(MAX(consumption), 1) AS peak,
       COUNT(*) AS readings FROM {T} GROUP BY season ORDER BY avg_consumption DESC""",
    "Readings above 3 standard deviations": f"""
WITH s AS (SELECT AVG(consumption) AS m,
           AVG(consumption * consumption) - AVG(consumption) * AVG(consumption) AS v FROM {T})
SELECT datetime, ROUND(consumption, 1) AS consumption, ROUND((consumption - m) / SQRT(v), 2) AS z_score
FROM {T}, s WHERE ABS(consumption - m) > 3 * SQRT(v) ORDER BY z_score DESC LIMIT 25""",
    "Recently added manual readings": f"SELECT * FROM {T} WHERE source LIKE 'manual%' ORDER BY datetime DESC LIMIT 25",
}
if "temperature" in cols:
    Q["Consumption by temperature band"] = f"""
SELECT CASE WHEN temperature < -20 THEN '1. below -20 C' WHEN temperature < -10 THEN '2. -20 to -10 C'
            WHEN temperature < 0 THEN '3. -10 to 0 C' WHEN temperature < 10 THEN '4. 0 to 10 C'
            WHEN temperature < 20 THEN '5. 10 to 20 C' ELSE '6. above 20 C' END AS band,
       ROUND(AVG(consumption), 1) AS avg_consumption, COUNT(*) AS readings
FROM {T} GROUP BY band ORDER BY band"""
if "price" in cols:
    Q.update({
        "Market: 5 most expensive hours": f"""
SELECT hour, ROUND(AVG(price), 2) AS avg_price, COUNT(*) AS readings
FROM {T} GROUP BY hour ORDER BY avg_price DESC LIMIT 5""",
        "Market: 5 cheapest hours": f"""
SELECT hour, ROUND(AVG(price), 2) AS avg_price, COUNT(*) AS readings
FROM {T} GROUP BY hour ORDER BY avg_price ASC LIMIT 5""",
        "Market: weekend discount": f"""
SELECT ROUND(AVG(CASE WHEN is_weekend = 0 THEN price END), 2) AS weekday_price,
       ROUND(AVG(CASE WHEN is_weekend = 1 THEN price END), 2) AS weekend_price,
       ROUND((1 - AVG(CASE WHEN is_weekend = 1 THEN price END) /
                  AVG(CASE WHEN is_weekend = 0 THEN price END)) * 100, 2) AS weekend_discount_pct
FROM {T}""",
        "Market: peak index (07-09 & 17-19 vs 02-05)": f"""
SELECT ROUND(AVG(CASE WHEN hour IN (7,8,9,17,18,19) THEN price END), 2) AS peak_price,
       ROUND(AVG(CASE WHEN hour BETWEEN 2 AND 5 THEN price END), 2) AS offpeak_price,
       ROUND(AVG(CASE WHEN hour IN (7,8,9,17,18,19) THEN price END) /
             AVG(CASE WHEN hour BETWEEN 2 AND 5 THEN price END), 3) AS peak_index
FROM {T}""",
        "Cost: monthly energy cost (consumption × price)": f"""
SELECT strftime('%Y-%m', datetime) AS month, ROUND(SUM(consumption * price), 0) AS cost_units,
       ROUND(SUM(consumption * price) / SUM(consumption), 2) AS weighted_price
FROM {T} GROUP BY month ORDER BY month""",
    })

choice = st.selectbox("Pre-built query", list(Q) + ["Custom SQL"])
sql = st.text_area("SQL (read-only: SELECT / WITH)", value=Q.get(choice, f"SELECT * FROM {T} LIMIT 20").strip(),
                   height=160)
if st.button("Run query"):
    if not sql.strip().lower().startswith(("select", "with", "pragma")):
        st.error("Only SELECT / WITH queries are allowed. Use the form below to add data.")
    else:
        try:
            st.session_state["sql_result"] = (sql, cu.db_query(sql))
        except Exception as e:
            st.error(f"SQL error: {e}")

if "sql_result" in st.session_state:
    q, res = st.session_state["sql_result"]
    st.dataframe(res, hide_index=True)
    num = res.select_dtypes("number").columns.tolist()
    if len(res.columns) >= 2 and num and len(res) > 1:
        x = res.columns[0]
        y = st.selectbox("Chart column", [c for c in num if c != x] or num)
        fig = px.bar(res, x=x, y=y) if len(res) <= 40 else px.line(res, x=x, y=y)
        st.plotly_chart(fig)
    st.download_button("Download result (CSV)", res.to_csv(index=False), "query_result.csv")
    if st.button("Explain this result with AI"):
        with st.spinner("Asking the local LLM..."):
            st.markdown(cu.call_llm(
                f"You are an energy analyst for a Calgary site. SQL:\n{q}\n\nResult (first 40 rows):\n"
                f"{res.head(40).to_string()}\n\nGive 3-5 quantified insights and one recommended action. "
                "Use only these numbers."))

# -------------------------------------------------------
# 3. ADD NEW READING
# -------------------------------------------------------
st.subheader("3. Add a new meter reading")
stats = cu.db_query(f"SELECT AVG(consumption) AS m, MAX(consumption) AS mx FROM {T}").iloc[0]
with st.form("add_reading"):
    c1, c2, c3 = st.columns(3)
    d = c1.date_input("Date")
    t = c1.time_input("Time", value=pd.Timestamp("2026-01-01 08:00").time(), step=3600)
    cons = c2.number_input(f"Consumption ({cu.unit_label(mapping)})", min_value=0.0, value=float(round(stats["m"], 1)))
    temp = c2.number_input("Temperature (°C)", value=-5.0) if "temperature" in cols else None
    price = c3.number_input("Electricity price ($/MWh)", value=100.0) if "price" in cols else None
    note = c3.text_input("Note (e.g. 'after boiler repair')")
    submitted = st.form_submit_button("Validate & save")
if submitted:
    ts = pd.Timestamp.combine(d, t)
    problems, warnings = [], []
    if cons <= 0:
        problems.append("Consumption must be positive.")
    if cons > 3 * stats["mx"]:
        problems.append(f"More than 3× the highest reading on record ({stats['mx']:,.1f}).")
    elif cons > stats["mx"]:
        warnings.append("Above the highest reading on record; check the meter value.")
    if temp is not None and not -45 <= temp <= 40:
        problems.append("Temperature outside Calgary's realistic range (-45 to 40 °C).")
    if ts > pd.Timestamp.now() + pd.Timedelta(hours=1):
        problems.append("Timestamp is in the future.")
    dup = cu.db_query(f"SELECT consumption FROM {T} WHERE datetime = ?", (ts.strftime("%Y-%m-%d %H:%M:%S"),))
    if not dup.empty:
        warnings.append(f"A reading already exists at {ts} ({dup.iloc[0, 0]:,.1f}); it will be replaced.")
    for w in warnings:
        st.warning(w)
    for p in problems:
        st.error(p)
    if not problems:
        cu.db_insert_reading(ts, cons, temp, price, note)
        st.success(f"Saved reading for {ts:%Y-%m-%d %H:%M}.")

# -------------------------------------------------------
# 4. DATABASE AS DATA SOURCE
# -------------------------------------------------------
st.subheader("4. Use the database as the data source")
st.caption("Loads `energy_readings` (including manual readings) as the shared dataset so every page "
           "retrains on the latest data. Cleaning was already applied when the table was written.")
if st.button("Load database into the app"):
    unit = mapping.get("unit")
    db = cu.db_to_dataset()
    cu.set_dataset(db, f"SQLite: {cu.DB_PATH.name}")
    st.session_state[cu.MAP_KEY] = {
        "date": "Date", "consumption": "Consumption",
        "temperature": "Temperature_C" if "Temperature_C" in db else None,
        "price": "Electricity_Price_CAD_MWh" if "Electricity_Price_CAD_MWh" in db else None,
        "gas_price": "Gas_Price_CAD_GJ" if "Gas_Price_CAD_GJ" in db else None,
        "unit": unit, "site_col": None, "site_val": None, "clean": "None", "clean_k": 6.0, "clip_temp": False}
    st.rerun()

cu.save_result("Energy database", {"rows": int(info["rows_"]), "first": info["first"], "last": info["last"],
                                   "manual_rows": int(info["manual"] or 0)})
