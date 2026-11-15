"""
AI Calgary Energy Decision (Decision Intelligence Platform) - entry point.

Run:  streamlit run Home.py

This file is the router: it sets the page config, draws the shared data setup in the sidebar
(used by every page) and the navigation grouped into the platform's sections.
"""
import streamlit as st

import calgary_utils as cu

st.set_page_config(page_title="AI Calgary Energy Decision", layout="wide")

S = "sections/"
PAGES = {
    "Start": [
        st.Page(S + "00_Home.py", title="0. Home & How to Use", default=True),
    ],
    "1 · Data": [
        st.Page(S + "01_Data_Preview_Quality.py", title="1. Data Preview & Quality"),
        st.Page(S + "02_Energy_Database_SQL.py", title="2. Energy Database & SQL"),
    ],
    "2 · Explore": [
        st.Page(S + "03_EDA_Market_Statistics.py", title="3. EDA & Market Statistics"),
        st.Page(S + "04_Seasonal_Time_Series.py", title="4. Seasonal & Time-Series Patterns"),
        st.Page(S + "05_Statistical_Analysis.py", title="5. Statistical Analysis"),
        st.Page(S + "06_Feature_Engineering.py", title="6. Feature Engineering"),
    ],
    "3 · Market & Risk": [
        st.Page(S + "07_Anomaly_Spark_Volatility.py", title="7. Anomalies, Spark Spread & Volatility"),
    ],
    "4 · Models & Forecasting": [
        st.Page(S + "08_Weather_Normalized_Regression.py", title="8. Weather-Normalized Regression"),
        st.Page(S + "09_ML_Hyperparameter_Tuning.py", title="9. ML Hyperparameter Tuning"),
        st.Page(S + "10_ML_Forecasting.py", title="10. ML Forecasting"),
        st.Page(S + "11_Time_Series_Forecasting.py", title="11. Time-Series Forecasting"),
        st.Page(S + "12_Consumption_Prediction_WhatIf.py", title="12. Consumption Prediction & What-If"),
    ],
    "5 · Facility Decisions": [
        st.Page(S + "13_Predictive_Maintenance.py", title="13. Predictive Maintenance"),
        st.Page(S + "14_Process_Optimization.py", title="14. Process Optimization"),
        st.Page(S + "15_Design_Optimization_ScaleUp.py", title="15. Design Optimization & Scale-Up"),
        st.Page(S + "16_Financial_Risk_Analysis.py", title="16. Financial & Procurement Risk"),
    ],
    "6 · AI Decision Support": [
        st.Page(S + "17_AI_EngInsight.py", title="17. AI EngInsight"),
        st.Page(S + "18_AI_Decision_Engine.py", title="18. AI Decision Engine & Report"),
        st.Page(S + "19_AI_Energy_Consultant.py", title="19. AI Energy Consultant (Agent + Docs)"),
    ],
}

nav = st.navigation(PAGES)
cu.sidebar_data_setup()
nav.run()
