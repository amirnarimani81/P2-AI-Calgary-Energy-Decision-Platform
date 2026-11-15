import io
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calgary_utils as cu

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

cu.page_header("ML Hyperparameter Tuning", "Section 4 · Models & Forecasting", """
Tunes **Random Forest, Gradient Boosting, XGBoost and Ridge** with **time-series cross-validation**
(every fold trains on the past and validates on the future, so nothing leaks from the future). The most
recent part of the data is held out as a final test and compared with a naive benchmark.
""")
raw, mapping, data = cu.require_data()
u = cu.unit_label(mapping)

c1, c2, c3, c4 = st.columns(4)
use_lags = c1.toggle("Use history features (lags)", True,
                     help="On = short-term forecasting model. Off = driver-only model (weather + calendar), "
                          "used for what-if and optimization.")
max_rows = c2.slider("Most recent rows", 1000, max(len(data), 1001), min(8760, len(data)), 500) if len(data) > 1500 else len(data)
n_splits = c3.slider("CV folds", 3, 8, 4)
test_pct = c4.slider("Hold-out test %", 10, 30, 20)
search_type = st.radio("Search", ["Randomized (fast)", "Grid (exhaustive)"], horizontal=True)
n_iter = st.slider("Randomized iterations", 5, 60, 15) if search_type.startswith("Rand") else None

df = (cu.add_lag_features(data) if use_lags else data.copy()).tail(max_rows)
all_feats = cu.base_features(df) + ([c for c in df if c.startswith(("lag_", "roll_"))] if use_lags else [])
features = st.multiselect("Features", all_feats, default=all_feats)
if not features:
    st.stop()
X, y = df[features].astype(float), df["consumption"].astype(float)
cut = int(len(X) * (1 - test_pct / 100))
X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

model_name = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Ridge"] + (["XGBoost"] if HAS_XGB else []))
st.subheader("Search space")
c = st.columns(4)
if model_name == "Random Forest":
    est = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = {"n_estimators": c[0].multiselect("n_estimators", [100, 200, 400], [100, 200]),
            "max_depth": c[1].multiselect("max_depth", [6, 10, 16, None], [10, 16]),
            "min_samples_leaf": c[2].multiselect("min_samples_leaf", [1, 3, 5, 10], [1, 5]),
            "max_features": c[3].multiselect("max_features", ["sqrt", 0.5, 1.0], ["sqrt", 1.0])}
elif model_name == "Gradient Boosting":
    est = GradientBoostingRegressor(random_state=42)
    grid = {"n_estimators": c[0].multiselect("n_estimators", [100, 200, 400], [100, 200]),
            "learning_rate": c[1].multiselect("learning_rate", [0.02, 0.05, 0.1, 0.2], [0.05, 0.1]),
            "max_depth": c[2].multiselect("max_depth", [2, 3, 4, 6], [3, 4]),
            "subsample": c[3].multiselect("subsample", [0.7, 0.8, 1.0], [0.8, 1.0])}
elif model_name == "XGBoost":
    est = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    grid = {"n_estimators": c[0].multiselect("n_estimators", [200, 400, 800], [200, 400]),
            "learning_rate": c[1].multiselect("learning_rate", [0.02, 0.05, 0.1], [0.05, 0.1]),
            "max_depth": c[2].multiselect("max_depth", [3, 4, 6, 8], [4, 6]),
            "subsample": c[3].multiselect("subsample", [0.7, 0.8, 1.0], [0.8])}
else:
    est = make_pipeline(StandardScaler(), Ridge())
    grid = {"ridge__alpha": c[0].multiselect("alpha", [0.01, 0.1, 1.0, 10.0, 100.0], [0.1, 1.0, 10.0])}
if any(len(v) == 0 for v in grid.values()):
    st.warning("Pick at least one value for every hyperparameter.")
    st.stop()
n_combos = int(np.prod([len(v) for v in grid.values()]))
st.caption(f"{n_combos} combinations × {n_splits} folds | train {len(X_tr):,} rows, test {len(X_te):,} rows "
           f"({X_te.index[0]:%Y-%m-%d} → {X_te.index[-1]:%Y-%m-%d})")

def make_search(n_jobs):
    common = dict(scoring="neg_root_mean_squared_error", cv=TimeSeriesSplit(n_splits=n_splits), n_jobs=n_jobs,
                  return_train_score=True)
    return (RandomizedSearchCV(est, grid, n_iter=n_iter, random_state=42, **common)
            if search_type.startswith("Rand") and n_combos > n_iter else GridSearchCV(est, grid, **common))


if st.button("Run hyperparameter search", type="primary"):
    t0 = time.time()
    with st.spinner("Searching..."):
        try:
            search = make_search(-1)          # all CPU cores
            search.fit(X_tr, y_tr)
        except Exception as e:                # e.g. Windows process-pool errors: retry on one core
            st.caption(f"Parallel search failed ({type(e).__name__}); retried on a single core.")
            if hasattr(est, "n_jobs"):
                est.set_params(n_jobs=1)
            search = make_search(1)
            search.fit(X_tr, y_tr)
    st.session_state["tuning_result"] = {"search": search, "model_name": model_name, "features": features,
                                         "use_lags": use_lags, "seconds": time.time() - t0}

res = st.session_state.get("tuning_result")
if not res or res["model_name"] != model_name or res["features"] != features:
    st.info("Press **Run hyperparameter search**.")
    st.stop()

search, best = res["search"], res["search"].best_estimator_
st.success(f"Done in {res['seconds']:.0f} s. Best CV RMSE = {-search.best_score_:,.2f} {u}")
st.json(search.best_params_)
cvres = pd.DataFrame(search.cv_results_)
cvres["CV RMSE"], cvres["Train RMSE"] = -cvres["mean_test_score"], -cvres["mean_train_score"]
cvres["Overfit ratio"] = (cvres["CV RMSE"] / cvres["Train RMSE"]).round(2)
pcols = [c for c in cvres if c.startswith("param_")]
table = cvres[pcols + ["CV RMSE", "std_test_score", "Train RMSE", "Overfit ratio"]].sort_values("CV RMSE")
table = table.rename(columns=lambda c: c.replace("param_", ""))
table[[c[6:] for c in pcols]] = table[[c[6:] for c in pcols]].astype(str)
st.subheader("All candidates")
st.dataframe(table.round(3), hide_index=True)
st.caption("Overfit ratio = CV RMSE / train RMSE; well above 1.5 means the model memorizes training data.")
if pcols:
    p = st.selectbox("Effect of one hyperparameter", pcols, format_func=lambda c: c[6:])
    eff = cvres.assign(val=cvres[p].astype(str)).groupby("val")[["CV RMSE", "Train RMSE"]].min()
    st.plotly_chart(px.line(eff, markers=True, title=f"Best RMSE by {p[6:]}"))

st.subheader("Final hold-out test")
pred = best.predict(X_te)
m = cu.fit_metrics(y_te, pred)
cols = st.columns(5)
for i, (k, v) in enumerate(m.items()):
    cols[i].metric(k, v)
naive_col = next((c for c in ["lag_24h", "lag_7d", "lag_12m", "lag_1d", "lag_1h"] if c in X_te), None)
skill = None
if naive_col:
    naive = cu.fit_metrics(y_te, X_te[naive_col])
    skill = (1 - m["RMSE"] / naive["RMSE"]) * 100
    st.caption(f"Naive benchmark ({naive_col}: same time one period earlier) RMSE {naive['RMSE']:,.2f}. "
               f"Model skill vs naive: **{skill:.1f}%** " + ("(adds value)" if skill > 0 else "(NOT better than naive)"))
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_te.index, y=y_te, name="Actual", line=dict(width=1)))
fig.add_trace(go.Scatter(x=y_te.index, y=pred, name="Predicted", line=dict(width=1, dash="dot")))
fig.update_layout(title="Hold-out: actual vs predicted", height=380, yaxis_title=u)
st.plotly_chart(fig)
imp = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=42, scoring="neg_root_mean_squared_error")
imp_s = pd.Series(imp.importances_mean, index=features).sort_values()
st.plotly_chart(px.bar(imp_s, orientation="h", title="Permutation importance (RMSE increase when shuffled)"))

st.session_state["tuned_model"] = {"model": best, "features": features, "use_lags": res["use_lags"],
                                   "metrics": m, "model_name": model_name}
buf = io.BytesIO()
joblib.dump(st.session_state["tuned_model"], buf)
st.download_button("Download tuned model (.pkl)", buf.getvalue(),
                   f"calgary_{model_name.replace(' ', '_').lower()}_tuned.pkl")
if not res["use_lags"]:
    st.info("This driver-only model is now used by **14. Process Optimization** for weather scenarios.")

cu.ai_interpretation("ML hyperparameter tuning", {
    "model": model_name, "uses_history_features": res["use_lags"], "best_params": search.best_params_,
    "cv_rmse": -search.best_score_, "holdout_metrics": m, "skill_vs_naive_pct": skill,
    "top_features": imp_s.sort_values(ascending=False).head(6).round(3).to_dict(),
    "overfit_ratio_best": float(cvres.sort_values("CV RMSE")["Overfit ratio"].iloc[0])},
    "1) Model reliability and overfitting 2) Which drivers matter 3) Is accuracy good enough for "
    "operations / budgeting 4) How to improve (data, features, model).")
