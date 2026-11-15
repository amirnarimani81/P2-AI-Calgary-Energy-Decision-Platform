import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from scipy import stats

import calgary_utils as cu

cu.page_header("Statistical Analysis", "Section 2 · Explore", """
Which factors **significantly** affect Calgary consumption? Correlation (Pearson & Spearman),
simple and multiple regression, **ANOVA** across groups and **chi-square** between categories, each
with p-values and effect sizes, so significance and practical importance are both visible.
""")
raw, mapping, data = cu.require_data()
view = cu.raw_view(raw, mapping)
df = data.join(view.drop(columns=[c for c in mapping.values() if isinstance(c, str) and c in view]),
               how="left", rsuffix="_raw")
numeric = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 2]
categorical = [c for c in df.columns if (df[c].dtype == object or str(df[c].dtype).startswith(("category", "str")))
               and 1 < df[c].nunique() <= 20] + ["is_weekend", "month", "weekday"]
categorical = list(dict.fromkeys(c for c in categorical if c in df))

analysis = st.selectbox("Analysis", ["Correlation", "Regression", "ANOVA", "Chi-square"])
results = {"analysis": analysis}

if analysis == "Correlation":
    target = st.selectbox("Target", numeric, index=numeric.index("consumption"))
    rows = []
    for c in numeric:
        if c == target:
            continue
        pair = df[[c, target]].dropna()
        if len(pair) < 10:
            continue
        r, p = stats.pearsonr(pair[c], pair[target])
        rho, ps = stats.spearmanr(pair[c], pair[target])
        rows.append({"variable": c, "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": ps,
                     "strength": "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"})
    tab = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
    st.dataframe(tab.round(4), hide_index=True)
    st.caption("With thousands of rows almost every correlation is 'significant' (p < 0.05); judge importance "
               "by |r|. A big gap between Pearson and Spearman means a non-linear or outlier-driven relationship.")
    st.plotly_chart(px.bar(tab.head(15), x="pearson_r", y="variable", orientation="h", title=f"Correlation with {target}"))
    leak = tab[tab["pearson_r"].abs() > 0.95]
    if len(leak):
        st.warning("Near-duplicates of the target (|r| > 0.95), do not use as model inputs: " + ", ".join(leak["variable"]))
    results.update({"target": target, "top": tab.head(10).round(4).to_dict("records")})

elif analysis == "Regression":
    target = st.selectbox("Target", numeric, index=numeric.index("consumption"))
    default = [c for c in ["HDD", "CDD", "is_weekend", "price"] if c in df and c != target]
    feats = st.multiselect("Predictors", [c for c in numeric + ["is_weekend"] if c != target], default=default)
    level = st.radio("Data level", ["Daily (recommended)", "Original readings"], horizontal=True)
    if feats:
        d = df[feats + [target]].dropna()
        if level.startswith("Daily"):
            d = d.resample("D").mean().dropna()
        rows = []
        for f in feats:
            res = stats.linregress(d[f], d[target])
            rows.append({"predictor": f, "slope": res.slope, "R2": res.rvalue ** 2, "p_value": res.pvalue})
        st.write("**Simple regressions (one predictor at a time)**")
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)
        ols = sm.OLS(d[target], sm.add_constant(d[feats])).fit()
        coef = pd.DataFrame({"coef": ols.params, "std_err": ols.bse, "p_value": ols.pvalues,
                             "CI low": ols.conf_int()[0], "CI high": ols.conf_int()[1]}).round(4)
        st.write(f"**Multiple regression**: R² = {ols.rsquared:.3f}, adjusted R² = {ols.rsquared_adj:.3f}, n = {len(d):,}")
        st.dataframe(coef)
        vif = {f: round(1 / (1 - sm.OLS(d[f], sm.add_constant(d[[x for x in feats if x != f]])).fit().rsquared), 2)
               for f in feats} if len(feats) > 1 else {}
        if vif:
            st.caption("Variance inflation factors (VIF > 5 = predictors overlap): " +
                       ", ".join(f"{k} {v}" for k, v in vif.items()))
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.scatter(x=ols.fittedvalues, y=ols.resid, opacity=0.4, labels={"x": "fitted", "y": "residual"},
                                   title="Residuals vs fitted"))
        c2.plotly_chart(px.histogram(ols.resid, nbins=50, title="Residual distribution"))
        results.update({"target": target, "level": level, "r2": ols.rsquared, "adj_r2": ols.rsquared_adj,
                        "coefficients": coef.to_dict("index"), "vif": vif})

elif analysis == "ANOVA":
    group = st.selectbox("Group variable", categorical, index=categorical.index("month") if "month" in categorical else 0)
    target = st.selectbox("Numeric variable", numeric, index=numeric.index("consumption"))
    d = df[[group, target]].dropna()
    groups = [g[target].values for _, g in d.groupby(group) if len(g) > 1]
    if len(groups) < 2:
        st.warning("Need at least two groups.")
    else:
        f, p = stats.f_oneway(*groups)
        h, pk = stats.kruskal(*groups)
        grand = d[target].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        eta2 = ss_between / ((d[target] - grand) ** 2).sum()
        k = st.columns(4)
        k[0].metric("ANOVA F", f"{f:,.2f}")
        k[1].metric("p-value", f"{p:.2e}")
        k[2].metric("Effect size η²", f"{eta2:.3f}")
        k[3].metric("Kruskal-Wallis p (non-parametric)", f"{pk:.2e}")
        st.caption(f"η² = share of the variation in {target} explained by {group} "
                   "(0.01 small, 0.06 medium, 0.14 large).")
        summary = d.groupby(group)[target].agg(["mean", "std", "count"]).round(2)
        st.dataframe(summary)
        st.plotly_chart(px.box(d, x=group, y=target))
        results.update({"group": group, "target": target, "F": f, "p": p, "eta2": eta2, "kruskal_p": pk,
                        "group_means": summary["mean"].to_dict()})

else:
    a = st.selectbox("Variable 1", categorical)
    b = st.selectbox("Variable 2", [c for c in categorical if c != a])
    table = pd.crosstab(df[a], df[b])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))) if min(table.shape) > 1 else 0
    k = st.columns(3)
    k[0].metric("Chi-square", f"{chi2:,.1f}")
    k[1].metric("p-value", f"{p:.2e}")
    k[2].metric("Cramér's V", f"{v:.3f}")
    st.caption("Cramér's V: 0.1 weak, 0.3 moderate, 0.5 strong association.")
    st.plotly_chart(px.imshow(table, text_auto=True, aspect="auto", title=f"{a} × {b}"))
    results.update({"var1": a, "var2": b, "chi2": chi2, "p": p, "dof": dof, "cramers_v": v})

cu.ai_interpretation(f"Statistical analysis ({analysis})", results,
                     "1) What is statistically significant AND practically important 2) Physical/operational "
                     "meaning for a Calgary energy system 3) Caveats (sample size, collinearity, data quality) "
                     "4) Which variables to use in models.")
