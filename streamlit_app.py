from pathlib import Path
import io, json, zipfile, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import (
    train_test_split, KFold, RepeatedKFold, cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

# ====================== Page ======================
st.set_page_config(page_title="DMA818 — Boston Housing", layout="wide")
st.title("DMA818 — Boston Housing Interactive App")
st.caption("EDA • Regression (Linear/Ridge/Lasso) • Classification (Logistic) — Matplotlib visuals only")

# --- Shareable URL widget (copies current URL incl. query params) ---
def _safe_html(html_str, **kwargs):
    try:
        return st.html(html_str, **kwargs)  # Newer Streamlit
    except Exception:
        import streamlit.components.v1 as components
        return components.html(html_str, **kwargs)

_safe_html("""
<div style="display:flex;gap:8px;align-items:center;margin:6px 0 18px 0;">
  <input id="shareurl" style="flex:1;padding:8px;border:1px solid #ccc;border-radius:6px" />
  <button id="copybtn" style="padding:8px 12px;border-radius:6px;border:1px solid #ddd;cursor:pointer">
    Copy URL
  </button>
</div>
<script>
const i = document.getElementById('shareurl');
const b = document.getElementById('copybtn');
const set = () => { try { i.value = window.location.href; } catch(e){} };
set();
b.onclick = async () => {
  try { await navigator.clipboard.writeText(window.location.href);
        b.textContent = "Copied!"; setTimeout(()=>b.textContent="Copy URL", 1200); }
  catch(e){ i.select(); document.execCommand('copy');
            b.textContent="Copied!"; setTimeout(()=>b.textContent="Copy URL",1200); }
};
</script>
""", height=70)

# ====================== Helpers ======================
def _qp_get(key, default, cast):
    """Query-param getter with Streamlit new/legacy API support."""
    try:
        val = st.query_params.get(key)
    except Exception:
        try:
            v = st.experimental_get_query_params().get(key, [None])
            val = v[0] if isinstance(v, list) else v
        except Exception:
            val = None
    if val is None:
        return default
    try:
        return cast(val)
    except Exception:
        return default

def _qp_set(**kwargs):
    """Query-param setter with Streamlit new/legacy API support."""
    payload = {k: str(v) for k, v in kwargs.items() if v is not None}
    try:
        st.query_params.update(payload)
    except Exception:
        st.experimental_set_query_params(**payload)

def fig_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def df_csv_str(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

def json_str(obj: dict) -> str:
    buf = io.StringIO()
    json.dump(obj, buf, indent=2)
    return buf.getvalue()

# Master stash for the global ZIP export
if "exports" not in st.session_state:
    st.session_state["exports"] = {"eda": {}, "reg": {}, "cls": {}}

# ====================== Data loading (cached) ======================
@st.cache_data
def load_default() -> pd.DataFrame:
    csv = Path(__file__).resolve().parent / "BostonHousing.csv"
    return pd.read_csv(csv)

@st.cache_data
def load_from_upload(file) -> pd.DataFrame:
    file.seek(0)
    return pd.read_csv(file)

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = load_from_upload(uploaded) if uploaded is not None else load_default()

# Sidebar info + sample download
st.sidebar.write("Rows × Columns:", df.shape)
st.sidebar.write("Columns:", list(df.columns))
sample_path = Path(__file__).resolve().parent / "BostonHousing.csv"
if sample_path.exists():
    st.sidebar.download_button(
        "⬇️ Download sample BostonHousing.csv",
        data=sample_path.read_bytes(),
        file_name="BostonHousing.csv",
        mime="text/csv",
    )

# Reset UI
if st.sidebar.button("🔄 Reset selections"):
    st.session_state.clear()
    _qp_set()  # clear query params
    st.rerun()

# Global model settings (URL-backed)
ts_default   = _qp_get("ts",   0.20, float)
seed_default = _qp_get("seed", 42,   int)
with st.sidebar.expander("⚙️ Model settings", expanded=False):
    test_size = st.slider("Test size", 0.10, 0.50, value=ts_default, step=0.05, key="test_size")
    seed      = st.number_input("Random state", 0, 10000, value=seed_default, step=1, key="seed")
_qp_set(ts=test_size, seed=seed)

# ====================== Tabs ======================
tab1, tab2, tab3, tab_nb = st.tabs(["📊 EDA", "📈 Regression (MEDV)", "🎯 Classification (CAT.MEDV)", "📓 Notebook"])

# ====================== EDA ======================
with tab1:
    st.subheader("Overview")
    st.dataframe(df.head(20), width="stretch")
    st.write("Summary")
    st.dataframe(df.describe(include="all"), width="stretch")

    # Correlation heatmap (numeric)
    st.subheader("Correlation Heatmap (Numeric)")
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
    if corr.size:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, interpolation="nearest")
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
        fig.colorbar(im)
        st.pyplot(fig, width="stretch")

    # Feature signals: variance + |corr with MEDV|
    st.subheader("Feature Signals")
    num_df = df.select_dtypes(include=[np.number])
    sig_cols = st.columns(2)

    # Variance (unsupervised)
    with sig_cols[0]:
        if not num_df.empty:
            var_series = num_df.var().sort_values(ascending=False)
            fig_v, ax_v = plt.subplots(figsize=(6, 5))
            ax_v.barh(var_series.index[::-1], var_series.values[::-1])
            ax_v.set_xlabel("Variance"); ax_v.set_ylabel("Feature"); ax_v.set_title("Feature Variance")
            st.pyplot(fig_v, width="stretch")
            var_df = var_series.reset_index(names=["feature", "variance"])
            st.download_button("📥 Download variance (CSV)", df_csv_str(var_df),
                               file_name="eda_variance.csv", mime="text/csv")
            st.session_state["exports"]["eda"]["eda_variance.csv"] = df_csv_str(var_df)

    # |corr with MEDV| (supervised proxy)
    with sig_cols[1]:
        if "MEDV" in df.columns and not num_df.empty:
            corr_target = num_df.drop(columns=["MEDV"], errors="ignore").corrwith(df["MEDV"]).abs().sort_values(ascending=False)
            corr_df = corr_target.reset_index().rename(columns={"index": "feature", 0: "abs_corr_with_MEDV"})
            fig_c, ax_c = plt.subplots(figsize=(6, 5))
            ax_c.barh(corr_target.index[::-1], corr_target.values[::-1])
            ax_c.set_xlabel("|corr|"); ax_c.set_ylabel("Feature"); ax_c.set_title("|Correlation| with MEDV")
            st.pyplot(fig_c, width="stretch")
            st.download_button("📥 Download |corr(MEDV)| (CSV)", df_csv_str(corr_df),
                               file_name="eda_abs_corr_MEDV.csv", mime="text/csv")
            st.session_state["exports"]["eda"]["eda_abs_corr_MEDV.csv"] = df_csv_str(corr_df)
        else:
            st.info("MEDV not found — correlation-to-target panel hidden.")

# ====================== Regression (Linear/Ridge/Lasso) ======================
with tab2:
    st.subheader("Regression — Predict MEDV")
    if all(c in df.columns for c in ["MEDV", "CAT. MEDV"]):
        X = df.drop(columns=["MEDV", "CAT. MEDV"])
        y = df["MEDV"]

        # Model toggles (URL-backed)
        use_lin   = _qp_get("lin",   True,  lambda s: s.lower() != "false")
        use_ridge = _qp_get("ridge", True,  lambda s: s.lower() != "false")
        use_lasso = _qp_get("lasso", True,  lambda s: s.lower() != "false")
        ridge_a   = _qp_get("ra",    1.0,   float)
        lasso_a   = _qp_get("la",    0.1,   float)
        primary   = _qp_get("model", "linear", str)

        c1, c2, c3 = st.columns(3)
        with c1:
            use_lin   = st.checkbox("Linear", value=use_lin, key="use_lin")
        with c2:
            use_ridge = st.checkbox("Ridge",  value=use_ridge, key="use_ridge")
            ridge_a   = st.number_input("α (Ridge)", 0.0001, 1000.0, value=float(ridge_a), step=0.1, key="ridge_a")
        with c3:
            use_lasso = st.checkbox("Lasso",  value=use_lasso, key="use_lasso")
            lasso_a   = st.number_input("α (Lasso)", 0.0001, 1000.0, value=float(lasso_a), step=0.1, key="lasso_a")
        _qp_set(lin=use_lin, ridge=use_ridge, lasso=use_lasso, ra=ridge_a, la=lasso_a)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        def eval_model(pipe, name):
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2   = float(r2_score(y_test, y_pred))
            return {"model": name, "rmse": rmse, "r2": r2, "y_pred": y_pred, "pipe": pipe}

        results = []
        if use_lin:
            results.append(eval_model(Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]), "linear"))
        if use_ridge:
            results.append(eval_model(Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=ridge_a))]), "ridge"))
        if use_lasso:
            results.append(eval_model(Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=lasso_a, max_iter=10000))]), "lasso"))

        export_files = {}  # name -> bytes/str

        if not results:
            st.info("Select at least one model to compare.")
        else:
            # Comparison (holdout)
            met = pd.DataFrame([{"Model": r["model"].title(), "RMSE": r["rmse"], "R²": r["r2"]} for r in results])
            st.dataframe(met.style.format({"RMSE": "{:.3f}", "R²": "{:.3f}"}), width="stretch")
            export_files["regression_comparison.csv"] = df_csv_str(met)

            # Pick primary model
            allowed = [r["model"] for r in results]
            if primary not in allowed: primary = allowed[0]
            primary = st.radio("Inspect model", allowed, index=allowed.index(primary), horizontal=True, key="primary_reg")
            _qp_set(model=primary)
            chosen = next(r for r in results if r["model"] == primary)
            y_pred = chosen["y_pred"]

            # Residual plot
            fig_resid, ax = plt.subplots(figsize=(6, 4))
            resid = y_test.values - y_pred
            ax.scatter(y_pred, resid, s=12); ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted MEDV"); ax.set_ylabel("Residual")
            st.pyplot(fig_resid, width="stretch")
            export_files[f"residuals_{primary}.png"] = fig_png_bytes(fig_resid)

            # Predictions + metrics files
            pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
            export_files[f"regression_{primary}_predictions.csv"] = df_csv_str(pred_df)
            export_files[f"regression_{primary}_metrics.json"] = json_str({
                "model": primary, "rmse": chosen["rmse"], "r2": chosen["r2"],
                "test_size": float(test_size), "seed": int(seed),
                "ridge_alpha": float(ridge_a), "lasso_alpha": float(lasso_a)
            })

            # ---- CV / Sweeps / Learning curve ----
            with st.expander("📚 Cross-Validation • Hyperparameter Sweeps • Bias–Variance", expanded=False):
                ccv1, ccv2, ccv3 = st.columns([1,1,1])
                with ccv1:
                    k_folds = st.number_input("K-Folds", 3, 20, 5, step=1, key="kfolds")
                with ccv2:
                    repeats = st.number_input("Repeats", 1, 10, 1, step=1, key="repeats")
                with ccv3:
                    do_sweep = st.checkbox("Hyperparameter sweep (α)", value=True, key="do_sweep")

                cv = RepeatedKFold(n_splits=int(k_folds), n_repeats=int(repeats), random_state=seed)

                def cv_rmse(pipe):
                    scores = cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv)
                    return -scores

                cv_rows = []
                if use_lin:
                    s = cv_rmse(Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]))
                    cv_rows.append({"Model": "Linear", f"CV RMSE mean (k={k_folds}×{repeats})": s.mean(), "std": s.std()})
                if use_ridge:
                    s = cv_rmse(Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=ridge_a))]))
                    cv_rows.append({"Model": f"Ridge (α={ridge_a:g})", f"CV RMSE mean (k={k_folds}×{repeats})": s.mean(), "std": s.std()})
                if use_lasso:
                    s = cv_rmse(Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=lasso_a, max_iter=10000))]))
                    cv_rows.append({"Model": f"Lasso (α={lasso_a:g})", f"CV RMSE mean (k={k_folds}×{repeats})": s.mean(), "std": s.std()})

                if cv_rows:
                    cv_df = pd.DataFrame(cv_rows)
                    st.dataframe(cv_df.style.format({cv_df.columns[1]: "{:.3f}", "std": "{:.3f}"}), width="stretch")
                    export_files["cv_results.csv"] = df_csv_str(cv_df)

                # Hyperparameter sweeps (Ridge/Lasso)
                if do_sweep and (use_ridge or use_lasso):
                    def sweep_alphas(model_name):
                        alphas = np.logspace(-3, 3, 20)
                        means, stds = [], []
                        for a in alphas:
                            if model_name == "ridge":
                                pipe = Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=a))])
                            else:
                                pipe = Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=a, max_iter=10000))])
                            scores = cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv)
                            rmse_vals = -scores
                            means.append(rmse_vals.mean()); stds.append(rmse_vals.std())
                        return alphas, np.array(means), np.array(stds)

                    if use_ridge:
                        al, m, sdev = sweep_alphas("ridge")
                        fig_rs, ax_rs = plt.subplots()
                        ax_rs.semilogx(al, m); ax_rs.fill_between(al, m - sdev, m + sdev, alpha=0.2)
                        ax_rs.set_xlabel("Ridge α (log)"); ax_rs.set_ylabel("CV RMSE")
                        st.pyplot(fig_rs, width="stretch")
                        export_files["ridge_sweep.png"] = fig_png_bytes(fig_rs)

                    if use_lasso:
                        al, m, sdev = sweep_alphas("lasso")
                        fig_ls, ax_ls = plt.subplots()
                        ax_ls.semilogx(al, m); ax_ls.fill_between(al, m - sdev, m + sdev, alpha=0.2)
                        ax_ls.set_xlabel("Lasso α (log)"); ax_ls.set_ylabel("CV RMSE")
                        st.pyplot(fig_ls, width="stretch")
                        export_files["lasso_sweep.png"] = fig_png_bytes(fig_ls)

                # Learning curve (bias–variance)
                if primary == "linear":
                    pipe_chosen = Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())])
                elif primary == "ridge":
                    pipe_chosen = Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=ridge_a))])
                else:
                    pipe_chosen = Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=lasso_a, max_iter=10000))])

                kf = KFold(n_splits=int(k_folds), shuffle=True, random_state=seed)
                train_sizes = np.linspace(0.2, 1.0, 6)
                sizes, train_scores, val_scores = learning_curve(
                    pipe_chosen, X, y,
                    cv=kf,
                    train_sizes=train_sizes,
                    scoring="neg_root_mean_squared_error",
                    shuffle=True,
                    random_state=seed
                )
                tr_rmse = (-train_scores).mean(axis=1)
                val_rmse = (-val_scores).mean(axis=1)
                fig_lv, ax_lv = plt.subplots()
                ax_lv.plot(sizes * len(X), tr_rmse, marker="o", label="Train RMSE")
                ax_lv.plot(sizes * len(X), val_rmse, marker="o", label="CV RMSE")
                ax_lv.set_xlabel("Training examples"); ax_lv.set_ylabel("RMSE"); ax_lv.legend()
                ax_lv.set_title("Bias–Variance (Learning Curve)")
                st.pyplot(fig_lv, width="stretch")
                export_files[f"learning_curve_{primary}.png"] = fig_png_bytes(fig_lv)

            # Permutation Importance (Top-K + error bars)
            with st.expander("🧪 Permutation Importance", expanded=False):
                nrep = st.slider("Repeats", 2, 50, 10, step=1, key="pi_rep_reg")
                run_pi = st.checkbox("Compute permutation importance", value=False, key="pi_run_reg")
                if run_pi:
                    with st.spinner("Computing feature importances..."):
                        pi = permutation_importance(
                            estimator=chosen["pipe"],
                            X=X_test, y=y_test,
                            scoring="neg_root_mean_squared_error",
                            n_repeats=int(nrep),
                            random_state=seed
                        )
                        imp = pd.DataFrame({
                            "feature": X.columns,
                            "importance": np.abs(pi.importances_mean),
                            "std": pi.importances_std
                        }).sort_values("importance", ascending=False)
                        kmax = max(1, min(15, len(imp)))
                        k = st.slider("Top-K features", 1, kmax, min(8, kmax), key="pi_topk_reg")
                        imp_k = imp.head(k).iloc[::-1]
                        fig_pi, ax_pi = plt.subplots(figsize=(7, 0.45*len(imp_k)+1))
                        ax_pi.barh(imp_k["feature"], imp_k["importance"], xerr=imp_k["std"], capsize=3)
                        ax_pi.set_xlabel("Permutation importance (|ΔRMSE|)")
                        ax_pi.set_ylabel("Feature")
                        st.pyplot(fig_pi, width="stretch")
                        export_files[f"perm_importance_{primary}_top{k}.csv"] = df_csv_str(imp_k[::-1])
                        export_files[f"perm_importance_{primary}_top{k}.png"] = fig_png_bytes(fig_pi)

            # Partial Dependence (top-2 by |coef|)
            with st.expander("🧩 Partial Dependence (top-2 by |coef|)", expanded=False):
                est = chosen["pipe"].named_steps["m"]
                if hasattr(est, "coef_"):
                    coefs = np.abs(np.ravel(est.coef_))
                    if np.any(coefs):
                        top2_idx = np.argsort(coefs)[-2:][::-1]
                        top2_feats = [X.columns[i] for i in top2_idx]
                        st.write(f"Top-2: **{top2_feats[0]}**, **{top2_feats[1]}**")
                        for f in top2_feats:
                            fig_pd, ax_pd = plt.subplots()
                            PartialDependenceDisplay.from_estimator(chosen["pipe"], X, [f], ax=ax_pd)
                            ax_pd.set_title(f"Partial dependence: {f}")
                            st.pyplot(fig_pd, width="stretch")
                            export_files[f"partial_dependence_{primary}_{f}.png"] = fig_png_bytes(fig_pd)
                    else:
                        st.info("Coefficients are all zero; PD not informative.")
                else:
                    st.info("Primary model has no coefficients; PD not available.")

            # Pipeline config JSON
            pipeline_cfg = {
                "task": "regression",
                "dataset": {"rows": int(df.shape[0]), "cols": list(df.columns)},
                "settings": {
                    "test_size": float(test_size), "seed": int(seed),
                    "use_linear": bool(use_lin), "use_ridge": bool(use_ridge), "use_lasso": bool(use_lasso),
                    "ridge_alpha": float(ridge_a), "lasso_alpha": float(lasso_a),
                    "primary_model": str(primary)
                }
            }
            st.download_button("🧾 Download pipeline config (JSON)",
                               data=json_str(pipeline_cfg),
                               file_name="pipeline_config_regression.json",
                               mime="application/json")
            export_files["pipeline_config_regression.json"] = json_str(pipeline_cfg)

            # Export session (regression)
            st.divider()
            if st.button("🗂️ Build export (.zip)"):
                mem = io.BytesIO()
                stamp = time.strftime("%Y%m%d-%H%M%S")
                with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name, payload in export_files.items():
                        arc = f"regression/{name}"
                        if isinstance(payload, bytes):
                            zf.writestr(arc, payload)
                        else:
                            zf.writestr(arc, payload.encode("utf-8"))
                mem.seek(0)
                st.download_button(
                    "📦 Download session (regression).zip",
                    data=mem.getvalue(),
                    file_name=f"boston_regression_session_{stamp}.zip",
                    mime="application/zip"
                )

            # Merge into global export
            st.session_state["exports"]["reg"].update(export_files)

    else:
        st.warning("Columns MEDV and CAT. MEDV are required.")

# ====================== Classification (Logistic) ======================
with tab3:
    st.subheader("Logistic Regression — Predict CAT.MEDV")
    if all(c in df.columns for c in ["MEDV", "CAT. MEDV"]):
        X = df.drop(columns=["MEDV", "CAT. MEDV"])
        y = df["CAT. MEDV"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        clf = Pipeline([("scaler", StandardScaler()),
                        ("m", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        st.metric("Accuracy", f"{acc:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig_cm)

        st.text("Classification Report")
        cls_report = classification_report(y_test, y_pred)
        st.text(cls_report)

        # CV accuracy quick panel
        cls_export = {}
        with st.expander("📚 Cross-Validation (Accuracy)"):
            k_folds_c = st.number_input("K-Folds (classification)", 3, 20, 5, step=1, key="kfolds_cls")
            repeats_c = st.number_input("Repeats (classification)", 1, 10, 1, step=1, key="repeats_cls")
            cv_c = RepeatedKFold(n_splits=int(k_folds_c), n_repeats=int(repeats_c), random_state=seed)
            scores = cross_val_score(
                Pipeline([("scaler", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
                X, y, scoring="accuracy", cv=cv_c
            )
            st.write(f"CV Accuracy mean±std: **{scores.mean():.3f} ± {scores.std():.3f}**")
            cls_cv_df = pd.DataFrame({"metric": ["accuracy_mean", "accuracy_std"], "value": [scores.mean(), scores.std()]})
            cls_export["classification_cv_summary.csv"] = df_csv_str(cls_cv_df)

        # Predictions + metrics for export
        pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
        cls_export["classification_predictions.csv"] = df_csv_str(pred_df)
        cls_export["classification_metrics.json"] = json_str({"accuracy": acc, "test_size": float(test_size), "seed": int(seed)})
        cls_export["confusion_matrix.png"] = fig_png_bytes(fig_cm)
        cls_export["classification_report.txt"] = cls_report

        # Permutation Importance (Top-K + error bars)
        with st.expander("🧪 Permutation Importance"):
            nrep_c = st.slider("Repeats", 2, 50, 10, step=1, key="pi_rep_cls")
            run_pi_c = st.checkbox("Compute permutation importance", value=False, key="pi_run_cls")
            if run_pi_c:
                with st.spinner("Computing feature importances..."):
                    pi = permutation_importance(
                        estimator=clf,
                        X=X_test, y=y_test,
                        scoring="accuracy",
                        n_repeats=int(nrep_c),
                        random_state=seed
                    )
                    imp = pd.DataFrame({
                        "feature": X.columns,
                        "importance": np.abs(pi.importances_mean),
                        "std": pi.importances_std
                    }).sort_values("importance", ascending=False)
                    kmax = max(1, min(15, len(imp)))
                    k = st.slider("Top-K features", 1, kmax, min(8, kmax), key="pi_topk_cls")
                    imp_k = imp.head(k).iloc[::-1]
                    fig_pi_c, ax_pi_c = plt.subplots(figsize=(7, 0.45*len(imp_k)+1))
                    ax_pi_c.barh(imp_k["feature"], imp_k["importance"], xerr=imp_k["std"], capsize=3)
                    ax_pi_c.set_xlabel("Permutation importance (ΔAccuracy)")
                    ax_pi_c.set_ylabel("Feature")
                    st.pyplot(fig_pi_c, width="stretch")
                    cls_export[f"classification_perm_importance_top{k}.csv"] = df_csv_str(imp_k[::-1])
                    cls_export[f"classification_perm_importance_top{k}.png"] = fig_png_bytes(fig_pi_c)

        # Pipeline config JSON
        pipeline_cfg_cls = {
            "task": "classification",
            "dataset": {"rows": int(df.shape[0]), "cols": list(df.columns)},
            "settings": {
                "test_size": float(test_size), "seed": int(seed),
                "model": "logistic_regression", "max_iter": 1000
            }
        }
        st.download_button("🧾 Download pipeline config (JSON)",
                           data=json_str(pipeline_cfg_cls),
                           file_name="pipeline_config_classification.json",
                           mime="application/json")
        cls_export["pipeline_config_classification.json"] = json_str(pipeline_cfg_cls)

        # Export session (classification)
        st.divider()
        if st.button("🗂️ Build export (.zip)", key="zip_cls"):
            mem = io.BytesIO()
            stamp = time.strftime("%Y%m%d-%H%M%S")
            with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, payload in cls_export.items():
                    arc = f"classification/{name}"
                    if isinstance(payload, bytes):
                        zf.writestr(arc, payload)
                    else:
                        zf.writestr(arc, payload.encode("utf-8"))
            mem.seek(0)
            st.download_button(
                "📦 Download session (classification).zip",
                data=mem.getvalue(),
                file_name=f"boston_classification_session_{stamp}.zip",
                mime="application/zip"
            )

        # Merge into global export
        st.session_state["exports"]["cls"].update(cls_export)

    else:
        st.warning("Columns MEDV and CAT. MEDV are required.")

# ====================== Notebook Viewer ======================
with tab_nb:
    st.subheader("Notebook Viewer")

    # Embedded sample notebook (small, synthetic)
    SAMPLE_NB = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Sample Boston Housing Notebook\\n",
                "Quick demo: load CSV and preview head.\\n"
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import pandas as pd\\n",
                "df = pd.read_csv('BostonHousing.csv')\\n",
                "df.head()\\n"
            ]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5
    }
    sample_bytes = json.dumps(SAMPLE_NB, indent=2).encode("utf-8")

    cleft, cright = st.columns(2)
    with cleft:
        st.download_button("📥 Download sample notebook", data=sample_bytes,
                           file_name="sample_boston_notebook.ipynb", mime="application/x-ipynb+json")
    with cright:
        render_sample = st.button("👀 Render embedded sample now")

    nb_file = st.file_uploader("…or upload a .ipynb to render", type=["ipynb"])

    def _render_nb_html(nb_json_str: str):
        try:
            import nbformat
            from nbconvert import HTMLExporter
            nb = nbformat.reads(nb_json_str, as_version=4)
            html_exporter = HTMLExporter()
            html_exporter.exclude_input = False
            html_exporter.exclude_output = False
            body, _ = html_exporter.from_notebook_node(nb)
            _safe_html(body, height=800)
        except Exception as e:
            st.error(f"Render failed ({e}); showing raw JSON.")
            st.code(nb_json_str)

    if render_sample:
        _render_nb_html(json.dumps(SAMPLE_NB))

    if nb_file:
        try:
            _render_nb_html(nb_file.getvalue().decode("utf-8"))
        except Exception as e:
            st.error(f"Failed to decode notebook: {e}")

# ====================== Master Export (ALL) + Footer ======================
st.divider()
if st.button("📦 Download EVERYTHING (.zip)"):
    mem = io.BytesIO()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for section, files in st.session_state["exports"].items():
            for name, payload in files.items():
                arc = f"{section}/{name}"
                if isinstance(payload, bytes):
                    zf.writestr(arc, payload)
                else:
                    zf.writestr(arc, payload.encode("utf-8"))
    mem.seek(0)
    st.download_button("⬇️ Export ALL (EDA+Regression+Classification).zip",
                       data=mem.getvalue(),
                       file_name=f"boston_full_session_{stamp}.zip",
                       mime="application/zip")

# Build/version footer
from datetime import datetime, timezone
build_ts = datetime.fromtimestamp(Path(__file__).stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.caption(f"Build: {build_ts} • Streamlit {st.__version__}")
