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

# ---------------------- Page ----------------------
st.set_page_config(page_title="DMA818 — Boston Housing", layout="wide")
st.title("DMA818 — Boston Housing Interactive App")
st.caption("EDA • Regression (Linear/Ridge/Lasso) • Classification (Logistic) — Matplotlib visuals only")

# ---------------------- Helpers ----------------------
def _qp_get(key, default, cast):
    """Query-param getter with Streamlit new/legacy API support."""
    try: val = st.query_params.get(key)
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
    try: st.query_params.update(payload)
    except Exception: st.experimental_set_query_params(**payload)

def fig_png_bytes(fig) -> bytes:
    """Render a Matplotlib figure to PNG bytes."""
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

# ---------------------- Data loading (cached) ----------------------
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

tab1, tab2, tab3 = st.tabs(["📊 EDA", "📈 Regression (MEDV)", "🎯 Classification (CAT.MEDV)"])

# ---------------------- EDA ----------------------
with tab1:
    st.subheader("Overview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Summary")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    # Correlation heatmap (numeric)
    st.subheader("Correlation Heatmap (Numeric)")
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
    if corr.size:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, interpolation="nearest")
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
        fig.colorbar(im)
        st.pyplot(fig, use_container_width=True)

    # MEDV histogram
    if "MEDV" in df.columns:
        st.subheader("MEDV Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(df["MEDV"].values, bins=30)
        ax2.set_xlabel("MEDV"); ax2.set_ylabel("Frequency")
        st.pyplot(fig2, use_container_width=True)

# ---------------------- Regression (Linear/Ridge/Lasso) ----------------------
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

        # Storage for exports
        export_files = {}  # name -> bytes/str

        if not results:
            st.info("Select at least one model to compare.")
        else:
            # Comparison (holdout)
            met = pd.DataFrame([{"Model": r["model"].title(), "RMSE": r["rmse"], "R²": r["r2"]} for r in results])
            st.dataframe(met.style.format({"RMSE": "{:.3f}", "R²": "{:.3f}"}), use_container_width=True)
            export_files["regression_comparison.csv"] = df_csv_str(met)

            # Pick primary for plots
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
            st.pyplot(fig_resid, use_container_width=True)
            export_files[f"residuals_{primary}.png"] = fig_png_bytes(fig_resid)

            # Predictions + metrics files
            pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
            export_files[f"regression_{primary}_predictions.csv"] = df_csv_str(pred_df)
            export_files[f"regression_{primary}_metrics.json"] = json_str({
                "model": primary, "rmse": chosen["rmse"], "r2": chosen["r2"],
                "test_size": float(test_size), "seed": int(seed),
                "ridge_alpha": float(ridge_a), "lasso_alpha": float(lasso_a)
            })

            # ---------------------- CV & sweeps & learning curve ----------------------
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
                    return -scores  # RMSE positive

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
                    st.dataframe(cv_df.style.format({cv_df.columns[1]: "{:.3f}", "std": "{:.3f}"}), use_container_width=True)
                    export_files["cv_results.csv"] = df_csv_str(cv_df)

                # Hyperparameter sweeps
                ridge_fig = lasso_fig = None
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

                    cols = st.columns(2) if (use_ridge and use_lasso) else st.columns(1)
                    if use_ridge:
                        with cols[0]:
                            al, m, sdev = sweep_alphas("ridge")
                            ridge_fig, ax_s = plt.subplots()
                            ax_s.semilogx(al, m)
                            ax_s.fill_between(al, m - sdev, m + sdev, alpha=0.2)
                            ax_s.set_xlabel("Ridge α (log)"); ax_s.set_ylabel("CV RMSE")
                            st.pyplot(ridge_fig, use_container_width=True)
                            export_files["ridge_sweep.png"] = fig_png_bytes(ridge_fig)

                    if use_lasso:
                        target_col = cols[1] if (use_ridge and use_lasso) else cols[0]
                        with target_col:
                            al, m, sdev = sweep_alphas("lasso")
                            lasso_fig, ax_s2 = plt.subplots()
                            ax_s2.semilogx(al, m)
                            ax_s2.fill_between(al, m - sdev, m + sdev, alpha=0.2)
                            ax_s2.set_xlabel("Lasso α (log)"); ax_s2.set_ylabel("CV RMSE")
                            st.pyplot(lasso_fig, use_container_width=True)
                            export_files["lasso_sweep.png"] = fig_png_bytes(lasso_fig)

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

                # learning_curve returns negative RMSE; negate to get RMSE
                tr_rmse = (-train_scores).mean(axis=1)
                val_rmse = (-val_scores).mean(axis=1)

                fig_lv, ax_lv = plt.subplots()
                ax_lv.plot(sizes * len(X), tr_rmse, marker="o", label="Train RMSE")
                ax_lv.plot(sizes * len(X), val_rmse, marker="o", label="CV RMSE")
                ax_lv.set_xlabel("Training examples")
                ax_lv.set_ylabel("RMSE")
                ax_lv.legend()
                ax_lv.set_title("Bias–Variance (Learning Curve)")
                st.pyplot(fig_lv, use_container_width=True)
                export_files[f"learning_curve_{primary}.png"] = fig_png_bytes(fig_lv)

            # ---------------------- Permutation Importance ----------------------
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
                        # Importance as positive numbers (higher = more important)
                        imp = pd.DataFrame({
                            "feature": X.columns,
                            "importance": np.abs(pi.importances_mean),
                            "std": pi.importances_std
                        }).sort_values("importance", ascending=False)
                        st.dataframe(imp, use_container_width=True)

                        # Bar plot
                        fig_pi, ax_pi = plt.subplots(figsize=(7, 5))
                        ax_pi.barh(imp["feature"], imp["importance"])
                        ax_pi.invert_yaxis()
                        ax_pi.set_xlabel("Permutation importance (|ΔRMSE|)")
                        st.pyplot(fig_pi, use_container_width=True)

                        # Exports
                        export_files[f"perm_importance_{primary}.csv"] = df_csv_str(imp)
                        export_files[f"perm_importance_{primary}.png"] = fig_png_bytes(fig_pi)

            # ---------------------- Export session (.zip) ----------------------
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
    else:
        st.warning("Columns MEDV and CAT. MEDV are required.")

# ---------------------- Classification (Logistic) ----------------------
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

        # Confusion matrix figure
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig_cm)

        # Text report
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

        # Permutation Importance (classification)
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
                    st.dataframe(imp, use_container_width=True)

                    fig_pi_c, ax_pi_c = plt.subplots(figsize=(7, 5))
                    ax_pi_c.barh(imp["feature"], imp["importance"])
                    ax_pi_c.invert_yaxis()
                    ax_pi_c.set_xlabel("Permutation importance (ΔAccuracy)")
                    st.pyplot(fig_pi_c, use_container_width=True)

                    cls_export["classification_perm_importance.csv"] = df_csv_str(imp)
                    cls_export["classification_perm_importance.png"] = fig_png_bytes(fig_pi_c)

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
    else:
        st.warning("Columns MEDV and CAT. MEDV are required.")
