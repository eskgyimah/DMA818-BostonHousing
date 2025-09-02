from pathlib import Path
import io, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)

st.set_page_config(page_title="DMA818 — Boston Housing", layout="wide")
st.title("DMA818 — Boston Housing Interactive App")
st.caption("EDA • Regression (Linear/Ridge/Lasso) • Classification (Logistic) — Matplotlib visuals only")

# ---------------------- Query-param helpers (URL share) ----------------------
def _qp_get(key, default, cast):
    try:
        val = st.query_params.get(key)
    except Exception:
        try:
            val = st.experimental_get_query_params().get(key, [None])
            if isinstance(val, list):
                val = val[0]
        except Exception:
            val = None
    if val is None:
        return default
    try:
        return cast(val)
    except Exception:
        return default

def _qp_set(**kwargs):
    payload = {k: str(v) for k, v in kwargs.items() if v is not None}
    try:
        st.query_params.update(payload)
    except Exception:
        st.experimental_set_query_params(**payload)  # fallback for older versions

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
if uploaded is not None:
    df = load_from_upload(uploaded)
else:
    df = load_default()

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

# Reset
if st.sidebar.button("🔄 Reset selections"):
    st.session_state.clear()
    _qp_set()  # clears query params
    st.rerun()

# Global model settings (shared) with URL-backed defaults
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

        # Model toggles with URL-backed defaults
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

        # Train/test split view
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        def eval_model(pipe, name):
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2   = float(r2_score(y_test, y_pred))
            return {"model": name, "rmse": rmse, "r2": r2, "y_pred": y_pred}

        results = []
        if use_lin:
            results.append(eval_model(Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]), "linear"))
        if use_ridge:
            results.append(eval_model(Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=ridge_a))]), "ridge"))
        if use_lasso:
            results.append(eval_model(Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=lasso_a, max_iter=10000))]), "lasso"))

        if not results:
            st.info("Select at least one model to compare.")
        else:
            # Comparison (holdout)
            met = pd.DataFrame([{"Model": r["model"].title(), "RMSE": r["rmse"], "R²": r["r2"]} for r in results])
            st.dataframe(met.style.format({"RMSE": "{:.3f}", "R²": "{:.3f}"}), use_container_width=True)

            # Choose primary model for plots/downloads
            allowed = [r["model"] for r in results]
            if primary not in allowed:
                primary = allowed[0]
            primary = st.radio("Inspect model", allowed, index=allowed.index(primary), horizontal=True, key="primary_reg")
            _qp_set(model=primary)

            chosen = next(r for r in results if r["model"] == primary)
            y_pred = chosen["y_pred"]

            # Residual plot (holdout)
            fig, ax = plt.subplots(figsize=(6, 4))
            resid = y_test.values - y_pred
            ax.scatter(y_pred, resid, s=12); ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted MEDV"); ax.set_ylabel("Residual")
            st.pyplot(fig, use_container_width=True)

            # ---------------------- CV & Bias-Variance ----------------------
            with st.expander("📚 Cross-Validation & Bias–Variance", expanded=False):
                ccv1, ccv2, ccv3 = st.columns([1,1,1])
                with ccv1:
                    k_folds = st.number_input("K-Folds", 3, 20, 5, step=1, key="kfolds")
                with ccv2:
                    repeats = st.number_input("Repeats", 1, 10, 1, step=1, key="repeats")
                with ccv3:
                    do_sweep = st.checkbox("Hyperparameter sweep (α)", value=True, key="do_sweep")

                cv = RepeatedKFold(n_splits=int(k_folds), n_repeats=int(repeats), random_state=seed)

                # CV metrics for selected models
                def cv_rmse(pipe):
                    scores = cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv)
                    return -scores  # positive RMSE

                cv_rows = []
                if use_lin:
                    s = cv_rmse(Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]))
                    cv_rows.append({"Model": "Linear", "CV RMSE (mean)": s.mean(), "CV RMSE (std)": s.std()})
                if use_ridge:
                    s = cv_rmse(Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=ridge_a))]))
                    cv_rows.append({"Model": f"Ridge (α={ridge_a:g})", "CV RMSE (mean)": s.mean(), "CV RMSE (std)": s.std()})
                if use_lasso:
                    s = cv_rmse(Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=lasso_a, max_iter=10000))]))
                    cv_rows.append({"Model": f"Lasso (α={lasso_a:g})", "CV RMSE (mean)": s.mean(), "CV RMSE (std)": s.std()})

                if cv_rows:
                    cv_df = pd.DataFrame(cv_rows)
                    st.dataframe(cv_df.style.format({"CV RMSE (mean)": "{:.3f}", "CV RMSE (std)": "{:.3f}"}),
                                 use_container_width=True)

                    # Download CV table
                    buf = io.StringIO(); cv_df.to_csv(buf, index=False)
                    st.download_button("📥 Download CV results (CSV)", buf.getvalue(), "cv_results.csv", "text/csv")

                # Hyperparameter sweep plots (Ridge/Lasso)
                if do_sweep and (use_ridge or use_lasso):
                    sweep_cols = 2 if (use_ridge and use_lasso) else 1
                    sc1, sc2 = st.columns(sweep_cols)

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
                        with sc1:
                            al, m, sdev = sweep_alphas("ridge")
                            fig_s, ax_s = plt.subplots()
                            ax_s.semilogx(al, m)
                            ax_s.fill_between(al, m - sdev, m + sdev, alpha=0.2)
                            ax_s.set_xlabel("Ridge α (log scale)"); ax_s.set_ylabel("CV RMSE")
                            st.pyplot(fig_s, use_container_width=True)

                    if use_lasso:
                        target_col = sc2 if (use_ridge and use_lasso) else sc1
                        with target_col:
                            al, m, sdev = sweep_alphas("lasso")
                            fig_s2, ax_s2 = plt.subplots()
                            ax_s2.semilogx(al, m)
                            ax_s2.fill_between(al, m - sdev, m + sdev, alpha=0.2)
                            ax_s2.set_xlabel("Lasso α (log scale)"); ax_s2.set_ylabel("CV RMSE")
                            st.pyplot(fig_s2, use_container_width=True)

                # Bias–variance (learning curve) for the chosen model
                # Build the same pipeline object as 'chosen'
                if primary == "linear":
                    pipe_chosen = Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())])
                elif primary == "ridge":
                    pipe_chosen = Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=ridge_a))])
                else:
                    pipe_chosen = Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=lasso_a, max_iter=10000))])

                kf = KFold(n_splits=int(k_folds), shuffle=True, random_state=seed)
                train_sizes = np.linspace(0.2, 1.0, 6)
                tr_scores, val_scores = learning_curve(
                    pipe_chosen, X, y,
                    cv=kf,
                    train_sizes=train_sizes,
                    scoring="neg_root_mean_squared_error",
                    shuffle=True,
                    random_state=seed
                )[1:]  # returns (train_sizes, train_scores, test_scores)

                tr_rmse = np.sqrt((-tr_scores).mean(axis=1))
                val_rmse = np.sqrt((-val_scores).mean(axis=1))

                fig_lv, ax_lv = plt.subplots()
                ax_lv.plot(train_sizes * len(X), tr_rmse, marker="o", label="Train RMSE")
                ax_lv.plot(train_sizes * len(X), val_rmse, marker="o", label="CV RMSE")
                ax_lv.set_xlabel("Training examples")
                ax_lv.set_ylabel("RMSE")
                ax_lv.legend()
                ax_lv.set_title("Bias–Variance (Learning Curve)")
                st.pyplot(fig_lv, use_container_width=True)

            # ---------------------- Downloads: predictions + metrics ----------------------
            pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
            csv_buf = io.StringIO(); pred_df.to_csv(csv_buf, index=False)
            st.download_button("📥 Download predictions (CSV)", data=csv_buf.getvalue(),
                               file_name=f"regression_{primary}_predictions.csv", mime="text/csv")

            metrics_obj = {"model": primary, "rmse": chosen["rmse"], "r2": chosen["r2"],
                           "test_size": float(test_size), "seed": int(seed),
                           "ridge_alpha": float(ridge_a), "lasso_alpha": float(lasso_a)}
            json_buf = io.StringIO(); json.dump(metrics_obj, json_buf, indent=2)
            st.download_button("📥 Download metrics (JSON)", data=json_buf.getvalue(),
                               file_name=f"regression_{primary}_metrics.json", mime="application/json")
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

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # CV for classification (accuracy)
        with st.expander("📚 Cross-Validation (Accuracy)"):
            k_folds_c = st.number_input("K-Folds (classification)", 3, 20, 5, step=1, key="kfolds_cls")
            repeats_c = st.number_input("Repeats (classification)", 1, 10, 1, step=1, key="repeats_cls")
            cv_c = RepeatedKFold(n_splits=int(k_folds_c), n_repeats=int(repeats_c), random_state=seed)
            scores = cross_val_score(Pipeline([("scaler", StandardScaler()),
                                               ("m", LogisticRegression(max_iter=1000))]),
                                     X, y, scoring="accuracy", cv=cv_c)
            st.write(f"CV Accuracy mean±std: **{scores.mean():.3f} ± {scores.std():.3f}**")

        # Downloads (predictions + metrics)
        pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
        csv_buf = io.StringIO(); pred_df.to_csv(csv_buf, index=False)
        st.download_button("📥 Download predictions (CSV)", data=csv_buf.getvalue(),
                           file_name="classification_predictions.csv", mime="text/csv")

        metrics_obj = {"accuracy": acc, "test_size": float(test_size), "seed": int(seed)}
        json_buf = io.StringIO(); json.dump(metrics_obj, json_buf, indent=2)
        st.download_button("📥 Download metrics (JSON)", data=json_buf.getvalue(),
                           file_name="classification_metrics.json", mime="application/json")
    else:
        st.warning("Columns MEDV and CAT. MEDV are required.")
