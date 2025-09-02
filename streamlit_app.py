from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="DMA818 — Boston Housing", layout="wide")
st.title("DMA818 — Boston Housing Interactive App")
st.caption("EDA • Regression (MEDV) • Classification (CAT.MEDV) — Matplotlib visuals only")

# ---------- Data loading (fast + robust) ----------
@st.cache_data
def load_default() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    csv = base / "BostonHousing.csv"
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

# Sidebar quick info + sample download
st.sidebar.write("Rows × Columns:", df.shape)
st.sidebar.write("Columns:", list(df.columns))
with open(Path(__file__).resolve().parent / "BostonHousing.csv", "rb") as f:
    st.sidebar.download_button("⬇️ Download sample BostonHousing.csv", f, file_name="BostonHousing.csv", mime="text/csv")

# Reset controls
if st.sidebar.button("🔄 Reset selections"):
    st.session_state.clear()
    st.rerun()

# Shared model settings (used by both tabs)
with st.sidebar.expander("⚙️ Model settings", expanded=False):
    test_size = st.slider("Test size", 0.10, 0.50, 0.20, 0.05, key="test_size")
    seed = st.number_input("Random state", min_value=0, max_value=10000, value=42, step=1, key="seed")

tab1, tab2, tab3 = st.tabs(["📊 EDA", "📈 Regression (MEDV)", "🎯 Classification (CAT.MEDV)"])

# ---------- EDA ----------
with tab1:
    st.subheader("Overview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Summary")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    # Correlation heatmap (numeric only)
    st.subheader("Correlation Heatmap (Numeric)")
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
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

# ---------- Regression ----------
with tab2:
    st.subheader("Linear Regression — Predict MEDV")
    if all(c in df.columns for c in ["MEDV", "CAT. MEDV"]):
        X = df.drop(columns=["MEDV", "CAT. MEDV"])
        y = df["MEDV"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{rmse:.3f}")
        c2.metric("R²", f"{r2:.3f}")

        # Residuals
        fig, ax = plt.subplots(figsize=(6, 4))
        resid = y_test.values - y_pred
        ax.scatter(y_pred, resid, s=12); ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted MEDV"); ax.set_ylabel("Residual")
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("Columns MEDV and CAT. MEDV are required.")

# ---------- Classification ----------
with tab3:
    st.subheader("Logistic Regression — Predict CAT.MEDV")
    if all(c in df.columns for c in ["MEDV", "CAT. MEDV"]):
        X = df.drop(columns=["MEDV", "CAT. MEDV"])
        y = df["CAT. MEDV"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
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

        # Full report
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Columns MEDV and CAT. MEDV are required.")
