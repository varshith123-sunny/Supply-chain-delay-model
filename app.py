# app.py
"""
Supply Chain Delay Prediction — Streamlit single-file app
Project 18: Industrial Analytics - Predict shipment delays + dashboard + optimization hints
- Dataset expected at: data/DataCoSupplyChainDataset_no_zeros.csv
- Features and pipeline are chosen dynamically from available columns.
- Model: LogisticRegression (balanced), pipeline saved to models/delay_model.pkl
"""

import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Supply Chain Delay Prediction", layout="wide", page_icon="📦")
DATA_PATH = "data/DataCoSupplyChainDataset_no_zeros.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "delay_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# Dark/neon styling (inline CSS)
st.markdown(
    """
    <style>
    .stApp { background-color: #050505; color: #cfeeff; }
    .sidebar .sidebar-content { background: #070707; color: #cfeeff; }
    .kpi { background: linear-gradient(90deg, rgba(0,230,255,0.04), rgba(0,0,0,0.02));
           padding: 12px; border-radius: 8px; border: 1px solid rgba(0,230,255,0.06);
           }
    .neon { color: #00e6ff; font-weight:700; }
    .small { font-size:0.9em; color:#a6e7ff }
    .muted { color:#9fbccc }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Utility functions
# -----------------------
@st.cache_data
def load_raw(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_target(df):
    """Ensure Late_delivery_risk exists and is binary; recreate if degenerate."""
    if "Late_delivery_risk" not in df.columns or df["Late_delivery_risk"].nunique() < 2:
        if "Days for shipping (real)" in df.columns and "Days for shipment (scheduled)" in df.columns:
            df["Late_delivery_risk"] = (
                pd.to_numeric(df["Days for shipping (real)"], errors="coerce")
                > pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
            ).astype(int)
        else:
            if "Delivery Status" in df.columns:
                df["Late_delivery_risk"] = df["Delivery Status"].apply(
                    lambda x: 0 if isinstance(x, str) and "delivered" in x.lower() else 1
                ).astype(int)
            else:
                # fallback: set all zeros (will be handled downstream)
                df["Late_delivery_risk"] = 0
    return df

def basic_preprocess(df):
    df = df.copy()
    # parse dates if present
    if "order date (DateOrders)" in df.columns:
        df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="%Y-%m-%d", errors="coerce")

        df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
        df["order_year"] = df["order_date"].dt.year
    else:
        df["order_month"] = "Unknown"
        df["order_year"] = np.nan

    # numeric conversions for commonly used columns
    for c in ["Order Item Quantity", "Order Item Discount", "Product Price",
              "Order Item Total", "Sales", "Days for shipping (real)", "Days for shipment (scheduled)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Ensure target
    df = ensure_target(df)
    return df

def choose_features(df):
    """Return a sensible list of features (exists in df)."""
    # Candidate feature names prioritized
    candidates = [
        "Days for shipping (real)", "Days for shipment (scheduled)",
        "Order Item Quantity", "Order Item Discount", "Product Price",
        "Order Region", "Order State", "Order City",
        "Category Name", "Department Name", "Shipping Mode"
    ]
    features = [c for c in candidates if c in df.columns]
    # Remove extremely high-cardinality city if too many categories? Keep for now.
    return features

def build_pipeline(X):
    # determine numeric / categorical from X dtypes
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    # transformers
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="drop")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    return pipe, numeric_cols, categorical_cols

def save_model(pipe, metrics):
    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

def load_model():
    """Safe model loader — handles version or attribute errors gracefully."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.warning(f"⚠️ Model load failed ({e.__class__.__name__}). The file may be incompatible or corrupted.")
        st.info("Retraining the model is recommended — click 'Train Model' in the sidebar.")
        return None


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None

def model_evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, proba) if proba is not None else None
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {"accuracy": acc, "roc_auc": auc, "report": report, "confusion_matrix": cm.tolist()}
    return metrics

def optimization_suggestions(row):
    """Simple heuristic suggestions based on inputs and predicted probability."""
    # row: pandas Series for a single shipment (after preprocessing or raw)
    suggestions = []
    # if long scheduled days and probability high
    if "Order Item Quantity" in row.index and row.get("Order Item Quantity", 0) >= 10:
        suggestions.append("Large order quantity — consider splitting shipment or using multiple carriers.")
    if "Shipping Mode" in row.index:
        mode = str(row.get("Shipping Mode", "")).lower()
        if "standard" in mode or "ground" in mode:
            suggestions.append("Standard/Ground shipping has higher delay risk — consider upgrading to faster class.")
    # discount/profit related
    if "Order Item Discount" in row.index:
        disc = row.get("Order Item Discount", 0) or 0
        if disc > 0.2:
            suggestions.append("High discount orders may be dropshipped/not prioritized — confirm supplier SLA.")
    # supplier heuristics (if Department Name or Category exist)
    if "Category Name" in row.index:
        cat = str(row.get("Category Name", "")).lower()
        if "electronics" in cat:
            suggestions.append("Electronics shipments sensitive to transit — prefer insured/priority carriers.")
    if not suggestions:
        suggestions.append("No specific optimization detected — monitor carrier ETA and weather on route.")
    return suggestions

# -----------------------
# Load & prepare data (safe)
# -----------------------
st.title("📦 Supply Chain Delay Prediction — Dashboard + Model")
st.markdown("Predict shipment delays, analyze drivers, and get optimization hints.")

# Attempt to load dataset; show friendly message if missing
try:
    raw_df = load_raw()
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. Place your CSV there and refresh.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

df = basic_preprocess(raw_df)

# -----------------------
# Sidebar: filters and actions
# -----------------------
st.sidebar.markdown("<h3 class='neon'>Filters & Actions</h3>", unsafe_allow_html=True)

# Filters (dynamic)
def options_for(col):
    if col in df.columns:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = sorted(vals)[:500]  # avoid extreme lists
        return ["All"] + vals
    return ["All"]

city_opts = options_for("Order City")
state_opts = options_for("Order State")
cat_opts = options_for("Category Name")
mode_opts = options_for("Shipping Mode")
year_opts = ["All"] + sorted(df["order_year"].dropna().unique().astype(int).astype(str).tolist()) if "order_year" in df.columns else ["All"]

sel_city = st.sidebar.selectbox("Order City", city_opts, index=0)
sel_state = st.sidebar.selectbox("Order State", state_opts, index=0)
sel_category = st.sidebar.selectbox("Category", cat_opts, index=0)
sel_mode = st.sidebar.selectbox("Shipping Mode", mode_opts, index=0)
sel_year = st.sidebar.selectbox("Order Year", year_opts, index=0)

# quantity slider
max_qty = int(df["Order Item Quantity"].max()) if "Order Item Quantity" in df.columns and df["Order Item Quantity"].count()>0 else 100
sel_qty = st.sidebar.slider("Max Order Quantity (<=)", min_value=0, max_value=max_qty, value=max_qty)

st.sidebar.markdown("---")
# Actions
st.sidebar.markdown("### Model actions")
train_btn = st.sidebar.button("🔁 Train Model (Logistic Regression)")
view_metrics_btn = st.sidebar.button("📈 View Last Model Metrics")
download_filtered = st.sidebar.button("⬇️ Download Filtered CSV")

# -----------------------
# Apply filters to df
# -----------------------
def apply_filters(df):
    d = df.copy()
    if sel_city != "All":
        d = d[d["Order City"].astype(str) == sel_city]
    if sel_state != "All":
        d = d[d["Order State"].astype(str) == sel_state]
    if sel_category != "All":
        d = d[d["Category Name"].astype(str) == sel_category]
    if sel_mode != "All":
        d = d[d["Shipping Mode"].astype(str) == sel_mode]
    if sel_year != "All":
        d = d[d["order_year"].astype(str) == sel_year]
    if "Order Item Quantity" in d.columns:
        d = d[d["Order Item Quantity"].fillna(0) <= sel_qty]
    return d

filtered = apply_filters(df)

# CSV download
if download_filtered:
    st.sidebar.markdown("Preparing CSV...")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download filtered CSV", csv, "filtered_supply_chain.csv", "text/csv")

# -----------------------
# Main layout: KPIs, Charts, Tabs
# -----------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown("<div class='kpi'><div class='small'>Total Orders (filtered)</div><div class='neon' style='font-size:20px'>"
                f"{len(filtered):,}</div></div>", unsafe_allow_html=True)
with k2:
    delayed = int(filtered["Late_delivery_risk"].sum()) if "Late_delivery_risk" in filtered.columns else 0
    delay_pct = (delayed / len(filtered) * 100) if len(filtered) > 0 else 0
    st.markdown("<div class='kpi'><div class='small'>Delayed Orders</div><div class='neon' style='font-size:20px'>"
                f"{delayed:,} ({delay_pct:.1f}%)</div></div>", unsafe_allow_html=True)
with k3:
    if "Sales" in filtered.columns:
        total_sales = filtered["Sales"].sum()
        st.markdown(f"<div class='kpi'><div class='small'>Total Sales</div><div class='neon' style='font-size:20px'>₹{total_sales:,.2f}</div></div>", unsafe_allow_html=True)
    elif "Order Item Total" in filtered.columns:
        total_sales = filtered["Order Item Total"].sum()
        st.markdown(f"<div class='kpi'><div class='small'>Total Sales</div><div class='neon' style='font-size:20px'>₹{total_sales:,.2f}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='kpi'><div class='small'>Total Sales</div><div class='neon' style='font-size:20px'>N/A</div></div>", unsafe_allow_html=True)
with k4:
    if "Days for shipping (real)" in filtered.columns:
        avg_days = filtered["Days for shipping (real)"].dropna().astype(float).mean()
        st.markdown(f"<div class='kpi'><div class='small'>Avg Shipping Days</div><div class='neon' style='font-size:20px'>{avg_days:.2f}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='kpi'><div class='small'>Avg Shipping Days</div><div class='neon' style='font-size:20px'>N/A</div></div>", unsafe_allow_html=True)

st.markdown("---")

# Tabs area
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🤖 Model & Prediction", "📁 Data Preview"])

# -----------------------
# Tab 1: Analytics
# -----------------------
with tab1:
    st.header("Shipping & Delay Analytics")
    # Row: Pie for delay vs on-time, bar for shipping mode
    a1, a2 = st.columns((1,2))
    with a1:
        st.subheader("On-time vs Delayed")
        if "Late_delivery_risk" in filtered.columns:
            pie = px.pie(filtered, names="Late_delivery_risk", hole=0.4, title="On-time (0) vs Delayed (1)",
                         color_discrete_sequence=["#00e6ff", "#ff4d6d"])
            pie.update_traces(textinfo="percent+label")
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No Late_delivery_risk column available.")
    with a2:
        st.subheader("Delay % by Shipping Mode")
        if "Shipping Mode" in filtered.columns and "Late_delivery_risk" in filtered.columns:
            mode = filtered.groupby("Shipping Mode").agg(total=("Late_delivery_risk", "count"),
                                                        delayed=("Late_delivery_risk", "sum")).reset_index()
            mode["delay_pct"] = mode["delayed"] / mode["total"] * 100
            fig = px.bar(mode.sort_values("delay_pct", ascending=False), x="Shipping Mode", y="delay_pct",
                         color="delay_pct", color_continuous_scale=["#00e6ff", "#004d66"], title="Delay % by Shipping Mode",
                         template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need Shipping Mode & Late_delivery_risk for this chart.")

    st.markdown("---")
    # Row: Top cities by delay %
    st.subheader("Top Cities by Delay %")
    if "Order City" in filtered.columns and "Late_delivery_risk" in filtered.columns:
        city_stats = filtered.groupby("Order City").agg(total=("Late_delivery_risk", "count"),
                                                        delayed=("Late_delivery_risk", "sum")).reset_index()
        city_stats = city_stats[city_stats["total"] >= 10]  # ignore tiny groups
        city_stats["delay_pct"] = city_stats["delayed"] / city_stats["total"] * 100
        city_stats = city_stats.sort_values("delay_pct", ascending=False).head(20)
        fig_city = px.bar(city_stats, x="delay_pct", y="Order City", orientation="h", color="delay_pct",
                          title="Top Cities by Delay %", template="plotly_dark")
        st.plotly_chart(fig_city, use_container_width=True)
    else:
        st.info("City or target missing for city breakdown.")

    st.markdown("---")
    # Row: Monthly trend
    st.subheader("Monthly Orders & Delays")
    if "order_month" in filtered.columns:
        monthly = filtered.groupby("order_month").agg(orders=("Late_delivery_risk", "count"),
                                                      delayed=("Late_delivery_risk", "sum")).reset_index().sort_values("order_month")
        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(x=monthly["order_month"], y=monthly["orders"], name="Orders"))
        fig_month.add_trace(go.Scatter(x=monthly["order_month"], y=monthly["delayed"], mode="lines+markers", name="Delayed"))
        fig_month.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.info("No order_date/order_month column to show monthly trend.")

    st.markdown("---")
    # Discount vs Delay scatter
    st.subheader("Discount vs Delay")
    if "Order Item Discount" in filtered.columns and "Late_delivery_risk" in filtered.columns:
        s = filtered.dropna(subset=["Order Item Discount", "Late_delivery_risk"])
        if len(s) > 0:
            fig_sc = px.scatter(s, x="Order Item Discount", y="Late_delivery_risk", color="Late_delivery_risk",
                                hover_data=["Order City", "Category Name"], template="plotly_dark", title="Discount vs Delay")
            fig_sc.update_yaxes(tickvals=[0,1], ticktext=["On-time","Delayed"])
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Not enough discount data.")
    else:
        st.info("Discount or Late_delivery_risk not available for this chart.")

# -----------------------
# Tab 2: Model & Prediction
# -----------------------
with tab2:
    st.header("Model training, metrics & prediction")
    model_loaded = load_model()
if model_loaded is None:
    st.info("No model currently loaded — please train one before prediction.")

    metrics = load_metrics()

    colt1, colt2 = st.columns([2, 1])
    with colt1:
        st.subheader("Train model")
        st.markdown("Train a Logistic Regression pipeline on available sensible features.")
        st.info("Training will automatically choose available features (numeric + categorical).")
        if train_btn:
            # Training workflow
            st.info("Training started — please wait...")
            try:
                feats = choose_features(df)
                if len(feats) < 2:
                    st.error("Not enough sensible features detected to train a model. Edit the dataset or add features.")
                else:
                    df["Late_delivery_risk"] = pd.to_numeric(df["Late_delivery_risk"], errors="coerce")
                    df["Late_delivery_risk"] = df["Late_delivery_risk"].fillna(0).astype(int)

# ✅ Proceed with training
                    X = df.drop("Late_delivery_risk", axis=1)
                    y = df["Late_delivery_risk"]

                    # simple dropna rows for training (alternatively impute)
                    # We'll allow imputation in pipeline; don't drop too much
                    pipe, numeric_cols, categorical_cols = build_pipeline(X)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
                    pipe.fit(X_train, y_train)
                    new_metrics = model_evaluate(pipe, X_test, y_test)
                    # save
                    save_model(pipe, {"metrics": new_metrics, "features": list(X.columns), "trained_at": str(datetime.utcnow())})
                    st.success("Model trained and saved to models/delay_model.pkl")
                    st.json({"accuracy": new_metrics["accuracy"], "roc_auc": new_metrics["roc_auc"]})
                    model_loaded = pipe
                    metrics = {"metrics": new_metrics, "features": list(X.columns)}
            except Exception as e:
                st.exception(e)

    with colt2:
        st.subheader("Model info / Actions")
        if metrics:
            try:
                m = metrics.get("metrics", metrics)  # support older structure
                acc = m.get("accuracy") or m.get("accuracy")
                auc = m.get("roc_auc")
            except Exception:
                acc = None
                auc = None
        else:
            acc = None
            auc = None

        if view_metrics_btn:
            if os.path.exists(METRICS_PATH):
                with st.expander("View saved metrics"):
                    mm = load_metrics()
                    st.write(mm)
            else:
                st.info("No metrics file saved yet. Train the model to generate metrics.")

        if model_loaded is not None:
            st.markdown("**Trained model loaded.**")
            st.button("Show confusion matrix", key="show_cm")
            # If user clicked earlier, display CM if available
            if os.path.exists(METRICS_PATH):
                saved = load_metrics()
                conf = saved.get("metrics", {}).get("confusion_matrix") or saved.get("confusion_matrix")
                if conf:
                    fig = px.imshow(conf, text_auto=True, color_continuous_scale="blues")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Metrics not found — train model to see confusion matrix.")

        else:
            st.info("Model not loaded. Train model above to create one.")

    st.markdown("---")
    st.subheader("Predict a single shipment")
    st.markdown("Fill fields below and click **Predict**. The app will try to match model features automatically.")
    # Build predictive form using defaults from dataset sample
    sample = df.sample(200, replace=True) if len(df) > 200 else df
    # Determine training features if known
    trained_features = None
    loaded_metrics = load_metrics()
    if loaded_metrics:
        trained_features = loaded_metrics.get("features") or loaded_metrics.get("metrics", {}).get("features")
    # fallback choose sensible features
    sensible = choose_features(df)
    features_for_form = trained_features if trained_features else sensible

    # build dynamic form
    with st.form("predict_form"):
        input_dict = {}
        for feat in features_for_form:
            if feat in df.columns:
                if pd.api.types.is_numeric_dtype(df[feat]):
                    default_val = float(df[feat].median(skipna=True)) if df[feat].notna().any() else 0.0
                    input_dict[feat] = st.number_input(feat, value=default_val)
                else:
                    options = sorted(df[feat].dropna().astype(str).unique().tolist())
                    default = options[0] if options else ""
                    input_dict[feat] = st.selectbox(feat, options, index=0)
        predict_clicked = st.form_submit_button("Predict")

    if predict_clicked:
        if model_loaded is None:
            st.error("No trained model available. Use Train Model to create a model first.")
        else:
            # create DataFrame for prediction
            Xpred = pd.DataFrame([input_dict])
            # ensure model's preprocessor required columns exist
            try:
                pre = model_loaded.named_steps.get("preprocessor")
                req_cols = []
                if pre is not None:
                    for t in pre.transformers:
                        cols = t[2]
                        if isinstance(cols, (list, tuple)):
                            req_cols.extend(cols)
                # add missing cols with defaults
                for c in req_cols:
                    if c not in Xpred.columns:
                        # if numeric in df, use median, else blank
                        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                            Xpred[c] = float(df[c].median(skipna=True))
                        else:
                            Xpred[c] = ""
                proba = model_loaded.predict_proba(Xpred)[0][1] if hasattr(model_loaded, "predict_proba") else None
                pred = model_loaded.predict(Xpred)[0]
                if proba is not None:
                    st.metric("Delay probability", f"{proba:.3f}")
                st.markdown(f"## Prediction: {'🚨 DELAY' if pred==1 else '✅ ON-TIME'}")
                # optimization suggestions
                suggestions = optimization_suggestions(Xpred.iloc[0])
                st.subheader("Optimization suggestions")
                for s in suggestions:
                    st.write("- " + s)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------
# Tab 3: Data Preview
# -----------------------
with tab3:
    st.header("Data Preview & Schema")
    st.markdown("Preview the (filtered) dataset below and explore column types.")
    st.dataframe(filtered.head(500), use_container_width=True)
    st.markdown("### Columns & dtypes")
    cinfo = pd.DataFrame({"column": filtered.columns, "dtype": [str(filtered[c].dtype) for c in filtered.columns]})
    st.dataframe(cinfo, use_container_width=True)
    st.markdown("---")
    st.markdown("You can download the filtered dataset from the sidebar (Download Filtered CSV).")

# -----------------------
# Footer / tips
# -----------------------
st.markdown("---")
st.markdown("Built with Streamlit • Logistic Regression model • Heuristic AI-style suggestions")
st.caption("Tip: If prediction errors mention missing columns, re-run Train Model; the training script auto-detects available features.")
