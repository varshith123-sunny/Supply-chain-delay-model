# predictor.py — prediction UI and model accuracy button
import streamlit as st
import pandas as pd
import joblib
import os
import json
from sklearn.exceptions import NotFittedError

MODEL_PATH = "models/delay_model.pkl"
METRICS_PATH = "models/model_metrics.json"
DATA_PATH = "data/DataCoSupplyChainDataset_no_zeros.csv"

def run_predictor():
    st.title("🤖 Delay Predictor")
    st.markdown("Provide shipment details and click **Predict**. Use **View Model Accuracy** to see trained model metrics.")

    # attempt to load model & metrics
    model = None
    metrics = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error("Failed to load model: " + str(e))
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    st.sidebar.markdown("### Predictor Actions")
    if st.sidebar.button("View Model Accuracy"):
        if metrics:
            st.sidebar.write("**Accuracy:**", metrics.get("accuracy"))
            st.sidebar.write("**ROC AUC:**", metrics.get("roc_auc"))
            st.sidebar.write("**Features used:**", metrics.get("features"))
            # show classification report small
            report = metrics.get("report")
            if report:
                st.sidebar.write("Classification report (summary):")
                # print classwise precision/recall if available
                for k, v in report.items():
                    if k in ["0","1"]:
                        st.sidebar.write(f"Class {k}: precision {v.get('precision'):.2f}, recall {v.get('recall'):.2f}, f1 {v.get('f1-score'):.2f}")
        else:
            st.sidebar.warning("No metrics found. Train model using model_training.py")

    # Build form fields based on common features (best-effort)
    st.subheader("Input shipment details")
    df_sample = pd.read_csv(DATA_PATH, encoding="latin1", nrows=50)
    df_sample.columns = [c.strip() for c in df_sample.columns]
    # choose defaults from sample if possible
    def default(col, fallback=""):
        return df_sample[col].dropna().iloc[0] if col in df_sample.columns and df_sample[col].dropna().shape[0]>0 else fallback

    ocity = st.text_input("Order City", value=default("Order City",""))
    ostate = st.text_input("Order State", value=default("Order State",""))
    cat = st.text_input("Category Name", value=default("Category Name",""))
    qty = st.number_input("Order Item Quantity", min_value=1, value=int(default("Order Item Quantity",1)))
    discount = st.number_input("Order Item Discount", min_value=0.0, value=float(default("Order Item Discount",0.0)))
    price = st.number_input("Product Price", min_value=0.0, value=float(default("Product Price",100.0)))

    if st.button("Predict"):
        if model is None:
            st.error("No trained model found. Run model_training.py to train and save model into models/")
        else:
            inp = pd.DataFrame([{
                "Order City": ocity,
                "Order State": ostate,
                "Category Name": cat,
                "Order Item Quantity": qty,
                "Order Item Discount": discount,
                "Product Price": price
            }])
            # Ensure model's required columns exist
            try:
                # try to discover required feature names
                pre = model.named_steps.get("preprocessor")
                req_cols = []
                if pre is not None:
                    for t in pre.transformers:
                        cols = t[2]
                        if isinstance(cols, (list,tuple)):
                            req_cols.extend(cols)
                # add missing with defaults
                for c in req_cols:
                    if c not in inp.columns:
                        if "Quantity" in c or "Price" in c or "Discount" in c:
                            inp[c] = 0
                        else:
                            inp[c] = ""
                pred = model.predict(inp)[0]
                proba = model.predict_proba(inp)[0][1] if hasattr(model, "predict_proba") else None
                if pred == 1:
                    st.error(f"Prediction: DELAY (prob {proba:.2f})" if proba is not None else "Prediction: DELAY")
                else:
                    st.success(f"Prediction: ON-TIME (prob {proba:.2f})" if proba is not None else "Prediction: ON-TIME")
            except NotFittedError:
                st.error("Model not fitted. Retrain model.")
            except Exception as e:
                st.error("Prediction error: " + str(e))

