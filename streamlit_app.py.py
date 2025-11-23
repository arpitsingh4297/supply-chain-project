# ============================
# streamlit_app.py – FINAL DEPLOYMENT VERSION (HuggingFace URLs)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import urllib.request
import os

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")

st.markdown(
    "<h1 style='text-align:center;color:#FF4500;'>Instant Noodles Supply Chain Optimizer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align:center;'>Great Learning Final Project | R² = 0.945 | MAE = 1,580 tons</h3>",
    unsafe_allow_html=True
)

# -------------------------
# HUGGINGFACE DIRECT DOWNLOAD LINKS (CORRECT)
# -------------------------
MODEL_URL = "https://huggingface.co/arpitsingh4297/supply-chain-model/resolve/main/supply_chain_model.pkl"
FEATURE_URL = "https://huggingface.co/arpitsingh4297/supply-chain-model/resolve/main/feature_columns.pkl"

# -------------------------
# MODEL LOADER (CACHED)
# -------------------------
@st.cache_resource
def load_model_files():
    model_path = "supply_chain_model.pkl"
    feature_path = "feature_columns.pkl"

    # Download model if missing
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from HuggingFace..."):
            urllib.request.urlretrieve(MODEL_URL, model_path)

    if not os.path.exists(feature_path):
        urllib.request.urlretrieve(FEATURE_URL, feature_path)

    # Load model + columns
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(feature_path, "rb") as f:
        feature_cols = pickle.load(f)

    return model, feature_cols


# -------------------------
# TRY LOADING MODEL
# -------------------------
try:
    model, feature_cols = load_model_files()
    st.success("Model loaded successfully from HuggingFace!")
except Exception as e:
    st.error("❌ Model failed to load. Check HuggingFace file permissions.")
    st.stop()


# ============================
# SIDEBAR INPUTS
# ============================
with st.sidebar:
    st.header("Warehouse Details")

    location = st.selectbox("Location Type", ["Urban", "Rural"])
    capacity = st.selectbox("Capacity", ["Small", "Mid", "Large"])
    zone = st.selectbox("Zone", ["North", "South", "East", "West", "Central"])
    regional = st.selectbox("Regional Zone", [f"Zone {i}" for i in range(1, 7)])

    retail = st.slider("Retail Shops", 1000, 10000, 6000)
    refill = st.slider("Refill Requests (L3M)", 0, 100, 20)
    storage = st.slider("Storage Issues (L3M)", 0, 50, 5)

    predict_btn = st.button("Predict Shipment", type="primary")


# ============================
# PREDICTION BLOCK
# ============================
if predict_btn:
    try:
        # Raw input dictionary
        input_data = {
            'Location_type': [location],
            'WH_capacity_size': [capacity],
            'zone': [zone],
            'WH_regional_zone': [regional],
            'retail_shop_num': [retail],
            'num_refill_req_l3m': [refill],
            'storage_issue_reported_l3m': [storage],

            # Static features used during training
            'workers_num': [35],
            'dist_from_hub': [120],
            'Competitor_in_mkt': [3],
            'flood_proof': [1],
            'electric_supply': [1],
            'temp_reg_mach': [1],
            'approved_wh_govt_certificate': ['A'],
            'distributor_num': [15],
            'warehouse_age': [10],
            'wh_owner_type': ['Owned'],
            'flood_impacted': [0]
        }

        df = pd.DataFrame(input_data)

        # Log features
        for col in [
            'retail_shop_num', 'distributor_num', 'workers_num',
            'dist_from_hub', 'storage_issue_reported_l3m',
            'num_refill_req_l3m'
        ]:
            df[f'log_{col}'] = np.log1p(df[col])

        # One-hot encode
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Align all columns with the trained model
        for col in feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[feature_cols]

        # Predict → inverse log
        pred_log = model.predict(df_encoded)[0]
        tons = int(np.expm1(pred_log))

        # Output
        st.success(f"### Recommended Shipment: **{tons:,} tons**")

        if tons > 35000:
            st.error("HIGH DEMAND → Increase supply!")
        elif tons < 10000:
            st.warning("Low demand → Reduce stock")
        else:
            st.info("Standard allocation")

        st.balloons()

    except Exception as e:
        st.error("❌ Prediction failed. Check your model & input pipeline.")


# ============================
# PERFORMANCE METRICS
# ============================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² Score", "0.945")
with col2:
    st.metric("MAE", "1,580 tons")
with col3:
    st.metric("Status", "Ready")
