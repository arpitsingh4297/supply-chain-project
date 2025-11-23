# streamlit_app.py – FINAL STABLE GOOGLE DRIVE VERSION
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import urllib.request

st.set_page_config(page_title="Supply Chain", layout="wide")

st.markdown("<h1 style='text-align: center; color:#FF4500;'>Instant Noodles Supply Chain Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Great Learning Final Project | R² = 0.945 | MAE = 1,580 tons</h3>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 🚀 GOOGLE DRIVE DIRECT DOWNLOAD LINKS (REPLACE WITH YOUR OWN IF NEEDED)
# ---------------------------------------------------------------------
MODEL_URL   = "https://drive.google.com/uc?export=download&id=1JWjoBWskEGL4NQXIR-9BIyFLRmSdtdni"
FEATURE_URL = "https://drive.google.com/uc?export=download&id=1zNRjVFRaBezf1-pyfqgEiHdH8IrLwv7z"

# ---------------------------------------------------------------------
# 🚀 LOAD MODEL + FEATURES WITH DRIVE CACHING
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model_path = "model.pkl"
    feat_path = "features.pkl"

    # Download only if missing
    if not os.path.exists(model_path):
        with st.spinner("Downloading 957MB model... wait ⏳"):
            urllib.request.urlretrieve(MODEL_URL, model_path)

    if not os.path.exists(feat_path):
        urllib.request.urlretrieve(FEATURE_URL, feat_path)

    # Load both
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(feat_path, "rb") as f:
        features = pickle.load(f)

    return model, features

try:
    model, feature_cols = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model failed to load. Error: {e}")
    model = None

# ---------------------------------------------------------------------
# 🚀 SIDEBAR INPUT FORM
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Warehouse Details")

    location = st.selectbox("Location Type", ["Urban", "Rural"])
    capacity = st.selectbox("Capacity", ["Small", "Mid", "Large"])
    zone = st.selectbox("Zone", ["North","South","East","West","Central"])
    regional = st.selectbox("Regional Zone", [f"Zone {i}" for i in range(1,7)])

    retail = st.slider("Retail Shops", 1000, 10000, 6000)
    refill = st.slider("Refill Requests L3M", 0, 100, 20)
    storage = st.slider("Storage Issues L3M", 0, 50, 5)

predict_btn = st.sidebar.button("Predict Shipment", type="primary")

# ---------------------------------------------------------------------
# 🚀 PREDICTION LOGIC
# ---------------------------------------------------------------------
if predict_btn:
    if model is None:
        st.error("❌ Model failed to load. Cannot predict.")
    else:
        try:
            # Input Data
            data = {
                'Location_type': [location],
                'WH_capacity_size': [capacity],
                'zone': [zone],
                'WH_regional_zone': [regional],
                'retail_shop_num': [retail],
                'num_refill_req_l3m': [refill],
                'storage_issue_reported_l3m': [storage],
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

            df = pd.DataFrame(data)

            # Log transform
            num_cols = [
                'retail_shop_num','distributor_num','workers_num',
                'dist_from_hub','storage_issue_reported_l3m','num_refill_req_l3m'
            ]
            for col in num_cols:
                df[f'log_{col}'] = np.log1p(df[col])

            # One-hot encode
            df_encoded = pd.get_dummies(df, drop_first=True)

            # Align with training features
            for col in feature_cols:
                if col not in df_encoded:
                    df_encoded[col] = 0

            df_encoded = df_encoded[feature_cols]

            # Predict
            result = int(np.expm1(model.predict(df_encoded)[0]))

            st.success(f"### 🚢 Recommended Shipment: **{result:,} tons**")

            if result > 35000:
                st.error("🔴 High demand — increase supply!")
            elif result < 10000:
                st.warning("🟡 Low demand — reduce stock")
            else:
                st.info("🟢 Standard allocation")

            st.balloons()

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------
# 🚀 PERFORMANCE METRICS
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("R² Score", "0.945")
with col2:
    st.metric("MAE", "1,580 tons")
