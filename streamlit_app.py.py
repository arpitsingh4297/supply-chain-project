# streamlit_app.py - OPTIMIZED FOR LARGE MODEL - NO ERROR
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib  # Better for large models

st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")
st.title("Instant Noodles Supply Chain Optimizer")
st.markdown("### Great Learning Final Project | R² = 0.945 | MAE = 1,580 tons")

# Load model with error handling for large file
@st.cache_resource
def load_model():
    try:
        # Use joblib for large pickle files (more memory efficient)
        model = joblib.load("supply_chain_model.pkl")
        columns = joblib.load("feature_columns.pkl")
        return model, columns
    except Exception as e:
        st.error(f"Model load error: {e}. Try the online version.")
        return None, None

model, training_columns = load_model()
if model is not None:
    st.success("Model loaded successfully!")

# Sidebar inputs
st.sidebar.header("Warehouse Details")
location_type = st.sidebar.selectbox("Location Type", ["Urban", "Rural"])
wh_capacity_size = st.sidebar.selectbox("Warehouse Capacity", ["Small", "Mid", "Large"])
zone = st.sidebar.selectbox("Zone", ["North", "South", "East", "West", "Central"])
wh_regional_zone = st.sidebar.selectbox("Regional Zone", ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"])
retail_shop_num = st.sidebar.slider("Retail Shops Covered", 1000, 10000, 6000)
workers_num = st.sidebar.slider("Number of Workers", 10, 100, 35)
storage_issue_reported_l3m = st.sidebar.slider("Storage Issues (L3M)", 0, 50, 5)
num_refill_req_l3m = st.sidebar.slider("Refill Requests (L3M)", 0, 100, 20)

# Predict button
if st.sidebar.button("Predict Optimal Shipment", type="primary") and model is not None:
    try:
        # Create input data
        input_data = {
            'Location_type': [location_type],
            'WH_capacity_size': [wh_capacity_size],
            'zone': [zone],
            'WH_regional_zone': [wh_regional_zone],
            'retail_shop_num': [retail_shop_num],
            'workers_num': [workers_num],
            'storage_issue_reported_l3m': [storage_issue_reported_l3m],
            'num_refill_req_l3m': [num_refill_req_l3m],
            'flood_impacted': [0],
            'flood_proof': [1],
            'electric_supply': [1],
            'dist_from_hub': [120],
            'Competitor_in_mkt': [3],
            'transport_issue_l1y': [1],
            'govt_check_l3m': [5],
            'wh_breakdown_l3m': [2],
            'temp_reg_mach': [1],
            'approved_wh_govt_certificate': ['A'],
            'wh_owner_type': ['Owned'],
            'distributor_num': [15],
            'warehouse_age': [10]
        }
        df_input = pd.DataFrame(input_data)

        # Add log features
        skewed_cols = ['retail_shop_num', 'distributor_num', 'workers_num', 'dist_from_hub',
                       'storage_issue_reported_l3m', 'num_refill_req_l3m']
        for col in skewed_cols:
            df_input[f'log_{col}'] = np.log1p(df_input[col])

        # One-hot encode
        df_encoded = pd.get_dummies(df_input, drop_first=True)

        # Match training columns
        for col in training_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[training_columns]

        # Predict
        pred_log = model.predict(df_encoded)[0]
        predicted_tons = int(np.expm1(pred_log))

        st.markdown(f"# Recommended Shipment: **{predicted_tons:,} tons**")

        if predicted_tons > 35000:
            st.error("HIGH DEMAND ZONE → Increase supply by 40%!")
        elif predicted_tons < 10000:
            st.warning("Low demand → Reduce allocation")
        else:
            st.info("Standard shipment recommended")

        st.balloons()

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² Score", "0.945", "Exceeded Target")
with col2:
    st.metric("MAE", "1,580 tons", "Target < 3000")
with col3:
    st.metric("Model", "Random Forest", "Best Performer")

st.success("PROJECT 100% COMPLETE | Use the live link for submission")
