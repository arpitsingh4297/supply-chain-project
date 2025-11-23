# save_columns.py – run only once
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

print("Loading data...")
df = pd.read_csv("C:/Users/arpit/Documents/Supply+Chain+Project/Data.csv")


# --- SAME CLEANING AS BEFORE ---
df['wh_est_year'] = pd.to_numeric(df['wh_est_year'], errors='coerce')
df['warehouse_age'] = 2025 - df['wh_est_year']
df['warehouse_age'].fillna(df['warehouse_age'].median(), inplace=True)
df['approved_wh_govt_certificate'].replace('NA', np.nan, inplace=True)
df['approved_wh_govt_certificate'].fillna(df['approved_wh_govt_certificate'].mode()[0], inplace=True)
df['workers_num'].fillna(df['workers_num'].median(), inplace=True)
df.drop(['Ware_house_ID', 'WH_Manager_ID', 'wh_est_year'], axis=1, inplace=True)

# Log features
for col in ['retail_shop_num','distributor_num','workers_num','dist_from_hub',
            'storage_issue_reported_l3m','num_refill_req_l3m']:
    df[f'log_{col}'] = np.log1p(df[col])

y = np.log1p(df['product_wg_ton'])
X = df.drop('product_wg_ton', axis=1)
X = pd.get_dummies(X, drop_first=True)

# Save exact column order
pickle.dump(X.columns.tolist(), open("feature_columns.pkl", "wb"))
print("feature_columns.pkl saved!")

# Train & save model
model = RandomForestRegressor(n_estimators=1000, max_depth=22, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
model.fit(X, y)
pickle.dump(model, open("supply_chain_model.pkl", "wb"))
print("supply_chain_model.pkl saved!")
print("All files ready – you can now run the dashboard!")