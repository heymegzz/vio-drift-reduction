# %% [markdown]
# # Phase 1: Machine Learning Model Benchmark
# ### ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation
#
# This script/notebook demonstrates:
# 1. Loading the telemetry dataset and generating 142 leakage-safe IMU features
# 2. Applying whole-flight log splitting (`log_51`+`log_53` train, `log_52` val, `log_54` test)
# 3. Evaluating Ridge Regression, Random Forest, and Gradient Boosting
# 4. Analyzing velocity residual predictions across North, East, and Down axes
# 5. Inspecting Random Forest feature importances

# %%
import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('..'))

from models.utils import load_dataset, log_based_split, compute_metrics, TARGET_COLUMNS, TARGET_LABELS
from feature_engineering import engineer_features, get_feature_columns

# %% [markdown]
# ## 1. Load Dataset & Feature Engineering Pipeline

# %%
df_raw = load_dataset('../imu_drift_feature_dataset.csv')
print(f"Raw dataset: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

df_featured = engineer_features(df_raw, verbose=True)
feature_cols = get_feature_columns(df_featured)
print(f"Generated {len(feature_cols)} input features")

# %% [markdown]
# ## 2. Log-Based Partitioning

# %%
df_train, df_val, df_test = log_based_split(df_featured)
X_test = df_test[feature_cols].values
y_test = df_test[TARGET_COLUMNS].values
t_test = df_test['time_s'].values
print(f"Test set (log_54): {len(df_test):,} samples")

# %% [markdown]
# ## 3. Load Trained Model Checkpoints & Evaluate

# %%
models = {}
scalers = {}
for name, m_path, s_path in [
    ('Ridge', '../saved_models/ridge_model.pkl', '../saved_models/ridge_scaler.pkl'),
    ('Random Forest', '../saved_models/rf_model.pkl', '../saved_models/rf_scaler.pkl'),
    ('Gradient Boosting', '../saved_models/gbm_model.pkl', '../saved_models/gbm_scaler.pkl')
]:
    with open(m_path, 'rb') as f:
        models[name] = pickle.load(f)
    with open(s_path, 'rb') as f:
        scalers[name] = pickle.load(f)

results = []
predictions = {}
for name, model in models.items():
    X_s = scalers[name].transform(X_test)
    y_pred = model.predict(X_s)
    predictions[name] = y_pred
    m = compute_metrics(y_test, y_pred)
    m['Model'] = name
    results.append(m)

df_results = pd.DataFrame(results).sort_values('rmse_mean')
print(df_results[['Model', 'rmse_mean', 'mae_mean', 'r2_mean', 'rmse_N', 'rmse_E', 'rmse_D']])
