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

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else os.path.abspath('..')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils import load_dataset, log_based_split, compute_metrics, TARGET_COLUMNS, TARGET_LABELS
from src.feature_engineering import engineer_features, get_feature_columns

# %% [markdown]
# ## 1. Load Dataset & Feature Engineering Pipeline

# %%
df_raw = load_dataset()
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
models_dir = os.path.join(repo_root, "saved_models")
models = {}
scalers = {}
for name, m_name, s_name in [
    ('Ridge', 'ridge_model.pkl', 'ridge_scaler.pkl'),
    ('Random Forest', 'rf_model.pkl', 'rf_scaler.pkl'),
    ('Gradient Boosting', 'gbm_model.pkl', 'gbm_scaler.pkl')
]:
    m_path = os.path.join(models_dir, m_name)
    s_path = os.path.join(models_dir, s_name)
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

