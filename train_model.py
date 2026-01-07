import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import json
import numpy as np
import os

# 1. Create model folder
os.makedirs("model", exist_ok=True)

# 2. Load data
print("📊 Loading data...")
data = pd.read_csv('ecommerce_data.csv')  # Your file
print(f"✅ Loaded {len(data)} records")

# 3. Encode categoricals
print("🔤 Encoding...")
le_brand = LabelEncoder().fit(data['brand'])
le_category = LabelEncoder().fit(data['category'])
data['brand_enc'] = le_brand.transform(data['brand'])
data['category_enc'] = le_category.transform(data['category'])

# 4. Prepare features
X = data[['brand_enc', 'category_enc', 'discount']]  # Adjust columns
y = data['price']

# 5. Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("🤖 Training...")

# Baseline
baseline = LinearRegression().fit(X_train, y_train)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline.predict(X_test)))

# XGBoost
xgboost = XGBRegressor(n_estimators=100).fit(X_train, y_train)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgboost.predict(X_test)))

# 6. Save
joblib.dump(xgboost, 'model/price_model.pkl')
joblib.dump(le_brand, 'model/brand_encoder.pkl')
joblib.dump(le_category, 'model/category_encoder.pkl')

# Metrics
metrics = {
    "baseline": {"rmse": float(baseline_rmse), "r2": float(r2_score(y_test, baseline.predict(X_test)))},
    "xgboost": {"rmse": float(xgb_rmse), "r2": float(r2_score(y_test, xgboost.predict(X_test)))}
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ COMPLETE!")
print(f"📈 XGBoost RMSE: {xgb_rmse:.0f} | R²: {r2_score(y_test, xgboost.predict(X_test)):.2f}")
