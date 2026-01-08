import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import numpy as np
import os

# Create model directory
os.makedirs('model', exist_ok=True)

# Sample data (replace with your ecommerce_sales.csv)
print("📊 Creating sample training data...")
np.random.seed(42)
n_samples = 1000

data = {
    'brand': np.random.choice(['Samsung', 'Apple', 'Xiaomi', 'OnePlus'], n_samples),
    'category': np.random.choice(['mobile', 'laptop', 'tablet', 'watch'], n_samples),
    'discount': np.random.uniform(0, 90, n_samples),
    'price': np.random.normal(25000, 15000, n_samples)
}
data['price'] = np.clip(data['price'], 5000, 100000)

df = pd.DataFrame(data)
print(f"✅ Dataset ready: {len(df)} samples")

# Encode features
brand_encoder = LabelEncoder()
category_encoder = LabelEncoder()

df['brand_enc'] = brand_encoder.fit_transform(df['brand'])
df['category_enc'] = category_encoder.fit_transform(df['category'])

# Train model
X = df[['brand_enc', 'category_enc', 'discount']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ XGBoost RMSE: {rmse:.0f} | R²: {r2:.2f}")

# Save model and encoders
joblib.dump(model, 'model/price_model.pkl')
joblib.dump(brand_encoder, 'model/brand_encoder.pkl')
joblib.dump(category_encoder, 'model/category_encoder.pkl')

# Update metrics.json
metrics = {
    "xgboost": {
        "model": "XGBoostRegressor",
        "rmse": float(rmse),
        "r2": float(r2)
    }
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("🎉 ALL FILES SAVED!")
print("✅ model/price_model.pkl")
print("✅ model/brand_encoder.pkl") 
print("✅ model/category_encoder.pkl")
print("✅ metrics.json updated")
