import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib

st.title("E-commerce Price Predictor PRO")
st.markdown("Discount Demand Trends XAI Batch")

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("ecommerce_sales.csv")
        st.success(f"✅ Production Dataset Loaded: {len(df):,} records")
        st.dataframe(df.head(5))
        return df
    except:
        st.warning("📥 CSV filename check karo!")
        return pd.DataFrame()

df = load_dataset()

if not df.empty:
    st.metric("Avg Price", f"₹{df['price'].mean():.0f}")
    st.metric("Total Sales", f"₹{df['revenue'].sum():.0f}")
competitor_price = st.number_input("Competitor Price", 10000, 100000, 25000)
brand = st.text_input("Brand", "samsung")
category = st.text_input("Category", "mobile")

# FLIPKART STYLE DEAL FINDER
product_name = st.text_input("🔍 Product name", "iPhone 15")
if st.button("Find Best Deal"):
    deals = pd.DataFrame({
    "Platform": ["Flipkart", "Amazon", "Myntra"],
    "Original": [35000, 38000, 34000],
    "Discount": [15, 10, 18],
    "Final": [29750, 34200, 27880]
})
    best_deal = deals.loc[deals["Final"].idxmin()]
    st.success(f"🎉 Best Deal: {best_deal['Platform']} - Rs{best_deal['Final']}")
    st.dataframe(deals.style.highlight_min(subset="Final", color="lightgreen"))

csv_file = st.sidebar.file_uploader("CSV Batch", type="csv")
# Global model storage
if 'model' not in st.session_state:
    st.session_state.model = None

st.subheader("🤖 XGBoost Production")
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Safe defaults
rmse = 0
r2_score = 0
X_cols = ['discount', 'spend']  # Fallback
model = None

if len(numeric_cols) >= 3:  # Need target + 2 features
    X_cols = numeric_cols[1:3]  # Skip price as target
    y_col = numeric_cols[0]     # Price as target
    
    X = df[X_cols].fillna(0)
    y = df[y_col].fillna(df[y_col].mean())
    
    model = xgb.XGBRegressor(n_estimators=25)
    model.fit(X, y)
    
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2_score = model.score(X, y)
    
    st.session_state.model = model
    st.success(f"✅ Trained! RMSE: ₹{rmse:.0f}")

# Leaderboard always safe
st.subheader("🏆 Leaderboard")
leaderboard = pd.DataFrame({
    "Model": ["XGBoost", "Baseline"],
    "RMSE": [rmse, df['price'].std() if 'price' in df.columns else 1000],
    "R²": [r2_score, 0],
    "Status": ["✅ Ready" if model else "⚠️ Data needed", "📉"]
})
st.dataframe(leaderboard)

# Live Prediction (Safe + Balloons)
st.subheader("🎯 Live ML Prediction")
if model and len(X_cols) == 2:
    col1, col2 = st.columns(2)
    feat1_val = col1.slider(f"📊 {X_cols[0]}", 0.0, 100.0, 50.0)
    feat2_val = col2.slider(f"💰 {X_cols[1]}", 0.0, 1000.0, 100.0)
    
    if st.button("🔮 **PREDICT PRICE**", use_container_width=True):
        test_X = pd.DataFrame({X_cols[0]: [feat1_val], X_cols[1]: [feat2_val]})
        prediction = model.predict(test_X)[0]
        st.metric("Predicted Price", f"₹{prediction:.0f}")
        st.balloons()  # 🎉
else:
    st.info("📊 3+ numeric columns chahiye")

# SHAP (Only after prediction)
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

if st.session_state.prediction_made and model:
    st.subheader("🔍 SHAP Explanation")
    try:
        import shap
        explainer = shap.Explainer(model)
        shap_values = explainer(test_X)
        st.shap(shap.plots.waterfall(shap_values[0]))
    except:
        st.info("SHAP: pip install shap")

# CSV Batch
if csv_file:
    csv_df = pd.read_csv(csv_file)
    if model and len(X_cols) == 2:
        csv_df["predicted"] = model.predict(csv_df[X_cols].fillna(0))
        st.dataframe(csv_df[["predicted"]].head())
    else:
        st.info("Model ready hone ke baad CSV predict karo")
