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
if len(numeric_cols) > 1:
    X_cols = numeric_cols[:2]
    y_col = numeric_cols[0]
    
    X = df[X_cols].fillna(0)
    y = df[y_col].fillna(df[y_col].mean())
    
    # Train & store
    st.session_state.model = xgb.XGBRegressor(n_estimators=25)
    st.session_state.model.fit(X, y)
    model = st.session_state.model  # Reference
    
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    st.success(f"✅ Live! RMSE: ₹{rmse:.0f}")
    
    # Safe download
    try:
        joblib.dump(model, "model.pkl")
        st.download_button("💾 Model", open("model.pkl","rb"), "model.pkl")
    except:
        st.info("Download local mein karo")
# Leaderboard (Step 10)
st.subheader("🏆 Model Leaderboard")
leaderboard = pd.DataFrame({
    "Model": ["XGBoost Production", "Baseline"],
    "RMSE": [np.sqrt(mean_squared_error(y, model.predict(X))), y.std()],
    "R²": [model.score(X, y), 0.0],
    "Status": ["✅ LIVE", "📉 BEATEN"]
})
st.dataframe(leaderboard.style.highlight_max(subset=["R²"], color="green"), use_container_width=True)

# Fixed Live Prediction
st.subheader("🎯 Live ML Prediction")
col1, col2 = st.columns(2)
feat1_val = col1.slider(f"📊 {X_cols[0]}", float(df[X_cols[0]].min()), float(df[X_cols[0]].max()), float(df[X_cols[0]].mean()))
feat2_val = col2.slider(f"💰 {X_cols[1]}", float(df[X_cols[1]].min()), float(df[X_cols[1]].max()), float(df[X_cols[1]].mean()))

if st.button("🔮 **PREDICT PRICE**", use_container_width=True):
    test_X = pd.DataFrame({X_cols[0]: [feat1_val], X_cols[1]: [feat2_val]})
    prediction = model.predict(test_X)[0]
    st.metric("Predicted Optimal Price", f"**₹{prediction:.0f}**")
    st.balloons()

    # SHAP Explanations  
    st.subheader("🔍 Why This Prediction?")
try:
    import shap
    explainer = shap.Explainer(model)
    shap_values = explainer(test_X)
    
    st.shap(shap.plots.waterfall(shap_values[0]), height=400)
    st.success("✅ SHAP: Model decisions explained!")
except:
    st.info("Install: pip install shap")

if csv_file:
    df = pd.read_csv(csv_file)
    df["predicted"] = model.predict(df[X_cols].fillna(0))
    st.dataframe(df[["predicted"]].head())





