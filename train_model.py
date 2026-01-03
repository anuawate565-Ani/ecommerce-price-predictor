import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Sample data (replace with your data)
data = pd.read_csv('ecommerce_data.csv')  # Your dataset
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = XGBRegressor()
model.fit(X_train, y_train)
joblib.dump(model, 'price_model.pkl')
print("Model saved!")
