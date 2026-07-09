import pandas as pd
from xgboost import XGBRegressor
import pickle

cars_df = pd.read_csv("cars24-car-price-cleaned-new.csv")

X = cars_df[['km_driven','mileage','age','Petrol','Diesel','Electric']]
y = cars_df['selling_price']   # better as 1D array

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=6
)

# Fit model
xgb_model.fit(X, y)

# Save model
# wb - write binary
with open("xgb_car_price_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)