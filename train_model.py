# train_model.py

# make sure all this packages are installed 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from feature_engine.outliers import Winsorizer
import joblib

# Load & clean
url = "https://raw.githubusercontent.com/Alyxx-The-Sniper/Grament_production_analyis/refs/heads/main/garments_worker_productivity.csv"
df = pd.read_csv(url)
df['department'] = df['department'].str.strip().replace({'sweing':'sewing', 'finishing ':'finishing'})
df_sewing = df[df['department'] == 'sewing'].reset_index(drop=True)

X = df_sewing.drop(columns=['actual_productivity'])
y = df_sewing['actual_productivity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ['incentive']
numeric_pipeline = Pipeline([
    ('winsor', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols)
])

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)

pipeline = Pipeline([
    ('PREPROC', preprocessor),
    ('REGRESSOR', model)
])

pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'model_1.pkl')
print("Model trained and saved as model_1.pkl")

