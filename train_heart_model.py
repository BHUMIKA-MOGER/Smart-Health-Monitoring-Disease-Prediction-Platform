import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys

# --- Configuration ---
# NOTE: This script assumes 'heart.csv' is in the same directory.
DATA_FILE_NAME = 'heart.csv'

# New feature names provided by the user (matching the column headers in heart.csv)
CATEGORICAL_FEATURES = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
NUMERICAL_FEATURES = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
TARGET_COLUMN = 'HeartDisease' # The column to predict (0 or 1)

# --- 1. Load and Preprocess Data ---
print(f"Loading and preprocessing data from '{DATA_FILE_NAME}'...")
try:
    # Load the full dataset
    data = pd.read_csv(DATA_FILE_NAME)
except FileNotFoundError:
    print(f"❌ ERROR: '{DATA_FILE_NAME}' not found. Ensure the file exists.")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR reading data: {e}")
    sys.exit(1)

# Ensure the target is binary (0 or 1)
data[TARGET_COLUMN] = np.where(data[TARGET_COLUMN] > 0, 1, 0)

# --- 2. Feature Engineering: One-Hot Encoding and Scaling ---

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

# One-Hot Encoding for Categorical Features (Must be done BEFORE scaling)
X_encoded = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)

# Scaling Numerical Features
scaler = StandardScaler()
# Fit and transform only the numerical columns
X_encoded[NUMERICAL_FEATURES] = scaler.fit_transform(X_encoded[NUMERICAL_FEATURES])

# --- 3. Train Model ---
print("Training Logistic Regression Model...")
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_encoded, y)

# --- 4. Extract Parameters for JavaScript Deployment ---
# NOTE: This step is simplified here as we are now only focused on binary prediction (0 or 1).

# Print Model Coefficients and Intercept for interpretation
print("\n=========================================================================")
print("✅ Training Complete. Model is ready for Binary Prediction (0 or 1).")
print("=========================================================================")

print(f"\nINTERCEPT: {model.intercept_[0]:.4f}")
print("\nTOP 10 FEATURE COEFFICIENTS (Model Weights):")

# Create a readable Series of coefficients
coefs = pd.Series(model.coef_[0], index=X_encoded.columns).sort_values(ascending=False)
print(coefs.head(5))
print("...")
print(coefs.tail(5))
