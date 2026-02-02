import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

# Ignore minor warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1️⃣ Data Loading and Feature Engineering ---
def load_and_preprocess_data(file_path="dose.xlsx"):
    """Loads the dataset and performs necessary feature engineering."""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'. Please ensure 'dose.xlsx' is in the same directory.")
        return None, None, None

    # Drop 'Patient ID' as it's an identifier, not a feature
    df = df.drop(columns=['Patient ID'])

    # Feature Engineering: Extract SBP and DBP from the 'Daily BP Reading (Avg)' string
    # E.g., '95/60 mmHg (Low)' -> SBP=95.0, DBP=60.0
    bp_parts = df['Daily BP Reading (Avg)'].str.extract(r'(\d+)/(\d+)').astype(float)
    df['SBP'] = bp_parts[0]
    df['DBP'] = bp_parts[1]
    df = df.drop(columns=['Daily BP Reading (Avg)'])

    # Separate Features (X) and Target (y)
    target_column = "Likely Dose Problem"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y, df.columns.tolist()

# --- 2️⃣ Model Training Pipeline ---
def train_dose_classifier(X, y):
    """Defines and trains the classification pipeline."""

    # Encode the Categorical Target Variable (y)
    # The classification model requires a numerical target.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Identify Feature Columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['float64']).columns.tolist()

    # Preprocessing Steps
    # Standardize numerical BP readings (SBP, DBP)
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    # One-Hot Encode categorical symptom columns (Yes/No)
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    # Define the Model (Random Forest Classifier for a categorical outcome)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # Build the full Pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                               ("classifier", model)])

    # Split Data
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    # Train Model
    print("🚀 Starting Model Training...")
    pipeline.fit(X_train, y_train_enc)
    print("✅ Model Trained Successfully!")

    # Predict and Evaluate
    y_pred_enc = pipeline.predict(X_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred_enc)
    y_test_decoded = label_encoder.inverse_transform(y_test_enc)

    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)

    print("\n" + "="*50)
    print(f"🔹 Accuracy Score: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    print("="*50)

    # Save Model and Encoder
    joblib.dump(pipeline, "drug_model.pkl")
    joblib.dump(label_encoder, "dose_label_encoder.pkl")
    print("💾 Model saved as drug_model.pkl")
    print("💾 Label Encoder saved as dose_label_encoder.pkl")


# --- Main Execution Block ---
if __name__ == "__main__":
    X, y, columns = load_and_preprocess_data()
    
    if X is not None:
        # Note: If your dataset is very small (like the 10 rows shown in the image), 
        # the model training will be for demonstration only and may show 1.00 accuracy.
        # This is expected for minimal, perfectly defined data.
        train_dose_classifier(X, y)