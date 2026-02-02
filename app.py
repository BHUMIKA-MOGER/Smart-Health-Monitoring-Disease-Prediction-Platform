import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import joblib
import re 

app = Flask(__name__)
# IMPORTANT: Use a complex secret key in a real application
app.secret_key = 'a0b1c2d3e4f56789g9h0i1j2k3l4m5n6' 

# Initialize models and encoder to None
triage_model = None
triage_encoder = None
drug_model = None
dose_encoder = None

# --- Feature Configuration (MUST match training script) ---
# Features for Diabetes Scoring Model (9 features)
DIABETES_POSITIVE_COLS = [
    'Feeling very thirsty often', 
    'Urinating frequently', 
    'Always feeling hungry or tired', 
    'Having dry mouth or dry skin', 
    'Feels sleepy or slow most of the day'
]
DIABETES_NEGATIVE_COLS = [
    'Feeling sudden shakiness or dizziness', 
    'Sweating a lot suddenly', 
    'Feels cold or clammy sometimes', 
    'Fast heartbeat or irritability'
]
DIABETES_FEATURE_COLUMNS = DIABETES_POSITIVE_COLS + DIABETES_NEGATIVE_COLS
DIABETES_SCORING_AVAILABLE = True 

# Features for the original 7-symptom triage model
TRIAGE_FEATURE_COLUMNS = [
    'Crushing', 'Pressure', 'Squeezing', 'Duration', 
    'Relief', 'Spreading of Pain (One Side)', 'Vomiting'
]

# Features for Drug Dose Classification (MUST match the drug_model.py script)
# NOTE: 'Age' is not included here because the training script dropped it.
DRUG_DOSE_RAW_INPUTS = [
    'Age', # Included for form processing but dropped before modeling
    'Daily BP Reading', 
    'Feeling Sleepy More? (CNS Sign)', 
    'New Severe Headaches?', 
    'Feeling Like Fainting? (Hypotension)', 
    'New Rapid Weight Gain? (Fluid)'
]


# --- Model Loading Section ---
def load_asset(file_path, asset_name, is_joblib=False):
    """Utility function to load a model or encoder using joblib or pickle."""
    try:
        if is_joblib:
            asset = joblib.load(file_path)
        else:
            # Fallback to pickle for old models if necessary, though joblib is preferred for sklearn pipelines
            import pickle
            with open(file_path, 'rb') as file:
                asset = pickle.load(file)
        
        print(f"✅ Successfully loaded asset: {asset_name} from {file_path}")
        return asset
    except FileNotFoundError:
        print(f"❌ ERROR: File {file_path} for {asset_name} not found. Did you run the training script?")
        return None
    except Exception as e:
        print(f"❌ ERROR loading {asset_name}: {e}")
        return None
        # Check if HEART_RISK_PIPELINE was loaded (and ignore the assignment below)
        if asset_name == 'Binary Heart Risk Pipeline':
            print(f"✅ HEART_RISK_PIPELINE loaded successfully (but is disabled per user request).")
        else:
            print(f"✅ Successfully loaded {asset_name} from {file_path}")
        return asset
    except FileNotFoundError:
        print(f"❌ WARNING: '{file_path}' not found for {asset_name}.")
        return None
    except Exception as e:
        print(f"❌ ERROR loading {asset_name}: {e}")
        return None

# Load Emergency Triage Model (Original)
triage_model = load_asset('heart_triage_rf_model.pkl', 'Heart Symptom Triage Model', is_joblib=True)
triage_encoder = load_asset('triage_label_encoder.pkl', 'Triage Label Encoder', is_joblib=True)

# Load Drug Dose Model and Encoder
drug_model = load_asset('drug_model.pkl', 'Drug Dose Classifier', is_joblib=True)
dose_encoder = load_asset('dose_label_encoder.pkl', 'Dose Label Encoder', is_joblib=True)
# --- End Model Loading Section ---


# --- Prediction Helper Functions (Definitions) ---

def calculate_diabetes_condition(form_data):
    """
    Calculates the Total Score and Condition for Diabetes based on user-input form data
    using the user-defined scoring logic.
    """
    total_score = 0
    for col in DIABETES_POSITIVE_COLS:
        if form_data.get(col) == 'Yes':
            positive_score += 1
            
    negative_score = 0
    for col in DIABETES_FEATURE_COLUMNS:
        if form_data.get(col) == 'Yes':
            total_score += 1


    if total_score >= 6:
        condition = 'High Risk'
        color = 'text-red-700'
    elif total_score >= 3 and total_score <= 5:
        condition = 'Moderate Risk'
        color = 'text-orange-600'
    else:
        condition = 'Low Risk'
        color = 'text-green-600'
        
    result_html = f"<div class='{color} font-bold text-2xl'>Condition: {condition}</div>"
    result_html += f"<div class='text-lg'>Total Score: {total_score} / 9</div>"

    disclaimer = (
        "This score is based on self-reported symptoms only. "
        "It is NOT a medical diagnosis. Consult a healthcare professional for accurate testing."
    )
    
    return result_html, disclaimer, None


def validate_and_predict_triage(request_form, model, encoder):
    """Handles prediction for the 7 Triage symptoms."""

    
    binary_map = {'Yes': 1, 'No': 0}
    relief_map = {'Yes': 0, 'No': 1} 
    duration_mapping = {'5-10min': 1, '20-25min': 2, 'More': 3}
    
    features = []
    
    for feature_name in TRIAGE_FEATURE_COLUMNS:
        value = request_form.get(feature_name)
        
        if value is None or value == 'Select...':
            return None, None, f"<span class='text-red-800'>ERROR: Please select a value for all symptom fields.</span>"
        
        if feature_name == 'Duration':
            encoded_val = duration_mapping.get(value)
        elif feature_name == 'Relief':
            encoded_val = relief_map.get(value)
        else: 
            encoded_val = binary_map.get(value)
            
        if encoded_val is None:
            return None, None, f"<span class='text-red-800'>ERROR: Invalid input received for {feature_name}.</span>"

        features.append(encoded_val)

    final_features = np.array(features).reshape(1, -1)
    prediction_encoded = model.predict(final_features)[0]

    prediction_label = encoder.inverse_transform([prediction_encoded])[0]
    
    if prediction_label == 'HIGH (ER)':
        result = (
            "<span class='text-red-700 font-bold text-3xl'>URGENT EMERGENCY! Call the ambulance IMMEDIATELY.</span>"
        )
        disclaimer = (
            " High risk of a severe cardiac event. "
            "Call for emergency medical help now."
        )
    else: # LOW (Normal)
        result = (
            "<span class='text-green-700 font-bold text-3xl'>MINOR PROBLEM detected.</span>"
        )
        disclaimer = (
            " Low probability of an acute cardiac emergency. "
            "However, Please consult your doctor "
        )

    return result, disclaimer, None 
    

def validate_and_predict_dose(request_form, model, encoder):
    """Handles prediction for drug dose analysis."""
    
    data = {}
    for col in DRUG_DOSE_RAW_INPUTS:
        value = request_form.get(col)
        if col != 'Age' and (value is None or value == ''):
            return None, None, f"<span class='text-red-800'>ERROR: Please provide a value for all required fields. Missing: {col}</span>"
        data[col] = value
        
    try:
        bp_reading = data.get('Daily BP Reading', '')
        bp_match = re.match(r'(\d+)\/(\d+)', bp_reading)
        if not bp_match:
            return None, None, "<span class='text-red-800'>ERROR: BP Reading must be in 'SBP/DBP' format (e.g., 120/80).</span>"
        
        systolic_bp = float(bp_match.group(1))
        diastolic_bp = float(bp_match.group(2))
        
        categorical_features = ['Feeling Sleepy More? (CNS Sign)', 'New Severe Headaches?', 'Feeling Like Fainting? (Hypotension)', 'New Rapid Weight Gain? (Fluid)']
        numerical_features = ['SBP', 'DBP']
        
        input_data = {}
        for col in categorical_features:
            input_data[col] = data[col]
        
        input_data['SBP'] = systolic_bp
        input_data['DBP'] = diastolic_bp

        final_input_df = pd.DataFrame([input_data], columns=categorical_features + numerical_features)
        final_input_df['SBP'] = final_input_df['SBP'].astype(float)
        final_input_df['DBP'] = final_input_df['DBP'].astype(float)

        prediction_encoded = model.predict(final_input_df)[0]
        
    except Exception as e:
        print(f"Model Prediction Failure: {e}")
        print(f"Input DataFrame (Attempted): \n{final_input_df if 'final_input_df' in locals() else 'Not created'}")
        return None, None, f"<span class='text-red-800 font-bold'>Model Prediction Error. Check feature ordering/compatibility. Details: {e}</span>"
    
    prediction_label = encoder.inverse_transform([int(prediction_encoded)])[0]

    if 'HIGH' in prediction_label:
        color = 'text-red-700'
        action = "Immediate dose reduction is recommended. Check for toxicity signs and kidney/liver function."
    elif 'LOW' in prediction_label:
        color = 'text-orange-600'
        action = "Dose titration (increase) is likely needed. Monitor BP closely."
    else: # OPTIMAL
        color = 'text-green-600'
        action = "Dose is currently appropriate. Continue monitoring."

    result = (
        f"<div class='{color} font-bold text-3xl'>Likely Dose Problem: {prediction_label} </div>"
        f"<div class='text-lg mt-2'>Action Recommended: {action}</div>"
    )
    
    disclaimer = (
        "This analysis is based on symptom-to-dose correlation. Always verify with lab tests and a physician."
    )
    
    return result, disclaimer, None

# --- Placeholder for the removed Heart Risk API Route ---
# The function and route were removed as requested.

# --- Flask Routes (Login, Dashboard, Prediction Pages) ---

# --- 1. Login Route ---
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'user' and request.form['password'] == 'pass':
            session['logged_in'] = True
            return redirect(url_for('dashboard')) 
        
        return render_template('login.html', error="Invalid Username or Password.")
    
    return render_template('login.html', error=None)

# --- 2. Dashboard Route ---
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template(
        'dashboard.html', 
        username='Portal User',
        diabetes_available=DIABETES_SCORING_AVAILABLE, 
        heart_available=(triage_model is not None and triage_encoder is not None),
        drug_dose_available=(drug_model is not None and dose_encoder is not None)
    )

# --- 3. Diabetes Prediction Route (SCORING) ---
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes_prediction():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    result_html = None
    disclaimer_html = None
    error_html = None
    
    if request.method == 'POST':
        try:
            result_html, disclaimer_html, error_html = calculate_diabetes_condition(request.form)
        except Exception as e:
            error_html = f"<span class='text-red-800'>An internal calculation error occurred: {e}</span>"
            
    final_result_html = error_html if error_html else result_html

    return render_template(
        'diabetes.html',
        diabetes_result_html=final_result_html,
        disclaimer_html=disclaimer_html,
        features=DIABETES_FEATURE_COLUMNS,
        positive_cols=DIABETES_POSITIVE_COLS,
        negative_cols=DIABETES_NEGATIVE_COLS
    )

# --- 4. Heart Disease Prediction Route (Triage) ---
@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease_prediction():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    triage_result_html = None
    disclaimer_html = None
    error_html = None

    if request.method == 'POST':
        if triage_model and triage_encoder:
            try:
                triage_result_html, disclaimer_html, error_html = validate_and_predict_triage(
                    request.form, triage_model, triage_encoder
                )
            except Exception as e:
                error_html = f"<span class='text-red-800'>An internal calculation error occurred during prediction. Details: {e}</span>"
        else:
            error_html = "<span class='text-red-800 font-bold'>❌ ERROR: Heart Triage model files not loaded. Please run triage_model_training.py.</span>"
    
    final_result_html = error_html if error_html else triage_result_html
    
    return render_template(
        'heart_disease.html', 
        triage_result_html=final_result_html, 
        disclaimer_html=disclaimer_html,
        features=TRIAGE_FEATURE_COLUMNS 
    )


# --- 5. Drug Dose Optimization Route (PREDICTION) ---
@app.route('/drug_optimization', methods=['GET', 'POST'])
def drug_optimization():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    dose_result_html = None
    disclaimer_html = None
    error_html = None
    
    if request.method == 'POST':
        if drug_model and dose_encoder:
            try:
                dose_result_html, disclaimer_html, error_html = validate_and_predict_dose(
                    request.form, drug_model, dose_encoder
                )
            except Exception as e:
                error_html = f"<span class='text-red-800'>An internal calculation error occurred during dose prediction: {e}</span>"
        else:
            error_html = "<span class='text-red-800 font-bold'>❌ ERROR: Drug Dose model files not loaded. Please run drug_model.py.</span>"
            
    final_result_html = error_html if error_html else dose_result_html
    
    return render_template(
        'drug_dose.html',
        dose_result_html=final_result_html,
        disclaimer_html=disclaimer_html,
        features=DRUG_DOSE_RAW_INPUTS
    )


if __name__ == '__main__':
    # Since HEART_RISK_PIPELINE is removed, we run unconditionally (assuming other models are optional)
    app.run(debug=True)
