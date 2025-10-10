# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv
import io
import uuid

# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Continuing with system environment variables...")

# --- Gemini API Configuration ---
# IMPORTANT: Set this in your Render environment variables
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("Warning: GEMINI_API_KEY not found. Chat functionality will be limited.")
    gemini_model = None

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- Load Your Machine Learning Model ---
model = joblib.load('stacking_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
model_columns = joblib.load('model_columns.pkl')

# --- Feature metadata cache (computed on first request) ---
_FEATURE_METADATA_CACHE = None

# Human-readable descriptions for common KOI features
_FEATURE_DESCRIPTIONS = {
    'koi_score': 'Disposition score (0-1) indicating confidence in KOI vetting.',
    'koi_fpflag_nt': 'Not Transit-Like flag (1 indicates non-transit signal).',
    'koi_fpflag_ss': 'Stellar variability or systematics suspected (1 indicates stellar-statistic issue).',
    'koi_fpflag_co': 'Centroid offset indicates background source (1 indicates offset).',
    'koi_fpflag_ec': 'Ephemeris match with known variable/eclipsing source (1 indicates match).',
    'koi_period': 'Orbital period of the planet candidate in days.',
    'koi_duration': 'Transit duration in hours.',
    'koi_depth': 'Transit depth in parts-per-million (ppm).',
    'koi_prad': 'Estimated planet radius in Earth radii.',
    'koi_period_err1': 'Positive uncertainty on the orbital period.',
    'koi_period_err2': 'Negative uncertainty on the orbital period.',
    'koi_time0bk': 'Time of first transit in BKJD.',
    'koi_impact': 'Transit impact parameter (0 central, ~1 grazing).',
    'koi_teq': 'Planet equilibrium temperature in Kelvin.',
    'koi_insol': 'Insolation (stellar flux) in Earth units.',
    'koi_model_snr': 'Transit model signal-to-noise ratio.',
    'koi_tce_plnt_num': 'TCE planet number within the system.',
    'koi_steff': 'Stellar effective temperature in Kelvin.',
    'koi_slogg': 'Stellar surface gravity log10(cm/s^2).',
    'koi_srad': 'Stellar radius in Solar radii.',
    'ra': 'Right Ascension of the target (degrees).',
    'dec': 'Declination of the target (degrees).',
    'koi_kepmag': 'Kepler apparent magnitude (brightness).'
}

def _compute_feature_metadata():
    global _FEATURE_METADATA_CACHE
    if _FEATURE_METADATA_CACHE is not None:
        return _FEATURE_METADATA_CACHE

    csv_path = 'KOI_Dataset_Exoplanets.csv'
    metadata = {
        'ranges': {},
        'descriptions': _FEATURE_DESCRIPTIONS,
        'fpflag_options': [0, 1],
        'required_columns': model_columns,
    }
    try:
        # Read only columns used by model to reduce memory
        use_cols = [c for c in model_columns if c != 'id']
        df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in use_cols,
            low_memory=False,
            comment='#',        # ignore NASA header comment lines
            na_values=['', 'NA', 'NaN', 'nan']
        )
        # Coerce to numeric for safety
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        desc = df.describe(percentiles=[0.01, 0.99]).to_dict()
        for col in df.columns:
            col_stats = desc.get(col, {})
            vmin = col_stats.get('min')
            vmax = col_stats.get('max')
            # Reasonable fallbacks
            if pd.isna(vmin) or pd.isna(vmax):
                vmin, vmax = None, None
            # Clamp koi_score to [0,1]
            if col == 'koi_score':
                vmin, vmax = 0.0, 1.0
            # Provide default ranges if degenerate
            if vmin is not None and vmax is not None and vmin == vmax:
                # widen slightly
                pad = 1.0 if vmax == 0 else abs(vmax) * 0.1
                vmin, vmax = vmin - pad, vmax + pad
            metadata['ranges'][col] = {
                'min': None if vmin is None else float(vmin),
                'max': None if vmax is None else float(vmax)
            }
    except Exception as e:
        # If dataset not available, provide sensible defaults
        defaults = {
            'koi_score': (0.0, 1.0),
            'koi_period': (0.1, 1000.0),
            'koi_duration': (0.1, 50.0),
            'koi_depth': (10.0, 100000.0),
            'koi_prad': (0.1, 30.0),
            'koi_teq': (50.0, 4000.0),
            'koi_insol': (0.001, 10000.0),
            'koi_model_snr': (0.0, 1000.0),
            'koi_steff': (2500.0, 10000.0),
            'koi_slogg': (2.0, 5.5),
            'koi_srad': (0.1, 100.0),
            'koi_kepmag': (8.0, 20.0)
        }
        for col, (vmin, vmax) in defaults.items():
            metadata['ranges'][col] = {'min': vmin, 'max': vmax}
    _FEATURE_METADATA_CACHE = metadata
    return metadata

# --- API Endpoints ---
# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    # This endpoint remains the same as before
    data = request.get_json()
    df_new = pd.DataFrame([data], columns=model_columns)
    df_new_imputed = imputer.transform(df_new)
    df_new_scaled = scaler.transform(df_new_imputed)
    prediction_encoded = model.predict(df_new_scaled)
    prediction_proba = model.predict_proba(df_new_scaled)
    mapping = {0: "FALSE POSITIVE", 1: "CONFIRMED", 2: "CANDIDATE"}
    prediction_label = mapping[prediction_encoded[0]]
    confidence = np.max(prediction_proba) * 100
    return jsonify({
        'prediction': prediction_label,
        'confidence': f"{confidence:.2f}%"
    })

@app.route('/feature-metadata', methods=['GET'])
def feature_metadata():
    try:
        metadata = _compute_feature_metadata()
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    # This is the new endpoint for the chatbot
    data = request.get_json()
    user_message = data.get('message')
    context = data.get('context')

    if not user_message or not context:
        return jsonify({'error': 'Message and context are required.'}), 400

    # --- Prompt Engineering: The Secret to a Great Chatbot ---
    prompt = f"""
    You are Exo-Chat, a friendly NASA astronomer. Keep responses SHORT and SIMPLE.

    HERE IS THE DATA CONTEXT: Cumulative Table
Last update: Sept. 27, 2018
Current status: Done
Delivery History
Interactive Table
The cumulative KOI table gathers information from the individual KOI activity tables that describe the current results of different searches of the Kepler light curves. The intent of the cumulative table is to provide the most accurate dispositions and stellar and planetary information for all KOIs in one place. All the information in this table has provenance in other KOI activity tables.

The cumulative table is created algorithmically, following simple rules. The information for each KOI is pulled from the preferred activity table based on two priority lists. One priority list (Disposition Priority) indicates the activity table from which the disposition (e.g., CANDIDATE or FALSE POSITIVE) has been pulled. If the object is not dispositioned in the highest priority activity table for a specific KOI, then it is pulled from the next highest priority activity table, and so on. In this way the cumulative table contains the most current disposition for each KOI. The second priority list (Transit-Fit Priority) indicates where the remaining information for each KOI (e.g., the transit fits, stellar properties and vetting statistics) was obtained. The activity table with reliable transit fits to the longest data set is given priority for the cumulative table. This will not necessarily provide the best fit for every individual KOI, but gives the most reliable fits overall. The current Disposition Priority order is: Q1-Q17 DR 25 Supplemental, Q1-Q17 DR 25, Q1-Q17 DR 24, Q1-Q16, Q1-Q12, Q1-Q8, Q1-Q6. The current Transit-Fit Priority order is: Q1-Q17 DR 25, Q1-17 DR 24, Q1-Q16, Q1-Q12, Q1-Q8, Q1-Q6, and Q1-Q17 DR 25 Supplemental.

One consequence of having two priority lists is that the disposition for a KOI is not necessarily retrieved from the same activity table as the associated transit information. Also, since information for the cumulative table is gathered from a variety of activity tables, and since these activities use different methods for dispositioning, defining stellar parameters, and fitting transits, the cumulative table is a very disparate set of information and is not intended for statistical studies that require a uniform population.
    
    DATA: {context['input_data']}
    RESULT: {context['prediction']} ({context['confidence']} confidence)
    
    Answer the user's question briefly and clearly. Use simple words. Be enthusiastic but concise.
    
    USER: "{user_message}"
    """

    try:
        if gemini_model is None:
            return jsonify({
                'reply': "I'm sorry, but I'm currently offline. The AI chat feature requires a Gemini API key to be configured. However, I can still help you understand your classification results! Based on your data, this appears to be a " + context['prediction'] + " with " + context['confidence'] + " confidence. This means the machine learning model is quite confident in its assessment of this exoplanet candidate."
            })
        response = gemini_model.generate_content(prompt)
        return jsonify({'reply': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the CSV file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'File must be CSV or Excel format'}), 400
        
        # Check if all required columns are present
        missing_columns = set(model_columns) - set(df.columns)
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {list(missing_columns)}',
                'required_columns': model_columns
            }), 400
        
        # Select only the required columns and reorder them
        df_model = df[model_columns].copy()
        
        # Add ID column if not present
        if 'id' not in df.columns:
            df_model['id'] = [f'KOI_{i+1:03d}' for i in range(len(df_model))]
        else:
            df_model['id'] = df['id']
        
        # Process each row
        results = []
        for idx, row in df_model.iterrows():
            try:
                # Prepare data for prediction (exclude id column)
                prediction_data = row.drop('id').values.reshape(1, -1)
                
                # Impute missing values
                prediction_data_imputed = imputer.transform(prediction_data)
                
                # Scale the data
                prediction_data_scaled = scaler.transform(prediction_data_imputed)
                
                # Make prediction
                prediction_encoded = model.predict(prediction_data_scaled)
                prediction_proba = model.predict_proba(prediction_data_scaled)
                
                # Map prediction
                mapping = {0: "FALSE POSITIVE", 1: "CONFIRMED", 2: "CANDIDATE"}
                prediction_label = mapping[prediction_encoded[0]]
                confidence = np.max(prediction_proba) * 100
                
                results.append({
                    'id': row['id'],
                    'prediction': prediction_label,
                    'confidence': f"{confidence:.2f}%",
                    'status': 'success'
                })
                
            except Exception as e:
                results.append({
                    'id': row.get('id', f'KOI_{idx+1:03d}'),
                    'prediction': 'ERROR',
                    'confidence': '0.00%',
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'successful': len([r for r in results if r['status'] == 'success'])
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)