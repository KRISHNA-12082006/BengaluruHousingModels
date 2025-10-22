from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)

# Load metadata
with open('static/results/metadata.json', 'r') as f:
    metadata = json.load(f)

# Load model results
with open('static/results/model_results.pkl', 'rb') as f:
    model_results = joblib.load(f)

# Load the best model
best_model_name = metadata['best_model']
with open(f'saved_models/{best_model_name}.pkl', 'rb') as f:
    best_model_data = joblib.load(f)
    best_model = best_model_data['model']
    preprocessor = best_model_data['preprocessor']

# Load all models for comparison
all_models = {}
for model_name in metadata['model_list']:
    with open(f'saved_models/{model_name}.pkl', 'rb') as f:
        all_models[model_name] = joblib.load(f)


@app.route('/')
def home():
    """Home page with overview"""
    return render_template('home.html', 
                         metadata=metadata,
                         model_results=model_results)


@app.route('/eda')
def eda():
    """EDA page showing all exploratory analysis plots"""
    plots = {
        'price_distribution': 'static/plots/price_distribution.png',
        'categorical_analysis': 'static/plots/categorical_analysis.png',
        'correlation_heatmap': 'static/plots/correlation_heatmap.png',
        'price_relationships': 'static/plots/price_relationships.png'
    }
    return render_template('eda.html', plots=plots)


@app.route('/vif')
def vif():
    """VIF analysis page"""
    return render_template('vif.html')


@app.route('/models')
def models():
    """Model performance comparison page"""
    return render_template('models.html', 
                         model_results=model_results,
                         best_model=best_model_name)


@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html', 
                         metadata=metadata,
                         best_model=best_model_name)


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        # Extract features
        location = data.get('location')
        area_type = data.get('area_type')
        size = data.get('size')
        total_sqft = float(data.get('total_sqft'))
        bath = int(data.get('bath'))
        balcony = int(data.get('balcony'))
        society = data.get('society', 'Unknown')
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'location': [location],
            'area_type': [area_type],
            'size': [size],
            'total_sqft': [str(total_sqft)],
            'bath': [bath],
            'balcony': [balcony],
            'society': [society]
        })
        
        # Feature engineering (same as training)
        input_data['size_num'] = input_data['size'].str.extract(r'(\d+)').astype(float)
        input_data['total_sqft_num'] = total_sqft
        input_data['bath_per_size'] = input_data['bath'] / input_data['size_num'].replace(0, 1)
        
        # Preprocess
        X_processed = preprocessor.transform(input_data)
        
        # Get model to use
        model_name = data.get('model', best_model_name)
        model_data = all_models.get(model_name, all_models[best_model_name])
        model = model_data['model']
        
        # Predict
        log_prediction = model.predict(X_processed)
        prediction = np.expm1(log_prediction)[0]
        
        # Get predictions from all models for comparison
        all_predictions = {}
        for name, model_dict in all_models.items():
            log_pred = model_dict['model'].predict(X_processed)
            all_predictions[name] = round(float(np.expm1(log_pred)[0]), 2)
        
        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 2),
            'model_used': model_name,
            'all_predictions': all_predictions,
            'input_summary': {
                'location': location,
                'area_type': area_type,
                'size': size,
                'total_sqft': total_sqft,
                'bathrooms': bath,
                'balconies': balcony
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/summary')
def summary():
    """Analysis summary page"""
    with open('static/results/analysis_summary.txt', 'r') as f:
        summary_text = f.read()
    return render_template('summary.html', summary_text=summary_text)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
