"""
app.py
======
Flask Web Application for the Basic Crop Recommendation System.
Serves the responsive user interface and handles machine learning predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Feature configuration - order must strictly match model training
FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Agronomic boundary definitions for validation
FEATURE_RANGES = {
    'N': {'min': 0, 'max': 140, 'label': 'Nitrogen (N)', 'unit': 'kg/ha'},
    'P': {'min': 5, 'max': 145, 'label': 'Phosphorus (P)', 'unit': 'kg/ha'},
    'K': {'min': 5, 'max': 205, 'label': 'Potassium (K)', 'unit': 'kg/ha'},
    'temperature': {'min': 5.0, 'max': 50.0, 'label': 'Temperature', 'unit': '°C'},
    'humidity': {'min': 10.0, 'max': 100.0, 'label': 'Humidity', 'unit': '%'},
    'ph': {'min': 3.5, 'max': 10.0, 'label': 'Soil pH', 'unit': ''},
    'rainfall': {'min': 15.0, 'max': 350.0, 'label': 'Rainfall', 'unit': 'mm'}
}

# Crop knowledge base providing context for recommendations
CROP_METADATA = {
    'rice': {
        'icon': '🌾',
        'category': 'Cereal & Food Grain',
        'description': 'Rice thrives in warm, humid conditions with abundant standing water or assured heavy irrigation.',
        'season': 'Kharif (Monsoon)',
        'water': 'High (Puddled / Flooded fields)',
        'soil': 'Clayey Loam & Deep Alluvial',
        'ph_range': '6.0 - 7.5'
    },
    'maize': {
        'icon': '🌽',
        'category': 'Coarse Cereal & Feed Crop',
        'description': 'Maize adapts well across temperate and tropical zones with well-drained, fertile soil.',
        'season': 'Kharif & Rabi',
        'water': 'Moderate (500-800 mm)',
        'soil': 'Fertile Deep Loam',
        'ph_range': '5.8 - 7.0'
    },
    'chickpea': {
        'icon': '🧆',
        'category': 'Pulse / Legume',
        'description': 'Chickpea is a cool-season legume that improves soil fertility through biological nitrogen fixation.',
        'season': 'Rabi (Winter)',
        'water': 'Low to Moderate',
        'soil': 'Sandy Loam to Heavy Clay',
        'ph_range': '6.0 - 8.5'
    },
    'kidneybeans': {
        'icon': '🫘',
        'category': 'Pulse / Legume',
        'description': 'Kidney beans (Rajma) prefer mild temperate weather and well-aerated, moisture-retentive soils.',
        'season': 'Rabi / Spring',
        'water': 'Moderate',
        'soil': 'Rich Loamy Soil',
        'ph_range': '5.5 - 6.5'
    },
    'pigeonpeas': {
        'icon': '🌱',
        'category': 'Pulse / Legume',
        'description': 'Pigeonpea (Arhar/Tur) is a drought-tolerant legume with a deep taproot system suitable for semi-arid tropics.',
        'season': 'Kharif',
        'water': 'Low to Moderate',
        'soil': 'Well-drained Deep Loam',
        'ph_range': '5.0 - 7.5'
    },
    'mothbeans': {
        'icon': '🌿',
        'category': 'Arid Legume',
        'description': 'Mothbean is exceptionally drought-hardy and thrives in arid conditions with minimal precipitation.',
        'season': 'Kharif',
        'water': 'Very Low',
        'soil': 'Sandy to Sandy Loam',
        'ph_range': '6.0 - 8.0'
    },
    'mungbean': {
        'icon': '🟢',
        'category': 'Short-Duration Pulse',
        'description': 'Mungbean (Green Gram) matures quickly in warm climates and serves effectively in crop rotation cycles.',
        'season': 'Zaid / Summer & Kharif',
        'water': 'Low to Moderate',
        'soil': 'Fertile Loam',
        'ph_range': '6.2 - 7.2'
    },
    'blackgram': {
        'icon': '⚫',
        'category': 'Pulse / Legume',
        'description': 'Blackgram (Urad) requires warm temperatures and thrives in humid sub-tropical conditions.',
        'season': 'Kharif & Summer',
        'water': 'Moderate',
        'soil': 'Heavy Loam with good drainage',
        'ph_range': '6.5 - 7.5'
    },
    'lentil': {
        'icon': '🥣',
        'category': 'Pulse / Legume',
        'description': 'Lentil (Masoor) is an important cool-season winter legume requiring light moisture and moderate temperatures.',
        'season': 'Rabi (Winter)',
        'water': 'Low to Moderate',
        'soil': 'Alluvial and Sandy Clay Loam',
        'ph_range': '6.0 - 7.5'
    },
    'pomegranate': {
        'icon': '🍎',
        'category': 'Horticultural Fruit Crop',
        'description': 'Pomegranate prefers hot, dry summers and mild winters, producing high-value fruit in well-drained soils.',
        'season': 'Perennial (Year-round pruning)',
        'water': 'Low to Moderate (Drip irrigation)',
        'soil': 'Deep Loamy to Light Clay',
        'ph_range': '5.5 - 7.2'
    },
    'banana': {
        'icon': '🍌',
        'category': 'Tropical Fruit Crop',
        'description': 'Banana requires high humidity, consistent warm temperatures, heavy fertilization, and regular moisture.',
        'season': 'Perennial (Planting year-round)',
        'water': 'High (Continuous moisture)',
        'soil': 'Rich Volcanic / Clay Loam',
        'ph_range': '5.5 - 6.5'
    },
    'mango': {
        'icon': '🥭',
        'category': 'Tropical Fruit Orchard',
        'description': 'Mango is the king of fruits, thriving in tropical climates with distinct dry spells for flowering.',
        'season': 'Summer Harvest',
        'water': 'Moderate',
        'soil': 'Deep, well-drained Alluvial Loam',
        'ph_range': '5.5 - 7.5'
    },
    'grapes': {
        'icon': '🍇',
        'category': 'Viticulture / Fruit',
        'description': 'Grapevines flourish in warm, dry weather during berry development with high potassium levels in soil.',
        'season': 'Annual Pruning / Spring-Summer',
        'water': 'Moderate (Drip irrigation preferred)',
        'soil': 'Sandy Loam to Gravelly Loam',
        'ph_range': '6.0 - 7.5'
    },
    'watermelon': {
        'icon': '🍉',
        'category': 'Cucurbit / Summer Fruit',
        'description': 'Watermelon requires warm sunshine, long summer days, and warm sandy soil for high sugar content.',
        'season': 'Zaid / Summer',
        'water': 'Moderate (Reduced at ripening)',
        'soil': 'Sandy Loam with organic matter',
        'ph_range': '6.0 - 7.0'
    },
    'muskmelon': {
        'icon': '🍈',
        'category': 'Cucurbit / Summer Fruit',
        'description': 'Muskmelon thrives in hot, sunny dry climates with light sandy soil and balanced potassium.',
        'season': 'Zaid / Summer',
        'water': 'Moderate',
        'soil': 'Well-aerated Sandy Loam',
        'ph_range': '6.0 - 7.0'
    },
    'apple': {
        'icon': '🍏',
        'category': 'Temperate Fruit Crop',
        'description': 'Apple trees require temperate climates with chilling hours in winter and rich, well-drained loams.',
        'season': 'Late Summer to Autumn',
        'water': 'Moderate to High',
        'soil': 'Deep Loam with good aeration',
        'ph_range': '5.5 - 6.5'
    },
    'orange': {
        'icon': '🍊',
        'category': 'Citrus Fruit Crop',
        'description': 'Citrus trees thrive in subtropical to tropical climates with sunny exposure and well-aerated soil.',
        'season': 'Winter / Spring Harvest',
        'water': 'Moderate',
        'soil': 'Light Loam with good drainage',
        'ph_range': '6.0 - 7.5'
    },
    'papaya': {
        'icon': '🍈',
        'category': 'Fast-Growing Fruit Crop',
        'description': 'Papaya needs continuous warmth, frost-free climate, high organic matter, and zero waterlogging.',
        'season': 'Year-round flowering & fruiting',
        'water': 'Moderate (Strictly no waterlogging)',
        'soil': 'Rich Porous Sandy Loam',
        'ph_range': '6.0 - 7.0'
    },
    'coconut': {
        'icon': '🥥',
        'category': 'Coastal Plantation Crop',
        'description': 'Coconut palms thrive in humid tropical coastal belts with abundant sunlight and regular rainfall.',
        'season': 'Perennial (Continuous fruiting)',
        'water': 'High',
        'soil': 'Coastal Sandy & Alluvial Soils',
        'ph_range': '5.2 - 8.0'
    },
    'cotton': {
        'icon': '🌱',
        'category': 'Commercial Fiber Crop',
        'description': 'Cotton requires warm sunny weather, moderate rainfall, and nutrient-dense soil (e.g. Black Cotton Soil).',
        'season': 'Kharif',
        'water': 'Moderate (500-1000 mm)',
        'soil': 'Black Soil (Regur) or Alluvial',
        'ph_range': '6.0 - 8.0'
    },
    'jute': {
        'icon': '🌾',
        'category': 'Commercial Fiber Crop',
        'description': 'Jute is the golden fiber of the sub-tropics, requiring high humidity, warm monsoon weather, and alluvial silt.',
        'season': 'Pre-Kharif to Kharif',
        'water': 'High (Abundant monsoon rainfall)',
        'soil': 'New Alluvial / Floodplain Silt',
        'ph_range': '6.0 - 7.5'
    },
    'coffee': {
        'icon': '☕',
        'category': 'Plantation Cash Crop',
        'description': 'Coffee flourishes in hilly tropical terrain under canopy shade with moderate temperatures and rich organic mulch.',
        'season': 'Perennial (Winter Harvest)',
        'water': 'High (1500-2000 mm distributed rainfall)',
        'soil': 'Humus-rich Volcanic / Forest Loam',
        'ph_range': '6.0 - 7.0'
    }
}

# Model loading logic with graceful fallback
model_instance = None
model_metadata = {}

def load_trained_model():
    global model_instance, model_metadata

    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'model', 'crop_model.pkl'),
        os.path.join(os.path.dirname(__file__), 'crop_model.pkl')
    ]

    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    artifact = pickle.load(f)
                    if isinstance(artifact, dict) and 'model' in artifact:
                        model_instance = artifact['model']
                        model_metadata = {
                            'features': artifact.get('features', FEATURE_NAMES),
                            'accuracy': artifact.get('test_accuracy', None),
                            'classes': artifact.get('classes', [])
                        }
                    else:
                        model_instance = artifact
                        model_metadata = {'features': FEATURE_NAMES}
                    print(f"[Model Loaded] Successfully loaded model from {p}")
                    return True
            except Exception as e:
                print(f"[Model Error] Failed loading model from {p}: {e}")

    print("[Model Warning] No trained model artifact found on startup. Please run 'train_model.py'.")
    return False

# Attempt initial load
load_trained_model()

@app.route('/')
def home():
    """Renders the main crop recommendation interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives JSON or Form payload with 7 agricultural features,
    validates values, executes Random Forest classification,
    and returns recommendation details.
    """
    global model_instance

    # Ensure model is ready
    if model_instance is None:
        if not load_trained_model():
            return jsonify({
                'success': False,
                'errors': [
                    "Trained model not found. Please execute 'python train_model.py' to generate the model artifact."
                ]
            }), 500

    # Parse incoming request data (handles JSON or standard Form POST)
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict()

    errors = []
    validated_values = {}

    # Validate each of the 7 features against agronomic boundaries
    for feature in FEATURE_NAMES:
        cfg = FEATURE_RANGES[feature]
        val_str = data.get(feature)

        if val_str is None or str(val_str).strip() == '':
            errors.append(f"Missing required field: '{cfg['label']}'.")
            continue

        try:
            val_float = float(val_str)
        except (ValueError, TypeError):
            errors.append(f"'{cfg['label']}' must be a valid numeric value.")
            continue

        if val_float < cfg['min'] or val_float > cfg['max']:
            errors.append(
                f"'{cfg['label']}' ({val_float}) is out of acceptable agronomic bounds "
                f"[{cfg['min']} to {cfg['max']}{' ' + cfg['unit'] if cfg['unit'] else ''}]."
            )
            continue

        validated_values[feature] = val_float

    if errors:
        return jsonify({
            'success': False,
            'errors': errors
        }), 400

    try:
        # Construct feature DataFrame with explicit column names
        feature_df = pd.DataFrame([[validated_values[col] for col in FEATURE_NAMES]], columns=FEATURE_NAMES)

        # Execute classification
        prediction = model_instance.predict(feature_df)[0]
        crop_key = str(prediction).strip().lower()

        # Retrieve rich metadata or build fallback
        crop_info = CROP_METADATA.get(crop_key, {
            'icon': '🌱',
            'category': 'Agricultural Crop',
            'description': f'{crop_key.capitalize()} is recommended for your soil nutrient and climatic conditions.',
            'season': 'Regional Crop Season',
            'water': 'Moderate',
            'soil': 'Standard Farm Soil',
            'ph_range': '6.0 - 7.5'
        })

        return jsonify({
            'success': True,
            'crop_name': crop_key.capitalize(),
            'crop_info': crop_info,
            'input_data': validated_values
        }), 200

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({
            'success': False,
            'errors': [f"Internal classification error: {str(e)}"]
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_instance is not None
    })

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'errors': ['An unexpected server error occurred.']}), 500

if __name__ == '__main__':
    # Local development server
    port = int(os.environ.get('PORT', 5000))
    print(f"\n* Starting Basic Crop Recommendation System on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
