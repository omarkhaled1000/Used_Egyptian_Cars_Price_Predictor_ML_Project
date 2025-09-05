from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__, template_folder='templates', static_folder='static')

# === Load artifacts ===
MODEL_PATH = Path('Model/xgboost_car_price_model.pkl')
ARTIFACTS_PATH = Path('Model/artifacts.joblib')

model = joblib.load(MODEL_PATH)
artifacts = joblib.load(ARTIFACTS_PATH)

brand_classes = artifacts['brand_classes']
model_classes = artifacts['model_classes']
feature_columns = artifacts['feature_columns']

fuel_options = artifacts['fuel_options']
transmission_options = artifacts['transmission_options']
body_options = artifacts['body_options']
color_options = artifacts['color_options']
gov_options = artifacts['gov_options']

# Create index maps for Brand/Model to replicate LabelEncoder behavior
brand_index = {v: i for i, v in enumerate(brand_classes)}
model_index = {v: i for i, v in enumerate(model_classes)}

# Create brand-model mapping
# Since models don't have brand prefixes, we'll create a mapping based on common knowledge
# This is a simplified approach - in a real scenario, you'd want to load this from your training data
brand_model_mapping = {
    'Chevrolet': ['Cruze', 'Aveo', 'Lanos', 'Optra'],
    'Fiat': ['128', '131', 'Punto', 'Tipo', 'Uno'],
    'Hyundai': ['Accent', 'Avante', 'Elantra', 'Excel', 'I10', 'Tucson', 'Tucson GDI', 'Verna', 'Matrix', 'Shahin']
}

# Filter to only include models that actually exist in our model_classes
filtered_brand_model_mapping = {}
for brand, models in brand_model_mapping.items():
    if brand in brand_classes:
        filtered_models = [model for model in models if model in model_classes]
        if filtered_models:
            filtered_brand_model_mapping[brand] = filtered_models

brand_model_mapping = filtered_brand_model_mapping

# Helper: build a single-row feature vector in the exact training order
# Input payload should contain raw fields: Brand, Model, Year, Fuel, Trasmission, Body, Color, Gov, Engine, Kilometers
# We'll compute Car_Age = current_year - Year on the server for consistency.
from datetime import datetime
CURRENT_YEAR = datetime.now().year

ONEHOT_PREFIXES = ['Fuel', 'Trasmission', 'Body', 'Color', 'Gov']


def build_features(payload: dict) -> pd.DataFrame:
    # 1) Start with base columns
    year = int(payload.get('Year'))
    car_age = CURRENT_YEAR - year

    # Numeric inputs
    engine = float(payload.get('Engine'))
    kilometers = float(payload.get('Kilometers'))

    # Encoded Brand/Model
    brand = str(payload.get('Brand'))
    model_str = str(payload.get('Model'))

    if brand not in brand_index:
        # For unseen brand, fallback to the most frequent class index 0
        # (You can customize: choose index of a common brand in your data)
        brand_id = 0
    else:
        brand_id = brand_index[brand]

    if model_str not in model_index:
        model_id = 0
    else:
        model_id = model_index[model_str]

    # 2) Construct a DataFrame with all expected feature columns initialized to 0
    data = {col: 0 for col in feature_columns}

    # 3) Fill known numeric columns
    # These should exist in feature_columns from training
    if 'Car_Age' in data:
        data['Car_Age'] = car_age
    if 'Engine' in data:
        data['Engine'] = engine
    if 'Kilometers' in data:
        data['Kilometers'] = kilometers

    # 4) Fill label-encoded columns for Brand and Model
    if 'Brand' in data:
        data['Brand'] = brand_id
    if 'Model' in data:
        data['Model'] = model_id

    # 5) Set one-hot flags where the exact dummy column exists
    for prefix in ONEHOT_PREFIXES:
        val = str(payload.get(prefix))
        col_name = f"{prefix}_{val}"
        if col_name in data:
            data[col_name] = 1
        # If the exact dummy column doesn't exist (unseen category), we simply leave all zeros for that group.

    # 6) Return a single-row DataFrame ordered by feature_columns
    X_row = pd.DataFrame([[data[c] for c in feature_columns]], columns=feature_columns)
    return X_row


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/meta', methods=['GET'])
def meta():
    """Return option lists so the UI can populate dropdowns."""
    return jsonify({
        'brands': brand_classes,
        'models': model_classes,
        'fuel_options': fuel_options,
        'transmission_options': transmission_options,
        'body_options': body_options,
        'color_options': color_options,
        'gov_options': gov_options,
        'current_year': CURRENT_YEAR,
        'brand_model_mapping': brand_model_mapping
    })


@app.route('/models/<brand>', methods=['GET'])
def get_models_for_brand(brand):
    """Return models for a specific brand."""
    if brand in brand_model_mapping:
        return jsonify({'models': brand_model_mapping[brand]})
    else:
        return jsonify({'models': []})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    try:
        X_row = build_features(payload)
        pred = model.predict(X_row)[0]
        # If your target was in thousands, keep it consistent in UI wording
        return jsonify({
            'ok': True,
            'prediction': float(pred)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
