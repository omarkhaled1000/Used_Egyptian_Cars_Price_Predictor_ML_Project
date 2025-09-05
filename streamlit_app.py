from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st

# === Load artifacts ===
MODEL_PATH = Path('Model/xgboost_car_price_model.pkl')
ARTIFACTS_PATH = Path('Model/artifacts.joblib')

ml_model = joblib.load(MODEL_PATH)
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

# Build brand→models mapping (based on observed brands/models, filtered to existing models)
_brand_model_seed = {
    'Chevrolet': ['Cruze', 'Aveo', 'Lanos', 'Optra'],
    'Fiat': ['128', '131', 'Punto', 'Tipo', 'Uno'],
    'Hyundai': ['Accent', 'Avante', 'Elantra', 'Excel', 'I10', 'Tucson', 'Tucson GDI', 'Verna', 'Matrix', 'Shahin']
}
brand_model_mapping = {
    brand: [m for m in models if m in set(model_classes)]
    for brand, models in _brand_model_seed.items()
    if brand in set(brand_classes)
}

CURRENT_YEAR = datetime.now().year
ONEHOT_PREFIXES = ['Fuel', 'Trasmission', 'Body', 'Color', 'Gov']


def build_features(payload: dict) -> pd.DataFrame:
    year = int(payload.get('Year'))
    car_age = CURRENT_YEAR - year

    engine = float(payload.get('Engine'))
    kilometers = float(payload.get('Kilometers'))

    brand = str(payload.get('Brand'))
    model_str = str(payload.get('Model'))

    brand_id = brand_index.get(brand, 0)
    model_id = model_index.get(model_str, 0)

    data = {col: 0 for col in feature_columns}

    if 'Car_Age' in data:
        data['Car_Age'] = car_age
    if 'Engine' in data:
        data['Engine'] = engine
    if 'Kilometers' in data:
        data['Kilometers'] = kilometers

    if 'Brand' in data:
        data['Brand'] = brand_id
    if 'Model' in data:
        data['Model'] = model_id

    for prefix in ONEHOT_PREFIXES:
        val = str(payload.get(prefix))
        col_name = f"{prefix}_{val}"
        if col_name in data:
            data[col_name] = 1

    X_row = pd.DataFrame([[data[c] for c in feature_columns]], columns=feature_columns)
    return X_row


# === UI ===
st.set_page_config(page_title='Car Price Predictor', page_icon='🚗', layout='centered')
st.title('Car Price Predictor')
st.caption('Fill in the details and click Predict. Powered by an XGBoost model.')

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox('Brand', options=brand_classes, index=0 if len(brand_classes) > 0 else None)
with col2:
    # Filter models by selected brand
    models_for_brand = brand_model_mapping.get(brand, model_classes)
    selected_model = st.selectbox('Model', options=models_for_brand, index=0 if len(models_for_brand) > 0 else None)

col3, col4 = st.columns(2)
with col3:
    year = st.number_input('Year', min_value=1990, max_value=2100, value=CURRENT_YEAR, step=1)
with col4:
    engine = st.number_input('Engine (CC)', min_value=600, max_value=6000, value=1600, step=50)

kilometers = st.number_input('Kilometers', min_value=0, value=0, step=1000)

col5, col6 = st.columns(2)
with col5:
    fuel = st.selectbox('Fuel', options=fuel_options)
with col6:
    transmission = st.selectbox('Transmission', options=transmission_options)

col7, col8 = st.columns(2)
with col7:
    body = st.selectbox('Body', options=body_options)
with col8:
    color = st.selectbox('Color', options=color_options)

gov = st.selectbox('Governorate', options=gov_options)

if st.button('Predict'):
    payload = {
        'Brand': brand,
        'Model': selected_model,
        'Year': year,
        'Engine': engine,
        'Kilometers': kilometers,
        'Fuel': fuel,
        'Trasmission': transmission,
        'Body': body,
        'Color': color,
        'Gov': gov
    }

    with st.spinner('Predicting...'):
        try:
            X_row = build_features(payload)
            pred = float(ml_model.predict(X_row)[0])
            st.success(f'Estimated Price: {pred:.2f}')
        except Exception as e:
            st.error(f'Error: {e}')
