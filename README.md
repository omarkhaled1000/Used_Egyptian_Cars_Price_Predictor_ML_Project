# 🚗 Car Price Prediction with Machine Learning

## 📌 Project Overview
This project predicts used car prices based on various features such as brand, model, year, mileage, fuel type, and more.  
The model is deployed as a **Flask web application** with a simple **HTML frontend** for real-time predictions.

---

## ⚙️ Features
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Training & Evaluation  
  - Linear Regression  
  - Random Forest  
  - XGBoost (tuned & selected as final model)  
- Model Explainability with SHAP  
- Deployment with Flask + HTML frontend  

---

## 🗂 Project Structure
├── app.py # Flask backend
├── templates/
│ └── index.html # HTML frontend
├── xgboost_car_price_model.pkl # Trained model
├── artifacts.joblib # Encoders, feature columns, dropdown options
├── notebooks/ # Jupyter/Colab notebooks with full workflow
│ └── car_price_prediction.ipynb
└── README.md



---

## 🚀 Installation & Usage
pip install -r requirements.txt

📊 Model Performance
Best Model: Tuned XGBoost
MAE: ~12.4
RMSE: ~20.3
R²: ~0.93

🤝 Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
