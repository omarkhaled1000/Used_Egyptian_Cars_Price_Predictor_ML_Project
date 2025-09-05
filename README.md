# 🚗 Car Price Prediction with Machine Learning
## DATASET LICENSE
This project uses the dataset [Used Cars Prices in Egypt]([Kaggle link](https://www.kaggle.com/datasets/yousifahmedanwar/used-cars-prices-in-egypt)), which is licensed under the GNU GPL v2.0.  
The dataset is redistributed (if included) under the same license.  

## 📌 Project Overview
This project predicts used car prices based on attributes such as brand, model, year, fuel type, mileage, engine capacity, transmission, body type, color, and governorate.  
It also provides two ways to interact with the trained model:

1. **Locally** via a Flask web app (`app.py`).  
2. **Online** via a Streamlit app (deployed on Streamlit Cloud).  

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
Cars-AMIT-Project/
├── app.py # Flask backend for local deployment
├── app_streamlit.py # Streamlit app for online deployment
├── requirements.txt # Dependencies
├── Data/
│ └── merged_cars.csv # Cleaned dataset
├── Model/
│ ├── xgboost_car_price_model.pkl # Trained tuned XGBoost model
│ └── artifacts.joblib # Preprocessing artifacts (encoders, feature columns, options)
├── car_price_prediction.ipynb # Full ML workflow
├── templates/
│ └── index.html # Frontend for Flask app
└── pycache/ # Python cache files (auto-generated)


---

## ⚙️ Installation and Usage

## Run Locally (Flask App)
1. Clone the repository:
   ```bash
   git clone https://github.com/omarkhaled1000/Cars-AMIT-Project.git
   cd Cars-AMIT-Project
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Flask app:
```bash
python app.py
```
4. Open your browser at: http://127.0.0.1:5000/

 
## ⚙ Run Online (Streamlit App)

Go to the deployed Streamlit app:
👉 Live Demo on Streamlit
 (https://cars-amit-project-43vmivyimssrjesjkhjwuf.streamlit.app/)
You can enter car details in the sidebar form.
Get instant price predictions in the browser.

## 🧠 Model Architecture
Input Features:
Brand, Model, Year, Mileage, Engine Size, Fuel Type, Transmission, Body Type, Color, Governorate, and Car Condition (New/Used).
Encoding: One-hot encoding for categorical variables.
Models Tested:
Linear Regression
Random Forest
XGBoost (chosen as final model)
Final Model: Tuned XGBoost Regressor with optimized hyperparameters.

## 📊 Model Performance

Best Model: Tuned XGBoost
MAE: ~12.4
RMSE: ~20.3
R²: ~0.93

## 🤝 Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

## Support
For support or questions about this project, please open an issue in the GitHub repository.


## 🔮 Future Improvements Vision

Deploy a full backend + database to store user predictions and feedback.
Improve the dataset by adding more diverse car brands and newer models.
Experiment with deep learning models for regression.
Add more visualizations in the Streamlit app (feature impact, SHAP plots).
Deploy the project on Docker and cloud platforms (AWS/GCP/Azure) for scalability.
