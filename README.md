✈️ Flight Delay Prediction using Machine Learning

📌 Project Overview

This project aims to predict whether a flight will be delayed based on historical data from US domestic airlines in 2015. The problem is treated as a binary classification task where a delay of more than 15 minutes is considered a delay.

🧠 Problem Statement

Flight delays can impact passengers, airlines, and airport operations. Using flight and airport data, we predict whether a flight will be delayed. The goal is to develop a robust machine learning pipeline for accurate and scalable predictions.

📊 Dataset

Source: Kaggle – Airline Delay and Cancellation Data 2009–2018
Year Used: 2015 only (to reduce size and simplify processing)
Target Variable: Delayed (1 = Delayed over 15 minutes, 0 = On-time)
🔍 Key Findings from EDA

Majority of flights are not delayed.
ARR_DELAY, LATE_AIRCRAFT_DELAY, and NAS_DELAY were top influencing features as shown by SHAP.
Flights from certain carriers and airports tend to have higher delays.
Some features had many missing values and were either imputed or dropped.
⚙️ Modeling Choices

Three models were trained and evaluated using 26 features:

✅ Logistic Regression
✅ Random Forest
✅ XGBoost Classifier (Best Performance)
Later, for deployment purposes, an XGBoost pipeline was retrained using only 5 selected features to reduce complexity and size while maintaining good accuracy.

🧪 Model Performance

Three models were trained and evaluated on the dataset using 26 features. The Logistic Regression model achieved an accuracy of 99% with a ROC AUC score of 0.989, demonstrating high precision and recall. The Random Forest model achieved perfect accuracy and ROC AUC of 1.0, classifying all instances correctly. Similarly, the XGBoost Classifier also achieved 100% accuracy and a ROC AUC of 1.0, making it the best-performing model. Due to its excellent performance and efficiency, XGBoost was selected for deployment and later retrained using only five selected features to simplify the model without compromising much on accuracy.

🚀 Deployment

The final model (XGBoost trained on 5 features) was deployed using Streamlit to create a lightweight and interactive web app for predicting flight delays. The app allows users to input flight details like distance, departure hour, carrier, origin, and destination, and get a real-time prediction on whether the flight will be delayed.

To run the app locally: streamlit run streamlit_app.py

📦 Pipeline

A full Scikit-learn Pipeline was created using:

ColumnTransformer for preprocessing (numeric + categorical)
XGBoostClassifier for training
Saved using joblib as xgb_pipeline_model.pkl
📈 Visualizations

Bar chart showing flight delay distribution
SHAP summary plot highlighting feature importance
🛠️ Tools & Technologies

Python
Pandas, NumPy, scikit-learn
XGBoost
SHAP
Google Colab & VS Code
Matplotlib, Seaborn

📁 Project Structure

Flight-Delay-Prediction/
├── data/                        # Dataset (if included or sample)
├── notebook/                    # Jupyter or Colab Notebook
├── scripts/                     # Python pipeline script
├── models/                      # Saved ML models
├── visuals/                     # Plots/images
├── requirements.txt             # Required packages
├── README.md                    # This file

▶️ How to Run

# Create virtual environment
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python scripts/pipeline_model.py

✍️ Author

Arpita Panigrahi – MCA (AI & ML)
From Odisha, India 🇮🇳 | Passionate about applying ML in real-world problems
mail ID- arpitapanigrahi63@gmail.com
LinkedIn - www.linkedin.com/in/arpita-panigrahi-429b70269
