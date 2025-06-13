import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/2015.csv")

# Add this line to create the 'Delayed' target column
df['Delayed'] = (df['ARR_DELAY'] > 15).astype(int)


# Convert categorical columns to string
cat_cols = ['OP_CARRIER', 'ORIGIN', 'DEST', 'CANCELLATION_CODE']
for col in cat_cols:
    df[col] = df[col].astype(str)

# Drop unnecessary columns
df.drop(columns=['FL_DATE'], inplace=True)

# Separate features and target
X = df.drop(columns=['Delayed'])
y = df['Delayed']

# Identify numeric columns
num_cols = [col for col in X.columns if col not in cat_cols]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Create full pipeline with XGBoost
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print("--- XGBoost Pipeline ---")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the pipeline
joblib.dump(pipeline, "xgb_pipeline_model.pkl")

# Optional: Visualize class distribution
sns.countplot(x=y)
plt.title("Flight Delay Class Distribution")
plt.show()
