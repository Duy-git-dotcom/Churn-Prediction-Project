import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

with open ('model_weights.pkl','rb') as f:
    model_data = pickle.load(f)
model_loaded = model_data['model']
feature_names = model_data['feature_names']


input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

input_data_df = pd.DataFrame([input_data])

with open ('encoder', 'rb') as f:
    encoder = pickle.load(f)
    
for column, labeled_encode in encoder.items():
    if column != 'Churn' and column in input_data_df.columns:
        input_data_df[column] = labeled_encode.transform(input_data_df[column])

#Chạy model
prediction = model_loaded.predict(input_data_df)
probality = model_loaded.predict_proba(input_data_df)

print(prediction)
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f'Prediction Probality: {probality}')