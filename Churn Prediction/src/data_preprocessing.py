import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data(df, is_training=True, preprocessor=None):
    """
    Preprocess the telco churn dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        is_training (bool): Whether to fit or transform only.
        preprocessor (dict): Preprocessor state for prediction.
        
    Returns:
        pd.DataFrame: Preprocessed features.
        pd.Series: Target labels.
    """
    df = df.copy()
    
    # Target encoding
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        y = df['Churn']
    else:
        y = None
        
    # SeniorCitizen is already numeric (0/1)
    
    # Binary encoding (Yes/No -> 1/0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # One-hot encoding for categorical variables
    categorical_cols = [
        'Contract', 'InternetService', 'MultipleLines', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'gender'
    ]
    
    # Special handling for prediction (ensuring all columns exist)
    if not is_training and preprocessor:
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        # Add missing columns with 0s
        for col in preprocessor['columns']:
            if col not in df_encoded.columns and col != 'Churn':
                df_encoded[col] = 0
        # Reorder columns to match training
        df_encoded = df_encoded[preprocessor['columns']]
        
        # Scaling
        scaler = preprocessor['scaler']
        numeric_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
        df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
        
        return df_encoded.drop(columns=['Churn']) if 'Churn' in df_encoded.columns else df_encoded
        
    # For training
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Scaling
    scaler = MinMaxScaler()
    numeric_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    # Store preprocessor state
    preprocessor_state = {
        'columns': [col for col in df_encoded.columns if col != 'Churn'],
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'binary_cols': binary_cols
    }
    
    X = df_encoded.drop(columns=['Churn']) if 'Churn' in df_encoded.columns else df_encoded
    
    return X, y, preprocessor_state

def handle_imbalance(X, y):
    """
    Apply SMOTE to balance the training set.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
