import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_rows=7000, output_path='data/telco_churn.csv'):
    """
    Generates a synthetic telco churn dataset with specified requirements.
    """
    np.random.seed(42)
    
    data = {
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'SeniorCitizen': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_rows),
        'Dependents': np.random.choice(['Yes', 'No'], n_rows),
        'tenure': np.random.randint(0, 73, n_rows),
        'PhoneService': np.random.choice(['Yes', 'No'], n_rows, p=[0.9, 0.1]),
    }
    
    # Dependent features
    data['MultipleLines'] = []
    for ps in data['PhoneService']:
        if ps == 'No':
            data['MultipleLines'].append('No phone service')
        else:
            data['MultipleLines'].append(np.random.choice(['Yes', 'No']))
            
    data['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], n_rows, p=[0.3, 0.4, 0.3])
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    for service in services:
        data[service] = []
        for is_val in data['InternetService']:
            if is_val == 'No':
                data[service].append('No internet service')
            else:
                data[service].append(np.random.choice(['Yes', 'No']))
                
    data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows, p=[0.5, 0.25, 0.25])
    data['MonthlyCharges'] = np.random.uniform(18.0, 120.0, n_rows).round(2)
    
    # TotalCharges = tenure * MonthlyCharges + noise
    noise = np.random.normal(0, 50, n_rows)
    total_charges = (data['tenure'] * data['MonthlyCharges'] + noise).round(2)
    data['TotalCharges'] = np.maximum(0, total_charges) # Ensure non-negative
    
    # Churn probability based on features (to make it learnable)
    # Higher churn for: Month-to-month, Fiber optic, High MonthlyCharges, Low tenure
    churn_prob = 0.1
    
    # Simplified logic for synthetic churn generation
    probs = np.zeros(n_rows)
    for i in range(n_rows):
        p = 0.2 # baseline
        if data['Contract'][i] == 'Month-to-month': p += 0.3
        if data['InternetService'][i] == 'Fiber optic': p += 0.2
        if data['tenure'][i] < 12: p += 0.2
        if data['MonthlyCharges'][i] > 80: p += 0.1
        if data['SeniorCitizen'][i] == 1: p += 0.05
        
        probs[i] = p
        
    # Normalize and target ~27% churn rate
    target_churn_rate = 0.27
    threshold = np.percentile(probs, (1 - target_churn_rate) * 100)
    data['Churn'] = ['Yes' if p >= threshold else 'No' for p in probs]
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path} with {n_rows} rows.")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")

if __name__ == "__main__":
    generate_synthetic_data()
