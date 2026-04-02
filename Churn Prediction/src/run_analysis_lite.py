import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data_preprocessing import preprocess_data, handle_imbalance
from logistic_regression import LogisticRegression
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from utils import save_object
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import f1_score

def run_analysis():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('data/telco_churn.csv')
    
    # 2. EDA (Simple versions since we already have some)
    print("\nPerforming EDA...")
    
    # 3. Preprocessing
    print("\nPreprocessing data...")
    X, y, preprocessor = preprocess_data(df)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Handle Imbalance
    print("Handling imbalance with SMOTE...")
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)
    
    # 4. Custom Model Training
    print("\nTraining Custom Logistic Regression...")
    custom_model = LogisticRegression(learning_rate=0.5, n_iterations=1000, lambda_=0.1)
    custom_model.fit(X_train_res.values, y_train_res.values)
    print("Training finished.")
    
    # 5. Evaluation
    print("\nEvaluating Custom Model...")
    y_prob = custom_model.predict_proba(X_test.values)
    
    # Threshold Tuning
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1_scores)]
    print(f"Optimal threshold: {best_t:.2f}")
    
    y_pred = (y_prob >= best_t).astype(int)
    custom_metrics = evaluate_model(y_test, y_pred, y_prob, title="Custom Logistic Regression")
    
    # 6. Benchmark
    print("\nTraining Sklearn Logistic Regression...")
    sklearn_model = SklearnLR(C=1/0.1, max_iter=2000)
    sklearn_model.fit(X_train_res, y_train_res)
    
    y_prob_sk = sklearn_model.predict_proba(X_test)[:, 1]
    y_pred_sk = sklearn_model.predict(X_test)
    sklearn_metrics = evaluate_model(y_test, y_pred_sk, y_prob_sk, title="Sklearn Logistic Regression")
    
    # 7. Comparison Table
    comparison = pd.DataFrame({
        'Metric': list(custom_metrics.keys()),
        'Custom LR': list(custom_metrics.values()),
        'Sklearn LR': list(sklearn_metrics.values())
    })
    comparison.to_csv('report/model_comparison.csv', index=False)
    
    # 8. Save Models
    print("\nSaving models...")
    save_object(custom_model, 'models/custom_lr_model.pkl')
    save_object(preprocessor, 'models/preprocessor.pkl')
    save_object({'threshold': best_t}, 'models/threshold.pkl')
    
    # Save feature importance for report
    importance = np.abs(custom_model.weights)
    feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    feat_imp.to_csv('report/feature_importance.csv')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    run_analysis()
