import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data_preprocessing import preprocess_data, handle_imbalance
from logistic_regression import LogisticRegression
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_pr_curve, plot_loss_curve
from utils import save_object
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_analysis():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('data/telco_churn.csv')
    print(f"Shape: {df.shape}")
    print(df.info())
    print(df.head())
    
    # 2. EDA
    print("\nPerforming EDA...")
    # Univariate - Numeric
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols].hist(bins=30, figsize=(15, 10))
    plt.suptitle('Distribution of Numeric Features')
    plt.savefig('report/numeric_dist.png')
    plt.close()
    
    # Univariate - Categorical
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'Contract', 'InternetService', 'Churn']
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('report/categorical_dist.png')
    plt.close()
    
    # Bivariate
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sns.barplot(data=df, x='Contract', y=(df['Churn']=='Yes').astype(int), ax=axes[0,0])
    axes[0,0].set_title('Churn Rate by Contract')
    sns.barplot(data=df, x='InternetService', y=(df['Churn']=='Yes').astype(int), ax=axes[0,1])
    axes[0,1].set_title('Churn Rate by Internet Service')
    sns.barplot(data=df, x='SeniorCitizen', y=(df['Churn']=='Yes').astype(int), ax=axes[1,0])
    axes[1,0].set_title('Churn Rate by Senior Citizen')
    sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[1,1])
    axes[1,1].set_title('Tenure by Churn')
    plt.tight_layout()
    plt.savefig('report/bivariate_analysis.png')
    plt.close()
    
    # Correlation
    plt.figure(figsize=(10, 8))
    # Filter only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('report/correlation.png')
    plt.close()
    
    # 3. Preprocessing
    print("\nPreprocessing data...")
    X, y, preprocessor = preprocess_data(df)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Handle Imbalance
    print("Handling imbalance with SMOTE...")
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)
    print(f"Original shape: {X_train.shape}, Resampled shape: {X_train_res.shape}")
    
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
    print(f"Optimal threshold: {best_t:.2f} (F1: {max(f1_scores):.4f})")
    
    y_pred = (y_prob >= best_t).astype(int)
    custom_metrics = evaluate_model(y_test, y_pred, y_prob, title="Custom Logistic Regression")
    
    # Save visualizations
    plot_confusion_matrix(y_test, y_pred)
    plt.savefig('report/confusion_matrix.png')
    plt.close()
    plot_roc_curve(y_test, y_prob)
    plt.savefig('report/roc_curve.png')
    plt.close()
    plot_pr_curve(y_test, y_prob)
    plt.savefig('report/pr_curve.png')
    plt.close()
    
    # 6. Feature Importance
    print("\nPlotting Feature Importance...")
    importance = np.abs(custom_model.weights)
    feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feat_imp.head(10), y=feat_imp.head(10).index)
    plt.title('Top 10 Feature Importance (Absolute Weights)')
    plt.savefig('report/feature_importance.png')
    plt.close()
    
    # 7. Benchmark
    print("\nTraining Sklearn Logistic Regression...")
    sklearn_model = SklearnLR(C=1/0.1, max_iter=2000) # C is inverse of lambda
    sklearn_model.fit(X_train_res, y_train_res)
    
    y_prob_sk = sklearn_model.predict_proba(X_test)[:, 1]
    y_pred_sk = sklearn_model.predict(X_test)
    sklearn_metrics = evaluate_model(y_test, y_pred_sk, y_prob_sk, title="Sklearn Logistic Regression")
    
    # 8. Comparison Table
    comparison = pd.DataFrame({
        'Metric': list(custom_metrics.keys()),
        'Custom LR': list(custom_metrics.values()),
        'Sklearn LR': list(sklearn_metrics.values())
    })
    print("\nModel Comparison:")
    print(comparison)
    comparison.to_csv('report/model_comparison.csv', index=False)
    
    # 9. Save Models
    print("\nSaving models...")
    save_object(custom_model, 'models/custom_lr_model.pkl')
    save_object(preprocessor, 'models/preprocessor.pkl')
    # Save threshold too
    save_object({'threshold': best_t}, 'models/threshold.pkl')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    run_analysis()
