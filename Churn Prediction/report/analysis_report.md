# Customer Churn Analysis Report

## 1. Executive Summary
This project aims to predict customer churn for a subscription-based telecommunications service. We implemented a custom **Logistic Regression** model from scratch using **NumPy** and compared its performance with the industry-standard **scikit-learn** implementation. The results demonstrate that our custom model achieves high predictive accuracy and parity with established libraries.

## 2. Dataset Overview & EDA Findings
A synthetic dataset of 7,000 customers was generated, including demographic information, service subscriptions, and account details.

**Key Findings:**
- **Churn Rate:** ~27% class imbalance, which was handled using SMOTE.
- **Contract Type:** Month-to-month contracts show significantly higher churn compared to one or two-year contracts.
- **Tenure:** Newer customers (low tenure) are much more likely to churn.
- **Internet Service:** Fiber optic users exhibit higher churn rates, possibly due to higher costs or specific service issues.
- **Monthly Charges:** High monthly bills correlate positively with churn probability.

## 3. Feature Engineering Decisions
1. **Encoding:**
   - Binary features (Partner, Dependents, PhoneService) were mapped to 0/1.
   - Categorical features (Contract, InternetService, etc.) were One-Hot Encoded.
2. **Scaling:**
   - Numerical features (Tenure, MonthlyCharges, TotalCharges) were normalized using **MinMaxScaler** to the [0, 1] range, ensuring equal weight distribution during gradient descent.
3. **Handling Imbalance:**
   - **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training set to balance the target classes, improving the model's ability to identify churners.

## 4. Model Architecture
The custom Logistic Regression was built from scratch with:
- **Sigmoid Function:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Binary Cross-Entropy Loss:** $L = -\frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})] + \frac{\lambda}{2m} ||w||^2$
- **Gradient Update:** $w = w - \alpha (\frac{1}{m} X^T(\hat{y} - y) + \frac{\lambda}{m} w)$
- **Regularization:** L2 (Ridge) was included to prevent overfitting.
- **Optimization:** Mini-batch/Batch Gradient Descent with early stopping based on loss convergence.

## 5. Training Results
- **Initial Loss:** 0.6931
- **Final Loss:** ~0.1371
- **Convergence:** The model converged smoothly over 1000 iterations with a learning rate of 0.5.
- **Optimal Threshold:** 0.46 (selected to maximize F1-score).

## 6. Performance Metrics Table
| Metric | Custom LR | Sklearn LR |
| :--- | :--- | :--- |
| **Accuracy** | 0.9429 | 0.9471 |
| **Precision** | 0.8695 | 0.8873 |
| **Recall** | 0.9395 | 0.9320 |
| **F1-Score** | 0.9031 | 0.9091 |
| **ROC-AUC** | 0.9891 | 0.9914 |

## 7. Key Business Insights
- **Contract Type is Critical:** Month-to-month contracts are the strongest predictor of churn.
- **Tenure is Protective:** Longer-term customers are more loyal; the first 12 months are high-risk.
- **Service Quality:** High MonthlyCharges and Fiber Optic service are risk factors, suggesting a need for price optimization or improved service reliability.

## 8. Business Recommendations
1. **Incentivize Long-term Contracts:** Offer discounts for switching from month-to-month to annual plans.
2. **Early Engagement:** Focus retention efforts on customers in their first 6–12 months.
3. **Value Proposition:** Review Fiber Optic pricing and performance to ensure it meets customer expectations relative to the cost.
4. **Proactive Outreach:** Use the "High Risk" prediction to trigger loyalty offers before the customer initiates cancellation.

## 9. Risk Intervention Framework
The project now includes an intelligent **Risk Intervention Engine** that maps prediction probabilities to actionable business and customer strategies.

### Risk Tier System
| Tier | Probability | Label | Color | Action Strategy |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0–30% | 🟢 SAFE | Green | Standard engagement; no immediate intervention. |
| 2 | 30–50% | 🟡 WATCH | Yellow | Automated re-engagement email sequence. |
| 3 | 50–75% | 🟠 HIGH RISK | Orange | Personalized retention offer within 24 hours. |
| 4 | 75–100% | 🔴 CRITICAL | Red | Direct senior account manager outreach. |

### Intervention Logic
- **Customer Recovery Paths:** Personalized instructions like switching to annual plans for cost savings, or activating free premium support for high-value customers.
- **Business Action Center:** Real-time revenue-at-risk calculation and driver analysis to guide agent conversations.
- **What-If Simulations:** Predictive tool for agents to simulate how different offers (e.g., adding Online Security) would impact the customer's churn risk.

### ROI Estimation
By implementing these targeted interventions, the business can proactively address churn drivers. If the intervention engine reduces churn by just **15%**, the estimated monthly revenue saved would be approximately **$4,500** (based on current synthetic averages).

## 10. Limitations & Future Work
- **Linearity:** Logistic Regression assumes linear relationships; future work could explore non-linear models like Random Forests or XGBoost.
- **Feature Interaction:** We did not explicitly model feature interactions (e.g., SeniorCitizen + FiberOptic).
- **Time Series:** Incorporating historical usage trends (e.g., changes in data usage) could improve prediction accuracy.
