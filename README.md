# Loan Eligibility Prediction

![screenshot-localhost_8888-2025 04 07-09_19_56](https://github.com/user-attachments/assets/970d27f7-734e-40c9-ad9f-0620758cd41e)

### Overview

This project predicts loan approval decisions using machine learning models trained on applicant financial and demographic data. The goal is to automate and improve the accuracy of loan eligibility assessments for financial institutions.

### Dataset

#### Columns
- Demographics: Gender, Married, Dependents, Education, Self_Employed
- Financials: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
- Property: Property_Area (Urban/Rural/Semiurban)
- Target: Loan_Status (Y/N)

### Key Steps

1. Data Preprocessing
Handled missing values:
- Categorical: Filled with mode (e.g., Gender, Self_Employed).
- Numerical: Filled with median/mode (e.g., LoanAmount, Credit_History).

Feature engineering:
- Created Income_per_dependent, EMI, Balance_Income.
- Log-transformed skewed features (LoanAmount, Total_Income).

Encoding:
- Mapped categorical variables (e.g., Loan_Status: Y=1, N=0).
- One-hot encoded Education and Property_Area.

2. Exploratory Analysis (EDA)
- Approval Rates by Property Area: Semiurban areas had the highest approval rate.
- Statistical Tests: Chi-square confirmed property area significantly impacts approval (p < 0.05).
- Visualizations: Bar plots, ROC curves, and confusion matrices.

3. Model Training & Evaluation
- Logistic Regression	82.00% (Accuracy), 	84%	52% (Precision)
- Random Forest	83.00% (Accuracy), 	79%	61% (Precision)
- Tuned Random Forest	84.00% (Accuracy), 	89%	55% (Precision)
- Best Model: Tuned Random Forest

### Key Takeaways

- Credit History and Income Stability were the top predictors of approval.
- Trade-off: Higher precision for rejections (89%) but lower recall (55%) means the model is conservative—approving fewer risky loans but missing some eligible applicants.
- Next Steps: Address class imbalance, test gradient boosting (XGBoost), or build a frontend for loan officers.

### Conclusion

The tuned Random Forest (84% accuracy) strikes the best balance for this use case, but further refinement (e.g., threshold adjustment) could optimize risk vs. customer experience.

### Source

[Loan Application Data from Kaggle](https://www.kaggle.com/datasets/vipin20/loan-application-data)

