# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np
loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(loan_data):
    q1 = loan_data['ApplicantIncome'].quantile(q=0.95)
    q2 = loan_data['CoapplicantIncome'].quantile(q=0.95)
    q3 = loan_data['LoanAmount'].quantile(q=0.95)
    loan_numeric_filtered = loan_data[np.invert((loan_data['ApplicantIncome']>q1) | (loan_data['CoapplicantIncome']>q2) | (loan_data['LoanAmount']>q3))]
    return loan_numeric_filtered
outlier_removal(loan_data)



