# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(loan_data):
    num_cols = loan_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
    quant = num_cols.quantile(0.95)
    for col in num_cols:
        loan_data = loan_data.drop(loan_data[loan_data[col]>quant[col]].index)
    return loan_data

