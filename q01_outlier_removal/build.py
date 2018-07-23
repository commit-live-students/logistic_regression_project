# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(loan_data):
    loan_data = loan_data[(loan_data.LoanAmount <= loan_data.LoanAmount.quantile(0.962)) & (loan_data.LoanAmount >= loan_data.LoanAmount.quantile(0.04))]
    return loan_data

