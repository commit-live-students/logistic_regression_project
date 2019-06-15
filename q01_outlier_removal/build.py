# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(loan_data):
    loan = loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
    q = loan_data.quantile(0.95)
    for l in loan:
        loan_data = loan_data.drop(loan_data[loan_data[l] > q[l]].index)
    return loan_data
outlier_removal(loan_data).shape

