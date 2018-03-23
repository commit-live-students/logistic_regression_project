# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(loan_data):
    l = loan_data[["ApplicantIncome","CoapplicantIncome","LoanAmount"]]
    q = loan_data.quantile(0.95)
    for i in l:
        loan_data = loan_data.drop(loan_data[loan_data[i] > q[i]].index)
    return loan_data
