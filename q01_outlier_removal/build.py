# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(loan_data):
    col = loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
    quantile_all = loan_data.quantile(0.95)

    col_name = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
    for x in col:
    
        loan_data = loan_data.drop(loan_data[loan_data[x]>quantile_all[x]].index)
    return loan_data

outlier_removal(loan_data)



