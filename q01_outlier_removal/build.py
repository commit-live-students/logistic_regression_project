# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    num_cols = data[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
    
    quantile_values = num_cols.quantile(0.95)
    
    for col in num_cols:
        quantile = quantile_values[col]
        print(quantile)
        data = data.drop(data[data[col]>quantile].index)
        
    return data

outlier_removal(loan_data)


