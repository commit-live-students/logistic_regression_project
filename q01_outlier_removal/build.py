# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(loan_data):
    df= loan_data.quantile(.95)
    loan_data = loan_data.drop(loan_data [loan_data['ApplicantIncome'] > df['ApplicantIncome']].index)
    loan_data = loan_data.drop(loan_data [loan_data['CoapplicantIncome'] > df['CoapplicantIncome']].index)
    loan_data = loan_data.drop(loan_data [loan_data['LoanAmount'] > df['LoanAmount']].index)
    return loan_data 


