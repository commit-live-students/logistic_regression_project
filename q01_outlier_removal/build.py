# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(loan_data):
#     col = loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
#     quantile_all = loan_data.quantile(0.95)

#     col_name = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
#     for x in col:
    
#         loan_data = loan_data.drop(loan_data[loan_data[x]>quantile_all[x]].index)
    #num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
    q_ai = loan_data['ApplicantIncome'].quantile(0.95)
    q_ci = loan_data['CoapplicantIncome'].quantile(0.95)
    q_la = loan_data['LoanAmount'].quantile(0.95)
    l_ai = list(loan_data.index[loan_data['ApplicantIncome'] > q_ai])
    l_ci = list(loan_data.index[loan_data['CoapplicantIncome'] > q_ci])
    l_la = list(loan_data.index[loan_data['LoanAmount'] > q_la])
    
    l_95 = list(set(l_ai+l_ci+l_la))
    l_95 = np.sort(l_95) 
    loan_data.drop(loan_data.index[l_95], inplace=True)
    return loan_data

outlier_removal(loan_data)


























