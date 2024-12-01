# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

loan_data

def outlier_removal(loan_data):
    loan_data1=loan_data.loc[:,['ApplicantIncome','CoapplicantIncome','LoanAmount']]
    loan_data1=loan_data1.dropna()
    loan_data2=loan_data1.sort_values(['ApplicantIncome','CoapplicantIncome','LoanAmount'])
    upper_quartile = np.percentile(loan_data2,95)
    h2=loan_data[loan_data2<upper_quartile]
    return(h2)
outlier_removal(loan_data)
# Write your Solution here:


