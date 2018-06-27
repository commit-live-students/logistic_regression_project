# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(loan_data):
    numeric_columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    IQRU=list()
    for c in numeric_columns:
        IQRU.append([c,loan_data[c].quantile(0.95)])
    for c in IQRU:
        loan_data=loan_data.drop(loan_data[(loan_data[c[0]]>c[1])].index)
    return loan_data


