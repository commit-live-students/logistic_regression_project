# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

# Function will remove the outliers
def outlier_removal(loan_data):
    a = loan_data['ApplicantIncome'].quantile(0.95) # It comes out to be 14583.0
    b = loan_data['CoapplicantIncome'].quantile(0.95) # It comes out to be 4997.4
    c = loan_data['LoanAmount'].quantile(0.95) # It comes out to be 297.8
    loan_data = loan_data.drop(loan_data[loan_data['ApplicantIncome'] > a].index)
    loan_data = loan_data.drop(loan_data[loan_data['CoapplicantIncome'] > b].index)
    loan_data = loan_data.drop(loan_data[loan_data['LoanAmount'] > c].index)
    return loan_data

outlier_removal(loan_data).shape










