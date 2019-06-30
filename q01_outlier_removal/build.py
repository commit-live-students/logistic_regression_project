# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(loan_data):
    app_inc = loan_data[['ApplicantIncome']].quantile(0.95)  
    coap_inc = loan_data[['CoapplicantIncome']].quantile(0.95)  
    loan_amt = loan_data[['LoanAmount']].quantile(0.95) 
    loan_data = loan_data.drop(loan_data[(loan_data['ApplicantIncome'] > app_inc[0])].index)
    loan_data = loan_data.drop(loan_data[(loan_data['CoapplicantIncome'] > coap_inc[0])].index)
    loan_data = loan_data.drop(loan_data[(loan_data['LoanAmount'] > loan_amt[0])].index)
    return loan_data

