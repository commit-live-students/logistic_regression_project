# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
def outlier_removal(loan_data):
    #loan_data = loan_data.drop(loan_data[(loan_data['ApplicantIncome'] > loan_data['ApplicantIncome'].quantile(0.95)) & (loan_data['CoapplicantIncome'] > loan_data['CoapplicantIncome'].quantile(0.95)) & (loan_data['LoanAmount'] > loan_data['LoanAmount'].quantile(0.95))].index)
    #loan_data = loan_data.drop(loan_data[loan_data['CoapplicantIncome'] > loan_data['CoapplicantIncome'].quantile(0.95)].index)
    #loan_data = loan_data.drop(loan_data[loan_data['LoanAmount'] > loan_data['LoanAmount'].quantile(0.95)].index)
    loan_data = loan_data.drop(loan_data[loan_data['CoapplicantIncome'] > 4997.400000].index)
    loan_data = loan_data.drop(loan_data[loan_data['LoanAmount'] > 297.800000].index)
    loan_data = loan_data.drop(loan_data[loan_data['ApplicantIncome'] > 14583.000000].index)
    return loan_data

#print outlier_removal(loan_data)


# Write your Solution here:






