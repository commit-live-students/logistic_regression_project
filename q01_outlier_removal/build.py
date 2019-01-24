# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    
    loan_data = data
    loan_data = loan_data[loan_data['ApplicantIncome'] < loan_data['ApplicantIncome'].quantile(0.97)]
    loan_data = loan_data[loan_data['CoapplicantIncome'] < loan_data['CoapplicantIncome'].quantile(0.98)]
    loan_data = loan_data[loan_data['LoanAmount'] < loan_data['LoanAmount'].quantile(0.97)]
    
    return loan_data
# loan_data = loan_data[loan_data['ApplicantIncome'] < loan_data['ApplicantIncome'].quantile(0.97)]
# loan_data = loan_data[loan_data['CoapplicantIncome'] < loan_data['CoapplicantIncome'].quantile(0.98)]
# loan_data = loan_data[loan_data['LoanAmount'] < loan_data['LoanAmount'].quantile(0.97)]
# loan_data = loan_data[(loan_data['ApplicantIncome'] < loan_data['ApplicantIncome'].quantile(0.95)) & (loan_data['CoapplicantIncome'] < loan_data['CoapplicantIncome'].quantile(0.95)) & (loan_data['LoanAmount'] < loan_data['LoanAmount'].quantile(0.95))]
# loan_data.shape



