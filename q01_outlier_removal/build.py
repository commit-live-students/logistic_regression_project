# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(loan_data):
    qv=loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']].quantile(0.95)
#     return loan_data[((loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']]<=qv).all(axis=1))]
#     return loan_data[(loan_data['ApplicantIncome']<qv[0])& (loan_data['CoapplicantIncome']<qv[1])&(loan_data['LoanAmount']<qv[2])]
    imp_mean = Imputer(missing_values='NaN',strategy='mean')
    imp_mean.fit(loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']])
    loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']] = imp_mean.transform(loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']])
    
    
    return loan_data[(loan_data['ApplicantIncome']<=loan_data['ApplicantIncome'].quantile(0.95))& (loan_data['CoapplicantIncome']<=loan_data['CoapplicantIncome'].quantile(0.95))& (loan_data['LoanAmount']<=loan_data['LoanAmount'].quantile(0.95))]
    
    
    
    
c=outlier_removal(loan_data)
c.shape



