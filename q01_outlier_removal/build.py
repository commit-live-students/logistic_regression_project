# %load q01_outlier_removal/build.py
# Default imports
import numpy as np
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    data = data[np.invert((data['ApplicantIncome'] > data['ApplicantIncome'].quantile(0.95)) | 
                (data['CoapplicantIncome'] > data['CoapplicantIncome'].quantile(0.95)) | 
                (data['LoanAmount'] > data['LoanAmount'].quantile(0.95)))]
    return data    
    
outlier_removal(loan_data)


