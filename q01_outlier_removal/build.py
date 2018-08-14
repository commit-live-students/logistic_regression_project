# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

# Write your Solution here:
#loan_data.ApplicantIncome.plot(kind='box')
#loan_data.CoapplicantIncome.plot(kind='box')
#loan_data.LoanAmount.plot(kind='box')
def outlier_removal(loandata):
    numFeatures = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
    qv1 = loandata['ApplicantIncome'].quantile(q=0.965)
    qv2 = loandata['CoapplicantIncome'].quantile(q=0.965)
    qv3 = loandata['LoanAmount'].quantile(q=0.965)

    loandata = loandata[(loandata['ApplicantIncome'] < qv1) & 
                          (loandata['CoapplicantIncome'] < qv2) & 
                          (loandata['LoanAmount'] < qv3)]

    return loandata

#outlier_removal(loan_data)

