# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np
loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    App_inc_UQR = data.quantile(0.95,interpolation='nearest')[0]
    Coo_app_inc_UQR = data.quantile(0.95,interpolation='nearest')[1]
    Loan_amount_UQR = data.quantile(0.95,interpolation='nearest')[2]
    Loan_amount_term_UQR = data.quantile(0.95,interpolation='nearest')[3]

    data = data[data['ApplicantIncome']< App_inc_UQR]
    #loan_data = loan_data[loan_data['CoapplicantIncome']<= Coo_app_inc_UQR]
    data = data[data['LoanAmount'] < Loan_amount_UQR]
    #loan_data = loan_data[loan_data['Loan_Amount_Term']<= Loan_amount_term_UQR]

    return data










