# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
np.random.seed(9)

# Write your solution here :

def outlier_removal  (loan_data):
    quan_ai=loan_data['ApplicantIncome'].quantile(.95)
    quan_ci=loan_data['CoapplicantIncome'].quantile(.95)
    quan_ai=loan_data['LoanAmount'].quantile(.95)

    loan_data=loan_data.drop(loan_data[(loan_data['ApplicantIncome']>loan_data['ApplicantIncome'].quantile(.95))|(loan_data['CoapplicantIncome']>loan_data['CoapplicantIncome'].quantile(.95))|(loan_data['LoanAmount']>loan_data['LoanAmount'].quantile(.95))].index)

    return loan_data

#outlier_removal  (loan_data)
