# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal (data):
    a = data.quantile(q=0.95,interpolation = 'lower')
    Q_ApplicantIncome = a['ApplicantIncome']
    Q_CoapplicantIncome = a['CoapplicantIncome']
    Q_LoanAmount = a['LoanAmount']
    df1 = data.drop(data[(data['ApplicantIncome']>Q_ApplicantIncome)].index)
    df2 = df1.drop(df1[(df1['CoapplicantIncome']>Q_CoapplicantIncome)].index)
    loan_data = df2.drop(df2[ (df2['LoanAmount']>Q_LoanAmount)].index)
    return loan_data
# Write your Solution here:
