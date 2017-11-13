# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:

def outlier_removal(dataset):
    a = dataset.quantile(q=0.95,interpolation = 'lower')
    Q_ApplicantIncome = a['ApplicantIncome']
    Q_CoapplicantIncome = a['CoapplicantIncome']
    Q_LoanAmount = a['LoanAmount']
    df1 = dataset.drop(dataset[(dataset['LoanAmount']>Q_LoanAmount)].index)
    df2 = df1.drop(df1[(df1['ApplicantIncome']>Q_ApplicantIncome)].index)
    df3 = df2.drop(df2[ (df2['CoapplicantIncome']>Q_CoapplicantIncome)].index)
    return df3
