
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data_numerical = loan_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]

def outlier_removal(data):
    df = data
    qlt = df.quantile(q=0.95)
    
    df = df.drop(df[(df['ApplicantIncome']>qlt[0])].index)
    df = df.drop(df[(df['CoapplicantIncome']>qlt[1])].index)
    df = df.drop(df[(df['LoanAmount']>qlt[2])].index)
    
    return df

outlier_removal(loan_data)

    
    
    


