# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


def outlier_removal(df):


    a=loan_data.ApplicantIncome.quantile(0.95)
    b=loan_data.CoapplicantIncome.quantile(0.95)
    c=loan_data.LoanAmount.quantile(0.95)

    print (a,b,c)

    #a2 = loan_data.drop(loan_data[(loan_data['ApplicationIncome']>a) & (loan_data['CoapplicantIncome'] > b) & (loan_data['LoanAmount'] > c)].index)
    loan_data1 = loan_data[(loan_data.ApplicantIncome <= a) & (loan_data.CoapplicantIncome <= b) & (loan_data.LoanAmount <= c)]

    loan_data2 = loan_data[(loan_data.ApplicantIncome <= 15000) & (loan_data.CoapplicantIncome <= 5700) & (loan_data.LoanAmount <= 550)]

    #df = df.drop(df[(df['GrLivArea']>3000) & (df['GrLivArea']<6000)].index)

    return loan_data2
