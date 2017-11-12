# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


def outlier_removal(data):
    #print data.columns.tolist()
    q1=data["ApplicantIncome"].quantile(0.95)
    q2=data["CoapplicantIncome"].quantile(0.95)
    q3=data["LoanAmount"].quantile(0.95)
    data = data.drop(data[data["ApplicantIncome"]>q1].index)
    data = data.drop(data[data["CoapplicantIncome"]>q2].index)
    data = data.drop(data[data["LoanAmount"]>q3].index)
    return data
