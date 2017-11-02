# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(loan_data):
    df = loan_data

    q = df["ApplicantIncome"].quantile(0.95)
   # loan_data = loan_data[loan_data["ApplicantIncome"] < q]
    q1 = df["CoapplicantIncome"].quantile(0.95)
    #loan_data = loan_data[loan_data["CoapplicantIncome"] < q1]
    q2 = df["LoanAmount"].quantile(0.95)
   # df = loan_data[loan_data["LoanAmount"] < q2]

    df_out = df.drop(df[(df["ApplicantIncome"] > q)|
                        (df["LoanAmount"] > q2) |
                        (df["CoapplicantIncome"] > q1)].index)

    return df_out


# Write your Solution here:
