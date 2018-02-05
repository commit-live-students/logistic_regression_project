# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal (data):
    df = data.copy()
    #num_columns =df.select_dtypes(include=['float64','int64'])
    num_columns =df[:][['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
    quantile_95= num_columns.quantile(0.95)

    for colname in num_columns:
        quantile =  quantile_95[ colname ]
        #print colname, ":", quantile
        df=df.drop(df[df[colname]>quantile].index)
    return df

#print outlier_removal (loan_data).shape
