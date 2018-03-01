# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    df=data.quantile(0.95)
    for i in range(0,len(df)-2):
        data=data.drop(data[(data[df.index[i]]>df[i])].index)
    return data
    
