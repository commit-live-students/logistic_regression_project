# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
import numpy as np

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    df1=data.quantile(0.95)             #df1=Series
    for i in range(0,len(df1)-2):       #len(df1)-2 = 'taking only numeric values and dynamic not static values  
        data=data.drop(data[(data[df1.index[i]]>df1[i])].index)
    return data

df1=loan_data.quantile(0.95)
df1.index[0]
df1[0]
df1




