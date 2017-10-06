# Default Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load Data
loan_data = pd.read_csv('/data/loan_prediction.csv')
# Data Splitter solution
def data_splitter(df):
    X = loan_data.iloc[:,:-1]
    y = loan_data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 9)
    return X, y, X_train, X_test, y_train, y_test

X, y, X_train, X_test, y_train, y_test = data_splitter(loan_data)
# Feature Scaling
def feature_scaling(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.fit_transform(X_test)

X_train, X_test = feature_scaling(X_train, X_test)
