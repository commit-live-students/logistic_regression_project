# Default Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
from greyatomlib.logistic_regression_project.q02_data_cleaning_all_2.build import data_cleaning_2
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)
X_train, X_test, y_train, y_test = data_cleaning_2(X_train, X_test, y_train, y_test)


# Write your solution code here:
# %load q03_logistic_regression/build.py
# Default Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q02_data_cleaning_all_2.build import data_cleaning_2

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)
X_train, X_test, y_train, y_test = data_cleaning_2(X_train, X_test, y_train, y_test)


# Write your solution code here:
def logistic_regression(X_train,X_test,y_train,y_test):
    #scale the numeric columns
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    scaler = StandardScaler()
    scaler.fit(X_train[num_features])
    X_train.loc[:,num_features]= scaler.transform(X_train[num_features])
    X_test.loc[:,num_features]= scaler.transform(X_test[num_features])

    #Build the logistic regression model with random_state=9.
    clf = LogisticRegression( random_state=9)


    #Fit that model on the test part.
    clf.fit(X_train, y_train)
    y_predict= clf.predict(X_test)

    #Gives the confusion matrix as evaluation metric of how good fit model is.
    return confusion_matrix(y_pred=y_predict, y_true=y_test)

#print (logistic_regression(X_train,X_test,y_train,y_test))
