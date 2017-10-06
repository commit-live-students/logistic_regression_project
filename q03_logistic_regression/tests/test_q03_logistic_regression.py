# Default Imports
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from unittest import TestCase
import pandas as pd
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_oulier_removal.build import outlier_removal
from q03_logistic_regression.build import logistic_regression

class TestLogistic_regression(TestCase):
    def test_logistic_regression(self):
        loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
        loan_data = loan_data.drop('Loan_ID', 1)
        loan_data = outlier_removal(loan_data)
        X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)
        cm = logistic_regression(X_train, X_test, y_train, y_test)
        self.assertTrue(cm.max() == 101)
