import pandas as pd

from unittest import TestCase
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from ..build import data_cleaning
from inspect import getfullargspec
loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)
Return_val = X_train.isnull().values.any()
Return_val1 = X_test.isnull().values.any()
class TestData_cleaning(TestCase):
    def test_data_cleaning_arguments(self):

        # Input parameters tests
        args = getfullargspec(data_cleaning)
        self.assertEqual(len(args[0]), 1, "Expected arguments %d, Given %d" % (1, len(args[0])))
    def test_data_cleaning_defaults(self):
        args=getfullargspec(data_cleaning)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        

        # Return data types
    def test_data_cleaning_return_type_X_test(self):
        self.assertIsInstance(X_test, pd.core.frame.DataFrame,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(X_test)))
    def test_data_cleaning_return_type_X_train(self):
        self.assertIsInstance(X_train, pd.core.frame.DataFrame,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(X_train)))
    def test_data_cleaning_return_type_X(self):
        self.assertIsInstance(X, pd.core.frame.DataFrame,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(X)))
    def test_data_cleaning_return_type_y(self):
        self.assertIsInstance(y, pd.core.series.Series,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(y)))
    def test_data_cleaning_return_type_y_train(self):
        self.assertIsInstance(y_train, pd.core.series.Series,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(y_train)))
    def test_data_cleaning_return_type_y_test(self):
        self.assertIsInstance(y_test, pd.core.series.Series,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(y_test)))

        # Return value tests
       
    def test_data_cleaning_return_values_null(self):
        self.assertEqual(Return_val, False, "Return value contains NaN values")
    def test_data_cleaning_return_values_null_1(self):
        self.assertEqual(Return_val1, False, "Return value contains NaN values")