# Default Imports
from unittest import TestCase
import pandas as pd
from ..build import outlier_removal
from inspect import getfullargspec

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
data = outlier_removal(loan_data)

class TestOutlier_removal(TestCase):
    def test_outlier_removal_arguments(self):

        # Input parameters tests
        args = getfullargspec(outlier_removal)
        self.assertEqual(len(args[0]), 1, "Expected arguments %d, Given %d" % (1, len(args[0])))
    def test_outlier_removal_defaults(self):
        args = getfullargspec(outlier_removal)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        # Return data types
    def test_outlier_removal_return_instance(self):
        
        self.assertIsInstance(data, pd.core.frame.DataFrame,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(data)))

        # Return value tests
    def test_outlier_removal_return_shape(self):
        self.assertEqual(data.shape, (545, 12), "Return value shape does not match expected value")
