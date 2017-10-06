# Default Imports
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from unittest import TestCase
import pandas as pd
from q01_oulier_removal.build import outlier_removal


class TestOutlier_removal(TestCase):
    def test_outlier_removal(self):
        loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
        loan_data = loan_data.drop('Loan_ID', 1)
        data = outlier_removal(loan_data)
        self.assertTrue(data.shape == (550, 12))
