import unittest
from DLMDSPWP01_programming_with_python_NielsHumbeck_32003045 import create_database_dataframe
from DLMDSPWP01_programming_with_python_NielsHumbeck_32003045 import get_ideal_functions
import pandas as pd

class test_exam(unittest.TestCase):
    def setUp(self):
        self.df_test, self.df_train, self.df_ideal = create_database_dataframe()
        self.list_idealfct_dict, self.mapping_criterion_dict = get_ideal_functions(self.df_train, self.df_ideal)

    def test_create_database_dataframe(self):
        self.assertIs(type(self.df_test), pd.DataFrame)
        self.assertIs(type(self.df_train), pd.DataFrame)
        self.assertIs(type(self.df_ideal), pd.DataFrame)
        self.assertIsNotNone(self.df_train)
        self.assertIsNotNone(self.df_ideal)
        self.assertIsNotNone(self.df_test)

    def test_get_ideal_functions(self):
        self.assertIs(type(self.df_train), pd.DataFrame)
        self.assertIs(type(self.df_ideal), pd.DataFrame)
        self.assertIs(type(self.list_idealfct_dict),dict)
        self.assertNotEqual(self.list_idealfct_dict,{})
        self.assertIs(type(self.mapping_criterion_dict),dict)
        self.assertNotEqual(self.mapping_criterion_dict,{})



if __name__ == '__main__':
    unittest.main()
