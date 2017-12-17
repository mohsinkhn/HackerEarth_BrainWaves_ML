#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:21:28 2017

@author: Mohsin
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import is_string_dtype
from sklearn.base import BaseEstimator, TransformerMixin

def get_obj_cols(df):
    """Return columns with object dtypes"""
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object':
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def convert_input(X):
    """if input not a dataframe convert it to one"""
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, list):
            X = pd.DataFrame(np.array(X))
        elif isinstance(X, (np.generic, np.ndarray)):
            X = pd.DataFrame(X)
        elif isinstance(X, csr_matrix):
            X = pd.SparseDataFrame(X)
        else:
            raise ValueError('Unexpected input type: %s' % (str(type(X))))

        #X = X.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Binary encoding for categorical variables, similar to onehot, 
    but stores categories as binary bitstrings.
    Expects cols to numerical, else throws error
    """
    def __init__(self, verbose=0, cols=None, add_to_df=True, return_df=True):
        self.return_df = return_df
        self.verbose = verbose
        self.cols = cols
        self.add_to_df = add_to_df
        self._dim = None
        self.digits_per_col = {}

    def fit(self, X, y=None, **kwargs):
        # if the input dataset isn't already a dataframe, convert it to one (using default column names)
        # first check the type
        X = convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)
        #Check if all cols are numeric and no nan's else throws error
        if np.any([is_string_dtype(X[col]) for col in self.cols]):
            raise ValueError("Input contains non-numeric data or is has nan's")

        for col in self.cols:
            self.digits_per_col[col] = self.calc_required_digits(X, col)
            
        return self


    def transform(self, X):
        """Perform the transformation to new categorical data. """
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')
        # first check the type
        X = convert_input(X)
        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim, ))

        #Check if all cols are numeric and no nan's else throws error
        if np.any([is_string_dtype(X[col]) for col in self.cols]):
            raise ValueError("Input contains non-numeric data ")
        
        X = self.binary(X, cols=self.cols)
        print(X.shape)
        if self.return_df:
            return X
        else:
            return X.values


    def binary(self, X_in, cols=None):
        """
        Binary encoding encodes the integers as binary code with one column per digit.
        """
        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values
            pass_thru = []
        else:
            pass_thru = [col for col in X.columns.values if col not in cols]

        bin_cols = []
        for col in cols:
            # get how many digits we need to represent the classes present
            digits = self.digits_per_col[col]

            # map the ordinal column into a list of these digits, of length digits
            X[col] = X[col].map(lambda x: self.col_transform(x, digits))

            for dig in range(digits):
                X[str(col) + '_%d' % (dig, )] = X[col].map(lambda r: 
                                                int(r[dig]) if r is not None else None)
                bin_cols.append(str(col) + '_%d' % (dig, ))

        if self.add_to_df:
            X = X.reindex(columns=bin_cols + pass_thru)
        else:
            X =  X.reindex(columns=bin_cols)
        return X

        
    @staticmethod
    def calc_required_digits(X, col):
        """
        figure out how many digits we need to represent the classes present
        """
        return int( np.ceil(np.log2(X[col].nunique())) )

    
    @staticmethod
    def col_transform(col, digits):
        """
        The lambda body to transform the column values
        """
        if col is None or float(col) < 0.0:
            return None
        else:
            col = format(col, "0"+str(digits)+'b')
        return col
