import numpy as np
import pandas as pd
from feature_engine.imputation import EndTailImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer


class InfraredMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, anchor_map):
        self.impute_with = None
        self.anchor_column_map = anchor_map

    @staticmethod
    def get_anchor_median_map(df, x, anchor):
        try:
            return df.groupby(anchor)[x].median()
        except:
            return {}

    def fit(self, x, y=None):
        self.impute_with = {}

        if isinstance(x, pd.DataFrame):
            if isinstance(self.anchor_column_map, dict):
                for c, a in self.anchor_column_map.items():
                    self.impute_with[c] = x.groupby(by=a)[c].median().to_dict()
            elif isinstance(self.anchor_column_map, str):
                for c in x.columns:
                    self.impute_with[c] = x.groupby(by=self.anchor_column_map)[c].median().to_dict()
            else:
                AttributeError('anchor_map need to be either string or dictionary.')
        else:
            raise AttributeError('x need to be a DataFrame.')

        return self

    def transform(self, x, y=None):
        df = x.copy()

        for c, m in self.impute_with.items():
            for k, v in m.items():
                df[c] = np.where((df[c].isna()) & (df[self.anchor_column_map[c]] == k), v, df[c])

        return df


def imputation_pipeline():
    missing_cat_columns = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond',
                           'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'FireplaceQu', 'GarageType', 'GarageFinish',
                           'GarageQual', 'GarageCond', 'PoolQC',
                           'Fence', 'MiscFeature']
    missing_num_columns = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
    imputing_cols = [*missing_cat_columns,
                     'Electrical', 'LotFrontage',
                     'MSZoning', 'MasVnrArea',
                     'MasVnrType', 'GarageYrBlt']

    categorical_impute_pipeline_1 = Pipeline([
        ('categorical_features', FunctionTransformer(lambda df: df[missing_cat_columns])),
        ('mnar_impute', SimpleImputer(strategy='constant', fill_value='None'))
    ])

    categorical_impute_pipeline_2 = Pipeline([
        ('categorical_features', FunctionTransformer(lambda df: df[['Electrical']])),
        ('mar_impute', SimpleImputer(strategy='most_frequent'))
    ])

    numerical_impute_pipeline_1 = Pipeline([
        ('numerical_features', FunctionTransformer(lambda df: df[['LotFrontage', 'MSZoning',
                                                                  'MasVnrArea', 'MasVnrType']])),
        ('median_impute', InfraredMedianImputer({'LotFrontage': 'MSZoning',
                                                 'MasVnrArea': 'MasVnrType'}))
    ])

    numerical_impute_pipeline_2 = Pipeline([
        ('numerical_features', FunctionTransformer(lambda df: df[['GarageYrBlt']])),
        ('end_tail_impute', EndTailImputer(tail='right'))
    ])

    cat_full_pipes = Pipeline([('union', FeatureUnion([('impute_1', categorical_impute_pipeline_1),
                                                       ('impute_2', categorical_impute_pipeline_2)])),
                               ('to_df', FunctionTransformer(lambda array: pd.DataFrame(array,
                                                                                        columns=[
                                                                                            *missing_cat_columns,
                                                                                            'Electrical']))),
                               ('final_impute', FunctionTransformer(lambda df: df.fillna(method='ffill')))])

    num_full_pipes = Pipeline([('union', FeatureUnion([('impute_1', numerical_impute_pipeline_1),
                                                       ('impute_2', numerical_impute_pipeline_2)])),
                               ('to_df', FunctionTransformer(lambda array: pd.DataFrame(array,
                                                                                        columns=['LotFrontage',
                                                                                                 'MSZoning',
                                                                                                 'MasVnrArea',
                                                                                                 'MasVnrType',
                                                                                                 'GarageYrBlt']))),
                               ('final_impute', FunctionTransformer(lambda df: df.fillna(method='ffill')))])

    full_impute_pipeline = Pipeline([('concat_pipes', FeatureUnion([('cat_imputer', cat_full_pipes),
                                                                    ('num_imputer', num_full_pipes)]))])
    return full_impute_pipeline
