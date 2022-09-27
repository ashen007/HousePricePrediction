import numpy as np
import pandas as pd
from feature_engine.imputation import EndTailImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

nominal_columns = ['MSZoning', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                   'LotConfig', 'Neighborhood', 'Condition1',
                   'Condition2', 'BldgType', 'HouseStyle', 'Street',
                   'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                   'MasVnrType', 'Foundation', 'Heating', 'Electrical',
                   'Functional', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish',
                   'GarageType', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

ordinal_columns = ['MSSubClass', 'LandSlope', 'OverallQual',
                   'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
                   'BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual',
                   'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
                   'PavedDrive', 'CentralAir']

continues_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                     '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea',
                     'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                     'ScreenPorch', 'PoolArea', 'MiscVal']

discrete_columns = ['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath',
                    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']

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


def imputation_pipeline():
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
        ('median_impute', InfraredMedianImputer(anchor_map={'LotFrontage': 'MSZoning',
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

    return Pipeline([('concat_pipes', FeatureUnion([('cat_imputer', cat_full_pipes),
                                                    ('num_imputer', num_full_pipes)]))])


def improvement_pipeline(dataframe):
    full_impute_pipeline = imputation_pipeline()
    dataframe[imputing_cols] = full_impute_pipeline.fit_transform(dataframe)
    order_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': -1,
                 'NO': -1, 'No': -1, 'Av': 3, 'Mn': 2, 'Reg': 0, 'IR1': 1,
                 'IR2': 2, 'IR3': 3, 'Gtl': 1, 'Mod': 2, 'Sev': 3, 'Y': 1,
                 'N': 0, 'P': -1}

    nominal_encode_pipeline = Pipeline([('select_nominal', FunctionTransformer(lambda df: df[nominal_columns])),
                                        ('one_hot', OneHotEncoder()),
                                        (
                                            'feature_selection',
                                            SelectFromModel(RandomForestRegressor(min_samples_split=20,
                                                                                  min_samples_leaf=10,
                                                                                  n_jobs=-1))),
                                        ('to_df',
                                         FunctionTransformer(
                                             lambda metrix: pd.DataFrame.sparse.from_spmatrix(metrix)))])

    ordinal_encode_pipeline = Pipeline([('select_ordinal', FunctionTransformer(lambda df: df[ordinal_columns])),
                                        ('encode', FunctionTransformer(lambda df: df.replace(order_map)))])

    continues_pipeline = Pipeline([('select_cont', FunctionTransformer(lambda df: df[continues_columns])),
                                   ('transformation', FunctionTransformer(lambda df: df.apply(np.log1p)))])

    discrete_pipeline = Pipeline([('select_discrete', FunctionTransformer(lambda df: df[discrete_columns]))])

    return


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
