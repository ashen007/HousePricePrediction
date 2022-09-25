import pandas as pd
import numpy as np


def read_train_data(path):
    """
    read train csv
    :return:
    """
    data = pd.read_csv(path, index_col='Id')
    data.columns = data.columns.str.lower()
    target = data['saleprice']
    features = data.drop('saleprice', axis=1)

    return features, target


def read_test_data(path):
    """
    read test csv
    :param path:
    :return:
    """
    data = pd.read_csv(path, index_col='Id')
    data.columns = data.columns.str.lower()

    return data


def fill_nan_in_num(features):
    """fill nan values in numeric features"""
    impute = features.groupby(by='lotconfig')['lotfrontage'].mean().to_dict()
    replace = features['lotconfig'].map(impute)

    features['lotfrontage'] = np.where(features['lotfrontage'].isna(),
                                       replace,
                                       features['lotfrontage'])
    features['garageyrblt'] = features['garageyrblt'].fillna(value=-1)
    features['masvnrarea'] = features['masvnrarea'].fillna(value=0)


def fill_nan_in_cat(features):
    """fill nan values in categorical features"""
    cols = ['poolqc', 'miscfeature', 'alley', 'fence', 'fireplacequ', 'garagetype', 'garagequal',
            'garagecond', 'garagefinish', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2',
            'bsmtqual', 'bsmtcond']

    features[cols] = features[cols].fillna(value='NO')
    features['masvnrtype'] = features['masvnrtype'].fillna(value='None')
    features['electrical'] = features['electrical'].fillna(method='ffill')


def load_train_data(path):
    """load cleaned data"""
    feature, target = read_train_data(path)
    fill_nan_in_num(feature)
    fill_nan_in_cat(feature)
    return feature, target


def load_test_data(path):
    """load test data"""
    feature = read_test_data(path)
    fill_nan_in_num(feature)
    fill_nan_in_cat(feature)

    num_cols = feature.select_dtypes(include=[int, float]).columns
    cat_cols = feature.select_dtypes(include=object).columns

    feature[cat_cols] = feature[cat_cols].fillna(value='NO')
    feature[num_cols] = feature[num_cols].fillna(value=0)

    return feature
