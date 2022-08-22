import numpy as np
import pandas as pd


def encode_ordinal_features(features, mapping=None):
    """
    map numerical order to ordinal features
    :param features:
    :param mapping:
    :return:
    """
    ordinal_feat = ['exterqual', 'extercond', 'bsmtqual', 'bsmtcond',
                    'bsmtexposure', 'heatingqc', 'kitchenqual', 'fireplacequ',
                    'housestyle', 'garagequal', 'garagecond', 'poolqc']

    ordinal_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': -1,
                       'NO': -1, 'No': -1, 'Av': 3, 'Mn': 2,
                       '1Story': 10, '1.5Fin': 15, '1.5Unf': 12,
                       '2Story': 20, '2.5Fin': 25, '2.5Unf': 22,
                       'SFoyer': 21, 'SLvl': 23}

    features[ordinal_feat] = features[ordinal_feat].apply(lambda x: x.map(ordinal_mapping))
    features[['centralair', 'paveddrive']] = features[['centralair', 'paveddrive']].apply(lambda x: x.map({'Y': 1,
                                                                                                           'P': 0,
                                                                                                           'N': -1}))


def freq_encode_nominal_features(features, normalize=False):
    """
    map frequency of categories to string values in nominal features
    :param features:
    :param normalize:
    :return:
    """
    nominal_feat = ['salecondition', 'saletype', 'fence', 'miscfeature',
                    'garagetype', 'garagefinish', 'functional', 'electrical',
                    'heating', 'foundation', 'roofstyle', 'roofmatl',
                    'exterior1st', 'exterior2nd', 'masvnrtype',
                    'street', 'alley', 'lotshape', 'landcontour',
                    'bsmtfintype1', 'bsmtfintype2',
                    'utilities', 'lotconfig', 'landslope', 'neighborhood',
                    'condition1', 'condition2', 'bldgtype', 'mszoning']

    for col in nominal_feat:
        mapper = features.groupby(col)[col].apply(lambda x: x.count()).to_dict()
        features[col] = features[col].map(mapper)

    if normalize:
        features[nominal_feat] = features[nominal_feat].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
