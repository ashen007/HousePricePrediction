import numpy as np
import pandas as pd

from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.decomposition import PCA


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


def derived_features(features):
    """
    create features from existing features
    :param features:
    :return:
    """
    features['propage'] = features['yrsold'] - features['yearbuilt']
    features['modage'] = features['yrsold'] - features['yearremodadd']
    features['timetomod'] = features['yearremodadd'] - features['yearbuilt']

    def quarter(x):
        if x <= 4:
            return 1
        elif 4 < x <= 8:
            return 2
        elif x > 8:
            return 3

    features['quarter'] = features['mosold'].apply(quarter)
    features['totfhbsmntarea'] = features['bsmtfinsf1'] + features['bsmtfinsf2']
    features['totbathbsmnt'] = features['bsmtfullbath'] + features['bsmthalfbath']
    features['totbathbsabv'] = features['fullbath'] + features['halfbath']

    features['drivfeat1'] = features['overallqual'] * features['overallcond']
    features['drivfeat2'] = features['bsmtqual'] * features['bsmtcond']
    features['drivfeat3'] = features['exterqual'] * features['extercond']
    features['drivfeat4'] = np.sum(features[['wooddecksf', 'openporchsf', 'enclosedporch', '3ssnporch',
                                             'screenporch']], axis=1)
    features['drivfeat5'] = features.groupby('masvnrtype')['masvnrarea'].transform('mean')
    features['drivfeat6'] = features.groupby('garagetype')['garagearea'].transform('mean')

    features[['grlivarea', 'garagearea', 'lotarea',
              '1stflrsf', '2ndflrsf']] = np.sqrt(features[['grlivarea', 'garagearea', 'lotarea',
                                                           '1stflrsf', '2ndflrsf']])

    features['pcafeat1'] = features.groupby('bsmtfintype2')['bsmtfinsf2'].transform('mean')
    features['pcafeat2'] = features['fireplaces'] * features['fireplacequ']
    features['pcafeat3'] = features['exterior1st'] * features['exterior2nd']
    features['pcafeat4'] = features['garageyrblt'] * features['garagequal']


def kmeans(features):
    pass


def apply_pca(features, standardize=True):
    """apply principal component analysis to generate synthetic features"""
    if standardize:
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    pca = PCA()
    feat_pca = pca.fit_transform(features)
    comp_names = [f'pc{i + 1}' for i in range(feat_pca.shape[1])]
    feat_pca = pd.DataFrame(feat_pca, columns=comp_names)
    loading = pd.DataFrame(pca.components_.T, columns=comp_names, index=features.columns)

    return pca, feat_pca, loading


def normalize(features):
    """
    normalize features
    :param features:
    :return:
    """
    return features.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


def quantile_transformation(features, output_distribution, n_quantiles):
    """
    transform input data using quantile to given distribution
    :param n_quantiles:
    :param output_distribution:
    :param features:
    :return:
    """
    transformer = QuantileTransformer(n_quantiles=n_quantiles,
                                      output_distribution=output_distribution).fit(features)

    return transformer


def power_transformation(features, method, standardize):
    """
    mapping features to gaussian distribution
    :param features:
    :param method:
    :param standardize:
    :return:
    """
    transformer = PowerTransformer(method=method,
                                   standardize=standardize).fit(features)

    return transformer


def mi_score(features, target, discrete_features):
    """
    calculate mutual information score
    :param target:
    :param discrete_features:
    :param features:
    :return:
    """
    mi_scores = mutual_info_regression(features, target, discrete_features=discrete_features)

    return pd.Series(mi_scores, index=features.columns).sort_values(ascending=False)
