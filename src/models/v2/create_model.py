import json
import pickle
import numpy as np
from numpyencoder import NumpyEncoder
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV


class FitEstimator:
    """fitting estimators"""

    def __init__(self, estimator, random_state):
        self.best_random_est = None
        self._target = None
        self._fold_dtl = None
        self._est = estimator
        self.RANDOM_STATE = random_state
        self.para_dists = []

    def create_folds(self, k=5, shuffle=True):
        return KFold(n_splits=k, random_state=self.RANDOM_STATE, shuffle=shuffle)

    def fit(self, x, y):
        self._est.fit(x, y)

    def refit(self, x, y):
        self.best_random_est.fit(x, y)

    def predict(self, x):
        return self._est.predict(x)

    def save_model(self, file_name):
        with open(f'../model/{file_name}.sav', 'wb') as file:
            pickle.dump(self.best_random_est, file)

    @staticmethod
    def score(true_y, pred_y, method, squared=False):
        if not squared:
            return np.sqrt(method(true_y, pred_y))
        else:
            return method(true_y, pred_y)

    def fit_cv(self, x, y, scoring, test_x, test_y=None, k=5):
        folds = self.create_folds(k=k)
        self._fold_dtl = {}
        self._target = np.zeros(len(test_x))
        i = 0

        for t, v in folds.split(x, y):
            self.fit(x.iloc[t], y.iloc[t])
            train_pred = self.predict(x.iloc[t])
            valid_pred = self.predict(x.iloc[v])

            train_score = self.score(y.iloc[t], train_pred, scoring)
            valid_score = self.score(y.iloc[v], valid_pred, scoring)

            if test_y is not None:
                test_pred = self.predict(test_x)
                test_score = self.score(test_y, test_pred, scoring)

                self._fold_dtl[f'fold-{i}'] = {'train_score': train_score,
                                               'validation_score': valid_score,
                                               'test_score': test_score}

            else:
                test_pred = self.predict(test_x)
                self._target += test_pred

                self._fold_dtl[f'fold-{i}'] = {'train_score': train_score,
                                               'validation_score': valid_score}
            i += 1

    @staticmethod
    def re_arrange_para_dist(factor, best_vales, acpt_range, clip=None):
        new_param = {}

        for key, value in best_vales.items():
            if not isinstance(value, str):
                if isinstance(value, (int, np.int32, np.int64, np.int16, np.int8)):
                    boundary = value // factor
                else:
                    boundary = value / factor

                lower = value - boundary
                upper = value + boundary

                if clip is not None:
                    if key in clip.keys():
                        if lower < clip[key][0]:
                            lower = clip[key][0]

                        if upper > clip[key][1]:
                            upper = clip[key][1]

                if lower < 0:
                    if isinstance(value, (int, np.int32, np.int64, np.int16, np.int8)):
                        paras = []

                        while len(paras) != 3:
                            p = np.random.randint(0, upper, 1)[0]

                            if p not in paras:
                                paras.append(p)

                        new_param[key] = paras

                    elif isinstance(value, (float, np.float16, np.float32, np.float64)):
                        paras = []

                        while len(paras) != 3:
                            p = np.random.uniform(0, upper, 1)[0]

                            if p not in paras:
                                paras.append(p)

                        new_param[key] = paras

                else:
                    if isinstance(value, (int, np.int32, np.int64, np.int16, np.int8)):
                        paras = []

                        while len(paras) != 3:
                            if lower >= upper:
                                lower = acpt_range[key][0]
                                upper = acpt_range[key][1]

                            p = np.random.randint(lower, upper, 1)[0]

                            if p not in paras:
                                paras.append(p)

                        new_param[key] = paras

                    elif isinstance(value, (float, np.float16, np.float32, np.float64)):
                        paras = []

                        while len(paras) != 3:
                            if lower >= upper:
                                lower = acpt_range[key][0]
                                upper = acpt_range[key][1]

                            p = np.random.uniform(lower, upper, 1)[0]

                            if p not in paras:
                                paras.append(p)

                        new_param[key] = paras
            else:
                new_param[key] = value

        return new_param

    @staticmethod
    def tuner_progress(cv_result):
        train_mean = np.mean(np.sqrt(np.abs(cv_result['mean_train_score'])))
        test_mean = np.mean(np.sqrt(np.abs(cv_result['mean_test_score'])))
        train_std = np.std(np.sqrt(np.abs(cv_result['mean_train_score'])))
        test_std = np.std(np.sqrt(np.abs(cv_result['mean_test_score'])))

        print(f'mean train score: {train_mean} mean test score: {test_mean}')
        print(f'std train score: {train_std} std test score: {test_std}')
        print()

        return

    def random_search(self, param_distributions, scoring, n_iter, cv, verbose):
        return RandomizedSearchCV(estimator=self._est,
                                  param_distributions=param_distributions,
                                  scoring=scoring,
                                  n_iter=n_iter,
                                  cv=cv,
                                  return_train_score=True,
                                  n_jobs=-1,
                                  verbose=verbose)

    def grid_search(self, param_grid, scoring, cv, verbose):
        return GridSearchCV(estimator=self.best_random_est,
                            param_grid=param_grid,
                            scoring=scoring,
                            cv=cv,
                            return_train_score=True,
                            n_jobs=-1,
                            verbose=verbose)

    def fine_tune(self, x, y,
                  init_grid,
                  n_random_tunes,
                  log_name,
                  factor=2,
                  clip=None,
                  cv=None,
                  n_iter=10,
                  scoring=None):
        logs = []
        rs = self.random_search(param_distributions=init_grid,
                                scoring=scoring,
                                n_iter=n_iter,
                                cv=cv,
                                verbose=1).fit(x, y)

        current_para = rs.best_params_
        self.best_random_est = rs.best_estimator_

        logs.append({'init_fit': rs.cv_results_})

        for j in range(n_random_tunes):
            re_para_dis = self.re_arrange_para_dist(factor, current_para, clip)
            self.para_dists.append(re_para_dis)

            temp = self.random_search(param_distributions=re_para_dis,
                                      scoring=scoring,
                                      n_iter=n_iter,
                                      cv=cv,
                                      verbose=1).fit(x, y)

            current_para = temp.best_params_
            self.best_random_est = temp.best_estimator_

            print(f'{j + 1}/{n_random_tunes}')
            self.tuner_progress(temp.cv_results_)

            logs.append({f'iter{j}_fit': temp.cv_results_})

        with open(f'../data/log/{log_name}', 'w') as file:
            json.dump(logs, file, cls=NumpyEncoder)

        return self.best_random_est
