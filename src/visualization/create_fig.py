import plotly.express as ex
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.feature_selection import mutual_info_regression, chi2
from sklearn.ensemble import RandomForestRegressor


def grouped_hist(df, x, heu, kde=False, separate_cols=False):
    if kde:
        data_frame = df.copy()[[x, heu]].dropna()
        hist_data = [data_frame[data_frame[heu] == u_val][x] for u_val in data_frame[heu].unique()]
        g = ff.create_distplot(hist_data=hist_data,
                               group_labels=list(data_frame[heu].unique()),
                               show_hist=False)

        return g.show()

    if separate_cols:
        g = ex.histogram(data_frame=df,
                         x=x,
                         facet_col=heu)

        return g.show()

    g = ex.histogram(data_frame=df,
                     x=x,
                     color=heu)

    return g.show()


def feature_importance(X, y, method='mut_info'):
    if method == 'model_selection':
        base_estimator = RandomForestRegressor(min_samples_split=50, min_samples_leaf=25, n_jobs=-1)
        base_estimator.fit(X, y)
        importance = base_estimator.feature_importances_

    if method == 'mut_info':
        pass

    return
