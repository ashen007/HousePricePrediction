import plotly.express as ex
import plotly.graph_objects as go
import plotly.figure_factory as ff


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
