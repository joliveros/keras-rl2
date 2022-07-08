#! /usr/bin/env python

from exchange_data.models.study_wrapper import StudyWrapper
from plotly.subplots import make_subplots
import alog
import click
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


class TrialsFrame(StudyWrapper):
    def __init__(self, min_value, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        pd.options.plotting.backend = "plotly"
        df = self.study.trials_dataframe()

        df = df.drop(['datetime_start', 'datetime_complete', 'state',  'user_attrs_params', 'user_attrs_tuned'], axis=1)

        df = df[df['value'] >= self.min_value]

        pd.set_option('display.max_rows', len(df) + 1)
        alog.info(df)

        params = [col for col in df.columns if 'params_' in col]

        fig = make_subplots(rows=1, cols=len(params))

        col_ix = 1
        for param in params:
            fig.add_trace(go.Scatter(
                mode="markers",
                x=df[param].to_numpy(),
                y=df.value.to_numpy(),
                name=param
            ), row=1, col=col_ix)

            col_ix += 1

        fig.show()


@click.command()
@click.argument('symbol', type=str)
@click.option('--min-value', default=0.9, type=float)
def main(**kwargs):
    TrialsFrame(**kwargs)


if __name__ == '__main__':
    main()
