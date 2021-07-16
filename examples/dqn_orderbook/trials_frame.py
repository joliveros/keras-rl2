#! /usr/bin/env python
from exchange_data.models.resnet.study_wrapper import StudyWrapper

import alog
import click
import pandas as pd

import plotly.express as px

class TrialsFrame(StudyWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pd.options.plotting.backend = "plotly"
        df = self.study.trials_dataframe().filter(items=['value', 'params_lr'])
        fig = df.plot.scatter(x='params_lr', y='value')

        fig.show()


@click.command()
@click.argument('symbol', type=str)
def main(**kwargs):
    TrialsFrame(**kwargs)


if __name__ == '__main__':
    main()
