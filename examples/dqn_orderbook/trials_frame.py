#! /usr/bin/env python
from exchange_data.models.resnet.study_wrapper import StudyWrapper

import alog
import click


class TrialsFrame(StudyWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        alog.info(self.study.trials_dataframe())


@click.command()
@click.argument('symbol', type=str)
def main(**kwargs):
    TrialsFrame(**kwargs)


if __name__ == '__main__':
    main()
