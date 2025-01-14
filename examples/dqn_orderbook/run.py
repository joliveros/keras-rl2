#! /usr/bin/env python

import click
import tgym.envs

from examples.dqn_orderbook.symbol_tuner import SymbolTuner


class SymbolAgentTuner(SymbolTuner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@click.command()
@click.argument('symbol', type=str)
@click.option('--cache', is_flag=True)
@click.option('--clear-dir', is_flag=True)
@click.option('--database-name', default='binance_futures', type=str)
@click.option('--depth', '-d', default=48, type=int)
@click.option('--env-name', default='orderbook-frame-env-v0', type=str)
@click.option('--export-best', '-e', is_flag=True)
@click.option('--group-by', '-g', default='30s', type=str)
@click.option('--interval', '-i', default='30m', type=str)
@click.option('--interval2', default='30m', type=str)
@click.option('--leverage', default=1.0, type=float)
@click.option('--max-loss', default=-0.999, type=float)
@click.option('--max-negative-pnl', default=-0.0587896, type=float)
@click.option('--max-summary', default=30, type=int)
@click.option('--memory', '-m', default=0, type=int)
@click.option('--min-capital', default=1.0, type=float)
@click.option('--min-change', default=0.001, type=float)
@click.option('--nb-steps', default=0, type=int)
@click.option('--nb-steps-2', default=0, type=int)
@click.option('--num-locks', '-n', default=0, type=int)
@click.option('--offset-interval', default='2h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--retry', '-R', is_flag=True)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--sequence-length', '-l', default=36, type=int)
@click.option('--session-limit', '-s', default=None, type=int)
@click.option('--summary-interval', default=4, type=int)
@click.option('--test-interval', default='2h', type=str)
@click.option('--trading-fee', default=4e-4, type=float)
@click.option('--window-size', '-w', default='2m', type=str)
@click.option('--tune', is_flag=True)
def main(**kwargs):
    SymbolAgentTuner(**kwargs)


if __name__ == '__main__':
    main()
