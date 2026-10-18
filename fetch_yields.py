"""
Daily job: fetch Treasury par yields from FRED and store them in the database.

Locally:   python fetch_yields.py                         (writes to data_cache/yields.db)
Heroku:    heroku run python fetch_yields.py -a <app>     (writes to Heroku Postgres)
           and scheduled daily with Heroku Scheduler
"""

import pandas as pd

import db
from yield_data import get_fred_data


def get_treasury_data():
    treasury_tickers = [
        'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30'
    ]
    treasury_yields = get_fred_data(treasury_tickers[0])
    for ticker in treasury_tickers[1:]:
        treasury_yields = treasury_yields.join(get_fred_data(ticker), on='date', how='inner')
    treasury_yields.columns = [str(x[3:]) for x in treasury_yields.columns]
    treasury_yields = treasury_yields.apply(pd.to_numeric, errors='coerce')
    return treasury_yields.dropna()


def main():
    # Any FRED or database error is raised, so a failed run leaves the previous data in place
    yields = get_treasury_data()
    db.save_yields(yields)
    print(f'Saved {len(yields)} rows ({yields.index.min():%Y-%m-%d} to {yields.index.max():%Y-%m-%d})')


if __name__ == '__main__':
    main()
