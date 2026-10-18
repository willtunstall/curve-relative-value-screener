import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

TABLE = 'treasury_par_yields'
LOCAL_DB = Path(__file__).parent / 'data_cache' / 'yields.db'


def get_engine():
    """Heroku Postgres if DATABASE_URL is set, otherwise a local SQLite file."""
    url = os.environ.get('DATABASE_URL')
    if url:
        # Heroku uses postgres://, which SQLAlchemy rejects, and SQLAlchemy 2.1 defaults to the
        # psycopg (v3) driver. Name psycopg2 explicitly to match psycopg2-binary in requirements.txt
        url = url.split('://', 1)[1]
        url = f'postgresql+psycopg2://{url}'
    else:
        LOCAL_DB.parent.mkdir(exist_ok=True)
        url = f'sqlite:///{LOCAL_DB}'
    return create_engine(url)


def save_yields(df):
    """Replace the whole table in one transaction, so readers never see a half-written table."""
    with get_engine().begin() as conn:
        df.to_sql(TABLE, conn, if_exists='replace', index=True, index_label='date')


def load_yields():
    """Return yields in the same shape as get_treasury_data(): date index, one float column per tenor."""
    df = pd.read_sql_table(TABLE, get_engine(), index_col='date', parse_dates=['date'])
    return df.astype(float).sort_index()
