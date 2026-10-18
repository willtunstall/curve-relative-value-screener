import requests
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

def get_fred_data(ticker):
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    FRED_API_KEY = os.environ.get("FRED_API_KEY")

    response = requests.get(
        f"https://api.stlouisfed.org/fred/series/observations?series_id={ticker}&api_key={FRED_API_KEY}&file_type=json")

    if response.status_code == 200:
        data_json = response.json()
        data = pd.DataFrame(pd.Series(data_json)["observations"]).drop(columns=["realtime_start", "realtime_end"],
                                                                       axis=1)
        data["date"] = pd.to_datetime(data["date"])
        data = data.rename(columns={"value": ticker})
        data.set_index("date", inplace=True)
        return data

    else:
        raise ValueError("Could not fetch historical yield data from API")

if __name__ == "__main__":
    # treasury_tickers = [
    #     'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30'
    # ]
    # for ticker in treasury_tickers:
    #     print(get_fred_data(ticker))
    ...