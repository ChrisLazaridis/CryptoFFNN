from binance.client import Client
import pandas as pd

# Initialize Binance client
api_key = 'Lr5PZeEORgdjAbbHgLNqxoFC744FVVSAkEOhOVpyqPFr3RtP2ANZsnknGxWEG9AI'
api_secret = 'ky9oLgksmW6NGdqBLW14EihLmsRMJb5TG9RGvzMWIaimfh283vmw4oswuONLDREL'
client = Client(api_key, api_secret)

symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'EOSUSDT', 'XLMUSDT', 'TRXUSDT',
           'BNBUSDT', 'LINKUSDT', 'DOTUSDT', 'FILUSDT', 'DASHUSDT', 'NEOUSDT', 'WAVESUSDT',
           'ZRXUSDT', 'XMRUSDT', 'ETCUSDT', 'XTZUSDT', 'ALGOUSDT', ]
api_key = 'Lr5PZeEORgdjAbbHgLNqxoFC744FVVSAkEOhOVpyqPFr3RtP2ANZsnknGxWEG9AI'
api_secret = 'ky9oLgksmW6NGdqBLW14EihLmsRMJb5TG9RGvzMWIaimfh283vmw4oswuONLDREL'
start_date = '2019-12-31'
end_date = '2024-03-30'



def fetch_historical_data(symbol, start_date, end_date):
    interval = '5m'  # Set interval to 5 minutes
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df
