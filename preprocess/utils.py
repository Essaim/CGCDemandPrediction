import os

import pandas as pd
import numpy as np


def make_data_and_graph_sequence(pickup_path, dropoff_path, data_dir, seq_len, horizon, freq="1H"):
    _pickup_signal_sequence: pd.DataFrame = pd.read_hdf(pickup_path)
    _dropoff_signal_sequence: pd.DataFrame = pd.read_hdf(dropoff_path)

    data_len = seq_len + horizon
    for idx, now in enumerate(_pickup_signal_sequence.index):
        x1 = _pickup_signal_sequence[now: now + (data_len - 1) * pd.Timedelta(freq)].stack()
        x2 = _dropoff_signal_sequence[now: now + (data_len - 1) * pd.Timedelta(freq)].stack()

        x1.index.rename(['pickup_datetime', 'cluster_no'], inplace=True)
        x2.index.rename(['pickup_datetime', 'cluster_no'], inplace=True)

        x = pd.DataFrame({"pickup": x1, "dropoff": x2}).stack().unstack(level=[1, 2])

        if len(x) == data_len:
            x.to_hdf(os.path.join(data_dir, str(now).replace(':', '-')), key='taxi')
            print("#{}: {} done.".format(idx, os.path.join(data_dir, str(now).replace(':', '-'))))


def make_external_features(weather_path: str):
    weather = pd.read_csv(weather_path,
                          usecols=['temp', 'dewPt', 'rh', 'wdir_cardinal', 'wspd', 'pressure', 'wx_phrase',
                                   'valid_time_gmt', 'feels_like'])
    weather.fillna(method='ffill', axis=0, inplace=True)
    weather['Time'] = pd.to_datetime(weather['valid_time_gmt'], unit='s') - pd.Timedelta('5H')
    weather = weather.drop(columns=['valid_time_gmt'])
    weather = weather.set_index('Time')

    weather['day_of_week'] = weather.index.dayofweek
    weather['hour'] = weather.index.hour

    # 剔除该列里的多余值
    weather['wx_phrase'] = weather['wx_phrase'].apply(lambda val: val.split('/')[0].strip())
    weather = pd.get_dummies(weather)
    weather = weather.resample('1H').max()
    weather.to_hdf(weather_path.replace('csv', 'h5'), key='weather')
