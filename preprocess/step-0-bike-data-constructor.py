import os

import pandas as pd
from preprocess import init_path
raw_bike_data_dir = 'I:\\NYC\\CitiBike\\bike'
bike_data_result_path = os.path.join(init_path,'bike_concate.h5')

used_cols = ['starttime', 'start station id', 'stoptime', 'end station id', 'start station latitude',
             'start station longitude', 'end station latitude', 'end station longitude']


# 2018 NYC Holidays:
# nyc_holidays = ["2018-01-01", "2018-01-15", "2018-02-12", "2018-02-19", "2018-05-28", "2018-05-30"]


def construct_bike_sequence():
    """
    construct bike sequence from raw data, filter out holidays and weekends
    :return: pd.DataFrame, with ['starttime', 'start station id', 'stoptime', 'end station id'] as columns
    """
    print("Begin Bike Constructing.")

    raw_data = pd.DataFrame()

    for name in os.listdir(raw_bike_data_dir):

        print("Begin Constructing {}.".format(name))
        other = pd.read_csv(os.path.join(raw_bike_data_dir, name), header=0, usecols=used_cols,
                            parse_dates=['starttime', 'stoptime'], infer_datetime_format=True)

        # 筛掉节假期和周末的数据。
        # other = other[(((~other['starttime'].dt.strftime('%Y-%m-%d').isin(nyc_holidays)) &
        #                 (other['starttime'].dt.weekday < 5)) |
        #                ((~other['stoptime'].dt.strftime('%Y-%m-%d').isin(nyc_holidays)) &
        #                 (other['stoptime'].dt.weekday < 5)))]
        if not other.empty:
            raw_data = pd.concat([raw_data, other], axis=0, ignore_index=True)

        print("Constructing {} finished.".format(name))

    raw_data.to_hdf(bike_data_result_path, key='bike', mode='w')

    print("Bike Constructing finished.")


if __name__ == '__main__':
    construct_bike_sequence()
