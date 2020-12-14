import math
import os

import pandas as pd

from preprocess import shape, init_path

raw_taxi_data_dir = 'I:\\NYC\\Taxi\\taxi'
taxi_data_result_path = os.path.join(init_path,'taxi_concate.h5')

max_lat = 40.8
min_lat = 40.67
max_lng = -73.92
min_lng = -74.02

yellow_cols = (
    'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude')
yellow_rename = {'tpep_pickup_datetime': 'pickup_datetime', 'tpep_dropoff_datetime': 'dropoff_datetime'}
yellow_dates = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
green_cols = (
    'lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Pickup_longitude', 'Pickup_latitude',
    'Dropoff_longitude', 'Dropoff_latitude')
green_rename = {'lpep_pickup_datetime': 'pickup_datetime', 'Lpep_dropoff_datetime': 'dropoff_datetime',
                'Pickup_longitude': 'pickup_longitude', 'Pickup_latitude': 'pickup_latitude',
                'Dropoff_longitude': 'dropoff_longitude', 'Dropoff_latitude': 'dropoff_latitude'}
green_dates = ['lpep_pickup_datetime', 'Lpep_dropoff_datetime']

# 2016 NYC Holidays:
#   January 01, January 18,
#   February 12, February 15,
#   May 08, May 30, June 19
# nyc_holidays = ["2016-01-01", "2016-01-18", "2016-02-12", "2016-02-15", "2016-05-08", "2016-05-30", "2016-06-19"]


def construct_taxi_sequence():
    """
    construct taxi sequence from raw data, filter out holidays and weekends,
    change [pickup_(latitude, longitude), dropoff_(latitude, longitude)] to (pickup_unit_no, dropoff_unit_no)
    :return: pd.DataFrame, with [pickup_datetime, dropoff_datetime, pickup_unit_no, dropoff_unit_no] as columns
    """
    print("Begin Taxi Constructing.")

    raw_data = pd.DataFrame()

    for name in os.listdir(raw_taxi_data_dir):

        print("Begin Constructing {}.".format(name))

        if name.startswith('yellow'):
            other = pd.read_csv(os.path.join(raw_taxi_data_dir, name), header=0, usecols=yellow_cols,
                                parse_dates=yellow_dates, infer_datetime_format=True)
            other = other.rename(columns=yellow_rename)
        elif name.startswith('green'):
            other = pd.read_csv(os.path.join(raw_taxi_data_dir, name), header=0, usecols=green_cols,
                                parse_dates=green_dates, infer_datetime_format=True)
            other = other.rename(columns=green_rename)
        else:
            continue

        # 筛掉不在纽约地区的数据；筛掉节假期和周末的数据。
        other = other[(min_lat <= other['pickup_latitude']) & (other['pickup_latitude'] <= max_lat) &
                      (min_lat <= other['dropoff_latitude']) & (other['dropoff_latitude'] <= max_lat) &
                      (min_lng <= other['pickup_longitude']) & (other['pickup_longitude'] <= max_lng) &
                      (min_lng <= other['dropoff_longitude']) & (other['dropoff_longitude'] <= max_lng)
                      # &
                      # (((~other['pickup_datet00ime'].dt.strftime('%Y-%m-%d').isin(nyc_holidays)) &
                      #   (other['pickup_datetime'].dt.weekday < 5)) |
                      #  ((~other['dropoff_datetime'].dt.strftime('%Y-%m-%d').isin(nyc_holidays)) &
                      #   (other['dropoff_datetime'].dt.weekday < 5))) &
                      # (other['pickup_datetime'].dt.year == 2016) & (other['dropoff_datetime'].dt.year == 2016)
        ]
        if not other.empty:
            def calculate_unit_no(lat: float, lng: float):
                rows, cols = shape
                if max_lat > lat >= min_lat and max_lng > lng >= min_lng:
                    lat_n = math.floor((lat - min_lat) / (max_lat - min_lat) * rows)
                    lng_n = math.floor((lng - min_lng) / (max_lng - min_lng) * cols)
                    return lat_n * cols + lng_n
                else:
                    return -1

            other['pickup_unit_no'] = other.apply(
                lambda x: calculate_unit_no(x['pickup_latitude'], x['pickup_longitude']), axis=1)
            other['dropoff_unit_no'] = other.apply(
                lambda x: calculate_unit_no(x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
            other = other.drop(columns=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'])

            raw_data = pd.concat([raw_data, other], axis=0, ignore_index=True)

        print("Constructing {} finished.".format(name))

    raw_data.to_hdf(taxi_data_result_path, key='taxi', mode='w')
    # raw_data.to_csv(data_result_path.replace('h5', 'csv'))

    print("Taxi Constructing finished.")


if __name__ == '__main__':
    construct_taxi_sequence()
