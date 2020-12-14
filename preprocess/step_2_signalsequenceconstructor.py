import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import *
import h5py
import os
from preprocess import init_path

max_lat = 40.8
min_lat = 40.67
max_lng = -73.92
min_lng = -74.02
shape = (1000, 1000)
EARTH_REDIUS = 6378.137
K = 2
bike_drop_num = 250
grid = ((max_lat - min_lat) / shape[0], (max_lng - min_lng) / shape[1])


def rad(d):
    return d * pi / 180.0


def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * asin(sqrt(pow(sin(a / 2), 2) + cos(radLat1) * cos(radLat2) * pow(sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s


def preprocess_taxi_data(taxi_data_path, cluster_result):
    taxi_data: pd.DataFrame = pd.read_hdf(taxi_data_path, key='taxi')
    cluster_result: pd.DataFrame = pd.read_hdf(cluster_result, key='taxi')
    cluster_result = cluster_result.to_dict()
    taxi_data['pickup_cluster_no'] = taxi_data.apply(
        lambda row: cluster_result['belong'][int(row['pickup_unit_no'])],
        axis=1)
    taxi_data['dropoff_cluster_no'] = taxi_data.apply(
        lambda row: cluster_result['belong'][int(row['dropoff_unit_no'])],
        axis=1)
    taxi_data.to_hdf(taxi_data_path, key='taxi')
    taxi_data = taxi_data.drop(columns=['pickup_unit_no', 'dropoff_unit_no'])
    return taxi_data


def taxi_signal_sequence_construct(_data, _node_list, key, path, freq):
    """
    construct signal sequence from data
    :return: two pd.DataFrame, with node as columns and time as indexes
    """
    count_by_pickup_time = _data.groupby([pd.Grouper(key='pickup_datetime', freq=freq), 'pickup_cluster_no']).size()
    count_by_dropoff_time = _data.groupby(
        [pd.Grouper(key='dropoff_datetime', freq=freq), 'dropoff_cluster_no']).size()

    _pickup_signal_sequence: pd.DataFrame = count_by_pickup_time.unstack(level='pickup_cluster_no', fill_value=0)
    _dropoff_signal_sequence: pd.DataFrame = count_by_dropoff_time.unstack(level='dropoff_cluster_no', fill_value=0)

    _dropoff_signal_sequence = _dropoff_signal_sequence.loc[_pickup_signal_sequence.index]

    times = np.union1d(_pickup_signal_sequence.index.values, _dropoff_signal_sequence.index.values)

    _pickup_signal_sequence = _pickup_signal_sequence.reindex(index=times, columns=_node_list, fill_value=0)
    _dropoff_signal_sequence = _dropoff_signal_sequence.reindex(index=times, columns=_node_list, fill_value=0)

    # _pickup_signal_sequence.to_hdf(pickup_path, key=key)
    # _dropoff_signal_sequence.to_hdf(dropoff_path, key=key)
    if os.path.exists(path):
        os.remove(path)
    h5 = h5py.File(path, 'w')
    h5.create_dataset(f"{key}_pick", data=_pickup_signal_sequence.values)
    h5.create_dataset(f"{key}_drop", data=_dropoff_signal_sequence.values)
    h5.close()

    return _pickup_signal_sequence, _dropoff_signal_sequence


def bike_signal_sequence_construct(_data, _node_list, key, path, freq):
    """
    construct signal sequence from data
    :return: two pd.DataFrame, with node as columns and time as indexes
    """
    count_by_pickup_time = _data.groupby([pd.Grouper(key='starttime', freq=freq), 'start station id']).size()
    count_by_dropoff_time = _data.groupby(
        [pd.Grouper(key='stoptime', freq=freq), 'end station id']).size()

    _pickup_signal_sequence: pd.DataFrame = count_by_pickup_time.unstack(level='start station id', fill_value=0)
    _dropoff_signal_sequence: pd.DataFrame = count_by_dropoff_time.unstack(level='end station id', fill_value=0)

    # _dropoff_signal_sequence = _dropoff_signal_sequence.loc[_pickup_signal_sequence.index]
    #
    # times = np.union1d(_pickup_signal_sequence.index.values, _dropoff_signal_sequence.index.values)
    #
    # _pickup_signal_sequence = _pickup_signal_sequence.reindex(index=times, columns=_node_list, fill_value=0)
    # _dropoff_signal_sequence = _dropoff_signal_sequence.reindex(index=times, columns=_node_list, fill_value=0)

    # bike_data.groupby(['end station id', 'end station latitude', 'end station longitude']).size().to_csv(xxx_path)
    # xxx = pd.read_csv(xxx_path, header=None, names=['id', 'lat', 'lng', 'num'])
    # xxx = xxx.groupby('id').mean()
    # xxx = xxx[xxx['lat']>0]
    # xxx = xxx[xxx['lng']>0]
    # xxx = xxx.drop(columns=['lat', 'lng'])


    start = _data.groupby('start station id').size()
    end = _data.groupby('end station id').size()
    start = start.reindex(_node_list, fill_value=0)
    end = end.reindex(_node_list, fill_value=0)
    total = start + end
    total = total.sort_values()
    total = total.drop(total.index[:-bike_drop_num])
    _node_list = total.index

    _pickup_signal_sequence = _pickup_signal_sequence[total.index]
    _dropoff_signal_sequence = _dropoff_signal_sequence[total.index]
    _dropoff_signal_sequence = _dropoff_signal_sequence.loc[_pickup_signal_sequence.index]

    times = np.union1d(_pickup_signal_sequence.index.values, _dropoff_signal_sequence.index.values)
    _pickup_signal_sequence = _pickup_signal_sequence.reindex(index=times, columns=_node_list, fill_value=0)
    _dropoff_signal_sequence = _dropoff_signal_sequence.reindex(index=times, columns=_node_list, fill_value=0)


    if os.path.exists(path):
        os.remove(path)
    h5 = h5py.File(path, 'w')
    h5.create_dataset(f"{key}_pick", data=_pickup_signal_sequence.values)
    h5.create_dataset(f"{key}_drop", data=_dropoff_signal_sequence.values)
    h5.close()

    return _pickup_signal_sequence, _dropoff_signal_sequence, _node_list


def graph_sequence_construct(taxi_final_data,bike_final_data,taxi_data, bike_data, taxi_node_list, bike_node_list,bike_node_list_drop, graph_path,xxx_path):
    taxi_count_transform = taxi_data.groupby(['pickup_cluster_no', 'dropoff_cluster_no']).size()
    taxi_graph = taxi_count_transform.unstack(level='pickup_cluster_no', fill_value=0)
    taxi_graph = taxi_graph.reindex(index=taxi_node_list, columns=taxi_node_list, fill_value=0)



    bike_count_transform = bike_data.groupby(['start station id', 'end station id']).size()
    bike_graph = bike_count_transform.unstack(level='start station id', fill_value=0)
    bike_graph = bike_graph.reindex(index=bike_node_list_drop, columns=bike_node_list_drop, fill_value=0)



    bike_data.groupby(['end station id', 'end station latitude', 'end station longitude']).size().to_csv(xxx_path)
    xxx = pd.read_csv(xxx_path, header=None, names=['id', 'lat', 'lng', 'num']).drop(columns=['num'])
    xxx = xxx.groupby('id').mean()
    # xxx = xxx.set_index(["id"])
    xxx = xxx.reindex(bike_node_list_drop)
    if(xxx[xxx['lat']==0.0].size !=0 ):
        print("some lat == 0.0")
    else:
        print("no lat == 0.0")


    bike_node_list = bike_node_list_drop


    graph_bb = np.zeros([len(bike_node_list), len(bike_node_list)])
    graph_tt = np.zeros([len(taxi_node_list), len(taxi_node_list)])
    graph_tb = np.zeros([len(taxi_node_list), len(bike_node_list)])
    graph_bt = np.zeros([len(bike_node_list), len(taxi_node_list)])
    for i_num, i in enumerate(bike_node_list):
        for j_num, j in enumerate(bike_node_list):
            dis = getDistance(xxx.loc[i]['lat'], xxx.loc[i]['lng'], xxx.loc[j]['lat'], xxx.loc[j]['lng'])
            # print(dis,i,j)
            graph_bb[i_num][j_num] = np.exp(-dis ** 2) if dis < K else 0
        for j_num, j in enumerate(taxi_node_list):
            dis = getDistance(xxx.loc[i]['lat'], xxx.loc[i]['lng'], min_lat + (j // shape[1] + 0.5) * grid[0],
                              min_lng + (j % shape[1] + 0.5) * grid[1])
            # print(dis, i, j)
            graph_bt[i_num][j_num] = np.exp(-dis ** 2) if dis < K else 0
    for i_num, i in enumerate(taxi_node_list):
        for j_num, j in enumerate(bike_node_list):
            dis = getDistance(xxx.loc[j]['lat'], xxx.loc[j]['lng'], min_lat + (i // shape[1] + 0.5) * grid[0],
                              min_lng + (i % shape[1] + 0.5) * grid[1])
            # print(dis, i, j)
            graph_tb[i_num][j_num] = np.exp(-dis ** 2) if dis < K else 0
        for j_num, j in enumerate(taxi_node_list):
            dis = getDistance(min_lat + (i // shape[1] + 0.5) * grid[0], min_lng + (i % shape[1] + 0.5) * grid[1],
                              min_lat + (j // shape[1] + 0.5) * grid[0], min_lng + (j % shape[1] + 0.5) * grid[1])
            # print(dis, i, j)
            graph_tt[i_num][j_num] = np.exp(-dis ** 2) if dis < K else 0


    # f = h5py.File(taxi_final_data,"r")
    # pcc_taxi_drop = pd.DataFrame(f['taxi_drop'][:])
    # pcc_taxi_pick = pd.DataFrame(f['taxi_pick'][:])
    # f.close()
    # pcc_taxi_drop = pd.DataFrame(f['taxi_drop'][:])
    # pcc_taxi_drop = pd.DataFrame(f['taxi_drop'][:])




    h5 = h5py.File(graph_path,'w')
    h5.create_dataset("trans_bb", data=bike_graph.values)
    h5.create_dataset("trans_tt", data=taxi_graph.values)
    h5.create_dataset("dis_bb", data=graph_bb)
    h5.create_dataset("dis_bt", data=graph_bt)
    h5.create_dataset("dis_tb", data=graph_tb)
    h5.create_dataset("dis_tt", data=graph_tt)

    h5.close()



def main():

    taxi_data_path = os.path.join(init_path,'taxi_concate.h5')
    bike_data_path = os.path.join(init_path,'bike_concate.h5')
    cluster_path = os.path.join(init_path,'all-cluster.h5')
    bike_final_data = os.path.join(init_path,'bike_data.h5')
    taxi_final_data = os.path.join(init_path,'taxi_data.h5')
    to_graph = os.path.join(init_path,'all_graph.h5')
    xxx_path = os.path.join(init_path,'xxx.csv')
    
    # taxi_data_path = 'data/nogrid345/taxi_concate.h5'
    # bike_data_path = 'data/nogrid345/bike_concate.h5'
    # cluster_path = 'data/nogrid345/all-cluster.h5'
    # bike_final_data = 'data/nogrid345/bike_data.h5'
    # taxi_final_data = 'data/nogrid345/taxi_data.h5'
    # to_graph = 'data/nogrid345/all_graph.h5'
    # xxx_path = 'data/nogrid345/xxx.csv'
    
    # taxi_data: pd.DataFrame = pd.read_hdf(taxi_data_path).drop(columns=['pickup_unit_no', 'dropoff_unit_no'])
    taxi_data: pd.DataFrame = preprocess_taxi_data(taxi_data_path, cluster_result=cluster_path)
    bike_data: pd.DataFrame = pd.read_hdf(bike_data_path)
    
    print("preprocess finished...................................")
    taxi_node_list = np.array(
        list(set.union(set(taxi_data['pickup_cluster_no'].unique()), set(taxi_data['dropoff_cluster_no'].unique()))))
    taxi_node_list.sort()
    taxi_signal_sequence_construct(taxi_data, taxi_node_list, key='taxi', path=taxi_final_data,
                                   freq='30min')
    print("taxi signal finished.............................................")
    
    
    
    bike_node_list = np.array(
        list(set.union(set(bike_data['start station id'].unique()), set(bike_data['end station id'].unique()))))
    
    
    bike_node_list.sort()
    _,__, bike_node_list_drop = bike_signal_sequence_construct(_data=bike_data,  _node_list=bike_node_list, key='bike', path=bike_final_data,
                                   freq='30min')
    # print(bike_node_list_drop)
    print("bike signal finished.............................................")
    graph_sequence_construct(taxi_final_data,bike_final_data,taxi_data, bike_data, taxi_node_list, bike_node_list,bike_node_list_drop, to_graph,xxx_path)


if __name__ == '__main__':
    main()