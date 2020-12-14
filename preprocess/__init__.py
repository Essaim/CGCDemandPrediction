
shape = (1000, 1000)
init_path = "../data/nogrid"

#
# # for step 0: taxi sequence constructing
# raw_taxi_data_dir = '../data/taxi-data/raw-data'
# taxi_data_result_path = '../data/taxi-data/taxi-data.h5'  # [pickup_datetime, dropoff_datetime, pickup_unit_no, dropoff_unit_no]
#
# raw_bike_data_dir = '../data/bike-data/raw-data'
# bike_data_result_path = '../data/bike-data//bike-data.h5'  # [pickup_datetime, dropoff_datetime, pickup_station_no, dropoff_station_no]
#
# # for step 1: clustering
# _rho_threshold = 80
# _delta_threshold = 30 	
#
# cluster_result_path = '../data/taxi-data/cluster-result.h5'
#
# # for step 2: signals and graph constructing
# freq = '1H'  # 按小时分片, '30T' # 按半小时分片
#
# time_frame = 1  # 图按照每 time_frame 小时分槽，time_frame 需要被 24 整除
#
# taxi_pickup_data_path = '../data/taxi-data/pickup-data.h5'
# taxi_dropoff_data_path = '../data/taxi-data/dropoff-data.h5'
# taxi_graph_series_path = '../data/taxi-data/graph-{}.npy'.format(24 // time_frame)  # [n_time_slot, node_num, node_num]
#
# bike_pickup_data_path = '../data/bike-data/pickup-data.h5'
# bike_dropoff_data_path = '../data/bike-data/dropoff-data.h5'
# bike_graph_series_path = '../data/bike-data/graph-{}.npy'.format(24 // time_frame)  # [n_time_slot, node_num, node_num]
#
# train_percent = 0.7  # 训练数据所占的百分比，其余是测试数据
