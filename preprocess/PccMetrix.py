import h5py
import os
from preprocess import init_path
import pandas as pd
import numpy as np

pathout = os.path.join(init_path, "all_graph.h5")
pathin = os.path.join(init_path, "bike_data.h5")
with h5py.File(pathin,'r') as f:
    bike_pick = f['bike_pick'][:]
    b_graph = pd.DataFrame(bike_pick)
    b_graph = b_graph.corr().values
    print(b_graph)

with h5py.File(os.path.join(init_path, "taxi_data.h5"),'r') as f:
    taxi_pick = f['taxi_pick'][:]
    t_graph = pd.DataFrame(taxi_pick)
    t_graph = t_graph.corr().values
    print(t_graph)
with h5py.File(pathout, 'r+') as f:
    print(111)
    f['pcc_bb'] = b_graph
    f['pcc_tt'] = t_graph


