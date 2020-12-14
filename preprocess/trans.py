import pandas as pd
from preprocessor import shape

max_lat = 40.8
min_lat = 40.67
max_lng = -73.92
min_lng = -74.02



cluster = pd.read_hdf("../preprocessdata/all-cluster.h5")
xxx = cluster['belong'].value_counts()
x = pd.DataFrame({'num':xxx})
x['lngnum'] = x.index % shape[1]
x['latnum'] = x.index // shape[1]
x['lng'] = (x.index % shape[1])*((max_lng-min_lng)/shape[1]) + min_lng
x['lat'] = (x.index // shape[1])*((max_lat-min_lat)/shape[0]) + min_lat

x.to_csv("../preprocessdata/cluster_centers.csv",index_label='index')
