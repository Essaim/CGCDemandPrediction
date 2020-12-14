import pandas as pd
import numpy as np
import os

import preprocess.kdtree as kdt
from preprocess import shape,init_path

_delta_threshold = 10    #区域之间距离，相差多少个格子
_rho_threshold = 2000    #单个区域的密度
cluster_result_path = os.path.join(init_path,'all-cluster.h5')
taxi_data_result_path = os.path.join(init_path,'taxi_concate.h5')





class DensityPeakCluster:
    rows, cols = shape

    def __init__(self, rho: pd.Series):
        """
        load data from custom grid data
        :param rho: input data,
                a pd.Series, with indexes as unit_nos and values as local density
        """
        self.data = pd.DataFrame({'rho': rho}).sort_values(by='rho', ascending=False)
        self.data['index'] = self.data.index
        self.__calculate_delta_nneigh()

    def __calculate_delta_nneigh(self):
        """
        calculate delta from rho, which is the minimal distance from a point to another point that has a larger rho
        :return: nothing
        """
        print("Calculating min distance BEGIN")
        print('node num is {}'.format(len(self.data['rho'])))
        cols = self.cols

        data = list(map(lambda x: [x // cols, x % cols], self.data.index.values))

        tree, idx, dist = kdt.create(dimensions=2), [data[0][0] * cols + data[0][1]], [float('inf')]

        tree.add(data[0])
        for item in data[1:]:
            i, d = tree.search_nn(item)
            i = i.data
            idx.append(i[0] * cols + i[1])
            dist.append(d)
            tree.add(item)

        self.data['nneigh'] = idx
        self.data['delta'] = np.sqrt(dist)
        print("Calculating min distance END")

    def clustering(self, rho_threshold, delta_threshold):
        """
        choosing cluster centers and calculating the cluster every points belong
        :param rho_threshold: choosing points whose rho larger than or equal to the threshold
        :param delta_threshold: choosing points whose delta larger than or equal to the threshold
        :return: nothing
        """
        print("Clustering BEGIN")
        self.data['belong'] = self.data.apply(
            lambda row: row['index'] if row['rho'] >= rho_threshold and row['delta'] >= delta_threshold else -1,
            axis=1).astype(int)

        total_count = len(self.data)
        assigned_count = len(self.data[self.data['belong'] >= 0])

        if assigned_count == 0:
            raise ValueError('rho threshold or delta threshold is too big, cannot find any cluster center.')

        print("Calculating %d cluster centers." % assigned_count)

        def cal_belong(row):
            if row['belong'] >= 0:
                return row['belong']
            return self.data.at[int(row['nneigh']), 'belong']

        pre_assigned_count = 0
        while assigned_count < total_count:
            print("calculating times %d of %d" % (assigned_count, total_count))
            if assigned_count == pre_assigned_count:
                raise ValueError("Cannot assign more points to any cluster.")
            pre_assigned_count = assigned_count
            self.data['belong'] = self.data.apply(cal_belong, axis=1).astype(int)
            assigned_count = len(self.data[self.data['belong'] >= 0])
        print("Clustering END")


def clustering():
    """
    clustering from taxi data, with density peak cluster algorithm
    :return: cluster result, pd.DataFrame, with [rho, index, nneigh, delta, belong] as columns and unit_no as indexes
    """
    taxi_data: pd.DataFrame = pd.read_hdf(taxi_data_result_path, key='taxi')

    dpc = DensityPeakCluster(
        taxi_data['pickup_unit_no'].value_counts().add(taxi_data['dropoff_unit_no'].value_counts(), fill_value=0))
    dpc.clustering(_rho_threshold, _delta_threshold)
    dpc.data.to_hdf(cluster_result_path, key='taxi')
    # dpc.data.to_csv(cluster_result_path.replace('h5', 'csv'))


if __name__ == '__main__':
    clustering()
