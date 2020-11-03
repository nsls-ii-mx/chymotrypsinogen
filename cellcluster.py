import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

__author__ = "Thu Nguyen"
__copyright__ = "Copyright (C) 2020 Thu Nguyen"
__license__ = "LGPL 2.1"
__version__ = "1.0"

def cell_plot(input_csv, cluster=True):
    df = pd.read_csv(input_csv, header=None, usecols=[0,2,3], names=['a', 'c', 'cluster'])
    if not cluster:
        plt.scatter(df['a'],df['c'],c='b')
        plt.show()
    else:
        df_0 = df[df['cluster']==1]
        df_1 = df[df['cluster']==2]
        df_2 = df[df['cluster'] == 3]
        df_3 = df[df['cluster'] == 4]
        df_4 = df[df['cluster'] == 5]
        plt.scatter(df_0['a'], df_0['c'], c='g')
        plt.scatter(df_1['a'], df_1['c'], c='r')
        plt.scatter(df_2['a'], df_2['c'], c='c')
        plt.scatter(df_3['a'], df_3['c'], c='m')
        plt.scatter(df_4['a'], df_4['c'], c='y')
        plt.show()

if __name__ == '__main__':
    #input csv file which contain 'a, b, c, cluster_number'
    cell_plot(sys.argv[1])
