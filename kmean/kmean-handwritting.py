from __future__ import print_function

import numpy as np
# from sklearn import *
import pathlib as pathlib
from sklearn import datasets
from sklearn.datasets import fetch_openml
# from sklearn.datasets import fetch_mldata

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

data_dir = str(pathlib.Path(__file__).parent.parent.absolute())+ '/data' # path to your data folder
# mnist = fetch_mldata('MNIST original', data_home=data_dir)
print('...')
mnist = fetch_openml('mnist_784', data_home=data_dir, as_frame=False)
# mnist = datasets.fetch_openml(cache=True, data_home=data_dir, as_frame=False)
print("Shape of minst data:", mnist.data.shape)
# print (data_dir)


K = 10 # number of clusters
N = 10000
X = mnist.data[np.random.choice(mnist.data.shape[0], N)]
kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)

print(kmeans.cluster_centers_)
print(len(pred_label))