import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pathlib as pathlib
import numpy as np
from sklearn.cluster import KMeans

data_dir = str(pathlib.Path(__file__).parent.parent.absolute())+ '/data' # path to your data folder
img = mpimg.imread(data_dir + '/girl3.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()

X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))


for K in [2, 5, 10, 15, 20]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img4 = np.zeros_like(X)
    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()