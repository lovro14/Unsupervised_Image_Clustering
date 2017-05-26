from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from bhtsne import tsne


class ClusterAlgorithm(object):

    def __init__(self, model_name, image_names, image_embeddings, c):
        self.model_name = model_name
        self.image_names = image_names
        self.image_embeddings = image_embeddings
        self.c = c

    def fit(self):
        # dimensionality reduction using Barnes-Hut implementation of t-SNE
        print("SHAPE: ",np.array(self.image_embeddings, dtype=np.float64).shape)
        data_2d = tsne(np.array(self.image_embeddings, dtype=np.float64))

        # create model
        model = self._cluster_model(self.model_name, self.c)
        model.fit(self.image_embeddings)
        labels = model.labels_

        # visualization evaluation - 2D Scatter plot of data
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
        plt.show()

        clusters = defaultdict(list)
        k = 0
        for label in labels:
            clusters[label].append(self.image_names[k])
            k += 1

        for clust in clusters:
            print ("Cluster: ", clust)
            for filename in clusters[clust]:
                print (filename)
            print ("\n************************\n")
        print ("********** Clustering method over **************")
        return labels

    def _cluster_model(self, model_name, c):
        if model_name == 'KMeans':
            model = KMeans(n_clusters=c, init='k-means++')
        elif model_name == 'HAC':
            model = AgglomerativeClustering(n_clusters=c, affinity='euclidean', linkage='ward')
        elif model_name == 'Spectral':
            model = spectral_clustering(n_clusters=c)
        else:
            print ("Options for models are KMeans, HAC or Spectral.")
            exit(-1)
        return model
