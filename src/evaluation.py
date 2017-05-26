from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from bhtsne import tsne
import operator
from scipy.spatial.distance import cdist


class Evaluation():

    def __init__(self, model_name, data_name, data, k_range, supervised_labels=False):
        self.model_name = model_name
        self.data_name = data_name
        self.data = data
        self.k_range = k_range[0:9]  # max 9 k numbers
        self.supervised_labels = supervised_labels

    def evaluate(self):

        # dimensionality reduction using Barnes-Hut implementation of t-SNE
        data_2d = tsne(np.array(self.data, dtype=np.float64))

        silhouette = {}
        dunn_score = {}
        rand_score = {}
        nmi_score = {}

        # data for using elbow method
        centres_list = []
        k_distance_list = []
        cluster_index_list = []
        distance_list = []

        subplot_counter = 331
        for c in self.k_range:
            print ("Number of clusters c:", c)

            # create model
            model = self._cluster_model(self.model_name, c)
            model.fit(self.data)
            labels = model.labels_

            # visualization evaluation - 2D Scatter plot of data
            plt.subplot(subplot_counter)
            plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
            subplot_counter += 1

            # elbow method calculations AIC
            centres_list.append(model.cluster_centers_)
            k_dist = cdist(self.data, model.cluster_centers_, 'euclidean')
            k_distance_list.append(k_dist)
            cluster_index_list.append(np.argmin(k_dist, axis=1))
            distance_list.append(np.min(k_dist, axis=1))

            # internal evaluation methods not using true labels
            silhouette[c] = silhouette_score(np.array(self.data), labels, metric='euclidean')
            print ("silhoutete with cluster number: ", silhouette[c], c)
            dunn_score[c] = self.dunn_fast(self.data, labels)
            print ("dunn with cluster number: ", dunn_score[c], c)

            # external evaluation methods using true labels (supervised datasets)
            if self.supervised_labels:
                rand_score[c] = adjusted_rand_score(self.supervised_labels, labels)
                nmi_score[c] = normalized_mutual_info_score(self.supervised_labels, labels)

        plt.show()
        self._plot_elbow_method(distance_list)

        print (max(silhouette.items(), key=operator.itemgetter(1)))
        print (max(dunn_score.items(), key=operator.itemgetter(1)))

        print ("Silhoutete ", silhouette)
        print ("Dunn ", dunn_score)

    def _plot_elbow_method(self, distances):
        size = len(self.data)
        average_within = [np.sum(dist)/size for dist in distances]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.k_range, average_within, 'g*-')
        plt.grid(True)
        plt.xlabel('Number of clusters')
        plt.ylabel('Average within-cluster sum of squares')
        plt.title('Elbow for ' + self.model_name + ' clustering.')
        plt.show()

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

    def _delta_fast(self, ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]
        return np.min(values)

    def _big_delta_fast(self, ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]
        return np.max(values)

    def dunn_fast(self, points, labels):
        # https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py
        distances = euclidean_distances(points)
        ks = np.sort(np.unique(labels))
        deltas = np.ones([len(ks), len(ks)]) * 1000000
        big_deltas = np.zeros([len(ks), 1])
        l_range = list(range(0, len(ks)))

        for k in l_range:
            for l in (l_range[0:k] + l_range[k + 1:]):
                deltas[k, l] = self._delta_fast((labels == ks[k]), (labels == ks[l]), distances)

            big_deltas[k] = self._big_delta_fast((labels == ks[k]), distances)

        di = np.min(deltas) / np.max(big_deltas)
        return di
