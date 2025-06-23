import numpy as np
from math import trunc
#from scipy.cluster.vq import kmeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import CondensedNearestNeighbour

from sklearn.cluster import KMeans

def clc_selection(X, y, perc):
    # Number of selected instances per class
    class_counts = Counter(y)
    target_counts = {cls: max(1, trunc(count * perc)) for cls, count in class_counts.items()}
    
    # ClusterCentroids prototype generation method
    kmeans = KMeans(n_init=1)
    clc_sampler = ClusterCentroids(sampling_strategy=target_counts, voting='soft', estimator=kmeans)
    X_res, y_res = clc_sampler.fit_resample(X, y)
    return X_res, y_res

def srs_selection(X, y, perc):
    # Number of selected instances per class
    class_counts = Counter(y)
    target_counts = {cls: max(1, trunc(count * perc)) for cls, count in class_counts.items()}

    # RandomUnderSampler selection method
    rus = RandomUnderSampler(sampling_strategy=target_counts)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res

def cnn_selection(X, y, perc=None, n_neighbors=3):
    if perc is not None:
        # Number of selected instances per class
        class_counts = Counter(y)
        target_counts = {cls: max(1, trunc(count * perc)) for cls, count in class_counts.items()}

        cnn_sampler = CondensedNearestNeighbour(sampling_strategy=target_counts, n_neighbors=n_neighbors)
    else:
        cnn_sampler = CondensedNearestNeighbour(sampling_strategy='all', n_neighbors=n_neighbors)
    
    X_res, y_res = cnn_sampler.fit_resample(X, y)
    
    return X_res, y_res

