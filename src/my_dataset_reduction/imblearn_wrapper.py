import numpy as np
from math import trunc
#from scipy.cluster.vq import kmeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import CondensedNearestNeighbour

from sklearn.cluster import KMeans
##############################################################################
#CLC
#Clustering Centroids Selection

# def clc_selection_toscano(X,y,perc):
#     
#     classes = np.unique(y)
#     X_res = np.array([],dtype=float)
#     y_res = np.array([],dtype=float)
#     n_features = np.shape(X)[1]
#     
#     for cl in classes:
#         pool_cl = np.where(y==cl)
#         X_cl = X[pool_cl]
#         n_cl = trunc(len(X_cl)*perc)
#         centroids = kmeans(X_cl,k_or_guess=n_cl,iter=1)[0]
#         X_res = np.append(X_res, centroids)
#         y_res = np.append(y_res, np.repeat(cl,len(centroids)))
#         
#     return X_res.reshape(-1, n_features), y_res

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

