# Modified from original work by Perera-Lago J, Toscano-Duran V, Paluzo-Hidalgo E et al.
# Copyright 2024 Original Authors
# Modifications copyright 2025 Santiago Méndez García
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Citation request:
# If you use this code in your research, please cite:
# Perera-Lago J, Toscano-Duran V, Paluzo-Hidalgo E et al.
# "An in-depth analysis of data reduction methods for sustainable deep learning"
# Open Res Europe 2024, 4:101 (https://doi.org/10.12688/openreseurope.17554.2)

import numpy as np
from ripser import ripser
from scipy.spatial import KDTree
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def getMaxPersistence(ripser_pd):

	if ripser_pd.size == 0:
		max_persistence = 0
	else:
		finite_bars_where = np.invert(np.isinf(ripser_pd[:,1]))
		finite_bars = ripser_pd[np.array(finite_bars_where),:]
		max_persistence = np.max(finite_bars[:,1]-finite_bars[:,0], initial=0) # if there are no finite bars, return 0

	return max_persistence


def getPHOutlierScores_multiDim(point_cloud,topological_radius,max_dim_ripser):

	set_of_super_outliers = np.empty((0,point_cloud.shape[1]))
	super_outlier_indices = np.empty((0,0), int)
	outlier_scores_point_cloud_original_order = np.empty((0,0))

	kd_tree = KDTree(point_cloud)

	# precompute distance matrix
	distance_matrix = pairwise_distances(point_cloud, metric='euclidean', n_jobs=-1).astype(np.float16)

	#for point in point cloud, get delta nhood
	for point_index in range(point_cloud.shape[0]):
		outlier_score = 0

		point = point_cloud[point_index,:]

		indices = kd_tree.query_ball_point(point, r=topological_radius)
		number_of_neighbours = len(indices)-1


		if number_of_neighbours < 2:

			set_of_super_outliers = np.append(set_of_super_outliers, [point], axis=0)
			super_outlier_indices = np.append(super_outlier_indices,point_index)

		else:
			# delete point_index from indices if present
			indices = np.array(indices)
			indices = indices[indices != point_index]

			delta_point_cloud = distance_matrix[np.ix_(indices, indices)]
	
			diagrams = ripser(delta_point_cloud, maxdim=max_dim_ripser, distance_matrix=True)['dgms']

			for dimension in range(max_dim_ripser+1):
				intervals = diagrams[dimension]
				max_persistence = getMaxPersistence(intervals)
				#print max_persistence
				if max_persistence > outlier_score:
					outlier_score = max_persistence


		outlier_scores_point_cloud_original_order = np.append(outlier_scores_point_cloud_original_order,outlier_score)

		#print("This is the outlier score", outlier_score)

	return outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices


def getPHOutlierScores_restrictedDim(point_cloud,topological_radius,dimension):

	set_of_super_outliers = np.empty((0,point_cloud.shape[1]))
	super_outlier_indices = np.empty((0,0), int)
	outlier_scores_point_cloud_original_order = np.empty((0,0))

	kd_tree = KDTree(point_cloud)

    # precompute distance matrix
	distance_matrix = pairwise_distances(point_cloud, metric='euclidean').astype(np.float16)
	

	#for point in point cloud, get delta nhood
	for point_index in range(point_cloud.shape[0]):
		outlier_score = 0

		point = point_cloud[point_index,:]

		indices = kd_tree.query_ball_point(point, r=topological_radius)
	
		# delete point_index from indices if present
		indices = np.array(indices)
		indices = indices[indices != point_index]
		number_of_neighbours = len(indices)

		if number_of_neighbours < 2:

			set_of_super_outliers = np.append(set_of_super_outliers, [point], axis=0)
			super_outlier_indices = np.append(super_outlier_indices,point_index)

		else:

			delta_point_cloud = distance_matrix[np.ix_(indices, indices)]
			
			diagrams = ripser(delta_point_cloud, maxdim=dimension, distance_matrix=True)['dgms']

			intervals = diagrams[dimension]
			outlier_score = getMaxPersistence(intervals)


		outlier_scores_point_cloud_original_order = np.append(outlier_scores_point_cloud_original_order,outlier_score)

		#print("This is the outlier score", outlier_score)

	return outlier_scores_point_cloud_original_order, set_of_super_outliers, super_outlier_indices


#Landmark function

def getPHLandmarks(point_cloud, topological_radius, sampling_density, scoring_version, dimension, landmark_type):

	number_of_points = point_cloud.shape[0]
	number_of_PH_landmarks = int(round(number_of_points*sampling_density))

	if scoring_version == 'restrictedDim':

		outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices = getPHOutlierScores_restrictedDim(point_cloud,topological_radius,dimension)

	elif scoring_version == 'multiDim':

		max_dim_ripser = dimension
		outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices = getPHOutlierScores_multiDim(point_cloud,topological_radius,max_dim_ripser)


	number_of_super_outliers = set_of_super_outliers.shape[0]

	if landmark_type == 'representative':#small scores

		# sort outlier_scores_point_cloud_original_order
		sorted_indices_point_cloud_original_order = np.argsort(outlier_scores_point_cloud_original_order)

		#Permute zeros
		permuted_super_outlier_indices = np.random.permutation(sorted_indices_point_cloud_original_order[0:number_of_super_outliers])

		sorted_indices_point_cloud_original_order_without_super_outliers = np.array(sorted_indices_point_cloud_original_order[number_of_super_outliers:,])

		# We append the outliers at the end of the vector to select them last
		sorted_indices_point_cloud_original_order = np.append(sorted_indices_point_cloud_original_order_without_super_outliers,permuted_super_outlier_indices)

		PH_landmarks = point_cloud[sorted_indices_point_cloud_original_order[range(number_of_PH_landmarks)],:]

	elif landmark_type == 'vital':#large scores

		# sort outlier_scores_point_cloud_original_order
		sorted_indices_point_cloud_original_order = np.argsort(outlier_scores_point_cloud_original_order)

		#Permute zeros
		permuted_super_outlier_indices = np.random.permutation(sorted_indices_point_cloud_original_order[0:number_of_super_outliers])

		sorted_indices_point_cloud_original_order_without_super_outliers = np.array(sorted_indices_point_cloud_original_order[number_of_super_outliers:,])

		#We append the super outliers before the low scores so we select these last
		sorted_indices_point_cloud_original_order = np.append(permuted_super_outlier_indices,sorted_indices_point_cloud_original_order_without_super_outliers)

		# we flip the vector to keep in line with previous landmark call
		sorted_indices_point_cloud_original_order = np.flip(sorted_indices_point_cloud_original_order)

		PH_landmarks = point_cloud[sorted_indices_point_cloud_original_order[range(number_of_PH_landmarks)],:]


	return PH_landmarks, sorted_indices_point_cloud_original_order, number_of_super_outliers


def phl_selection(X, y, topological_radius, perc, scoring_version, dimension, landmark_type):
    
    classes = np.unique(y)
    X_res = np.array([],dtype=float)
    y_res = np.array([],dtype=float)
    n_features = np.shape(X)[1]
    
    for cl in classes:
        pool_cl = np.where(y==cl)
        X_cl = X[pool_cl]
        PHLandmarks = getPHLandmarks(point_cloud=X_cl, topological_radius=topological_radius, sampling_density=perc,  scoring_version=scoring_version, dimension=dimension, landmark_type=landmark_type)[0]
        n_cl = len(PHLandmarks)
        X_res = np.append(X_res, PHLandmarks)
        y_res = np.append(y_res, np.repeat(cl,n_cl))
        
    return X_res.reshape(-1, n_features), y_res

# Functions to get landmarks from outlier scores
# This way we can use the same outlier scores for different landmark types or data reduction percentages

def getPHLOutlierScores(point_cloud, topological_radius, scoring_version, dimension):
	if scoring_version == 'restrictedDim':
		outlier_scores_point_cloud_original_order, _, _ = getPHOutlierScores_restrictedDim(point_cloud,topological_radius,dimension)

	elif scoring_version == 'multiDim':
		max_dim_ripser = dimension
		outlier_scores_point_cloud_original_order, _, _ = getPHOutlierScores_multiDim(point_cloud,topological_radius,max_dim_ripser)
	return outlier_scores_point_cloud_original_order

# Get outlier scores for each instance in the dataset
def phl_scores(X, y, topological_radius, scoring_version, dimension):
	classes = np.unique(y)
	outlier_scores_total = np.zeros(X.shape[0],dtype=float)
 
	for cl in classes:
		pool_cl = np.where(y==cl)
		X_cl = X[pool_cl]

		outlier_scores = getPHLOutlierScores(point_cloud=X_cl, 
                                       topological_radius=topological_radius, 
                                       scoring_version=scoring_version, 
                                       dimension=dimension)
  
		outlier_scores_total[pool_cl] = outlier_scores

	return outlier_scores_total

def phl_from_scores(point_cloud, sampling_density, landmark_type, outlier_scores):
	number_of_points = point_cloud.shape[0]
	number_of_PH_landmarks = int(round(number_of_points*sampling_density))
	number_of_super_outliers = np.count_nonzero(outlier_scores == 0)

	# sort outlier_scores_point_cloud_original_order
	sorted_indices_point_cloud_original_order = np.argsort(outlier_scores)

	#Permute zeros
	permuted_super_outlier_indices = np.random.permutation(sorted_indices_point_cloud_original_order[0:number_of_super_outliers])
	
	sorted_indices_point_cloud_original_order_without_super_outliers = np.array(sorted_indices_point_cloud_original_order[number_of_super_outliers:,])

	if landmark_type == 'representative':#small scores

		# We append the outliers at the end of the vector to select them last
		sorted_indices_point_cloud_original_order = np.append(sorted_indices_point_cloud_original_order_without_super_outliers,permuted_super_outlier_indices)

	elif landmark_type == 'vital':#large scores

		#We append the super outliers before the low scores so we select these last
		sorted_indices_point_cloud_original_order = np.append(permuted_super_outlier_indices,sorted_indices_point_cloud_original_order_without_super_outliers)

		# we flip the vector to keep in line with previous landmark call
		sorted_indices_point_cloud_original_order = np.flip(sorted_indices_point_cloud_original_order)

	PH_landmarks = point_cloud[sorted_indices_point_cloud_original_order[range(number_of_PH_landmarks)],:]

	return PH_landmarks

# Get landmarks from outlier scores
def phl_selection_from_scores(X, y, perc, landmark_type, outlier_scores):
    classes = np.unique(y)
    X_res = np.array([],dtype=float)
    y_res = np.array([],dtype=float)
    n_features = np.shape(X)[1]
    
    for cl in classes:
        pool_cl = np.where(y==cl)
        X_cl = X[pool_cl]
        PHLandmarks = phl_from_scores(point_cloud=X_cl, 
                                      sampling_density=perc, 
                                      landmark_type=landmark_type, 
                                      outlier_scores=outlier_scores[pool_cl])
        n_cl = len(PHLandmarks)
        X_res = np.append(X_res, PHLandmarks)
        y_res = np.append(y_res, np.repeat(cl,n_cl))
        
    return X_res.reshape(-1, n_features), y_res

def get_max_distance(X, y):
	"""
	Calculate the maximum distance between any two points of the same class in X.
	"""
	classes = np.unique(y)
	max_distance = 0.0

	for cl in classes:
		pool_cl = np.where(y == cl)
		X_cl = X[pool_cl]
		
		if len(X_cl) > 1:  # Ensure there are at least two points to compare
			distance_matrix = pairwise_distances(X_cl, metric='euclidean').astype(np.float16)
			max_distance_cl = np.max(distance_matrix)
			if max_distance_cl > max_distance:
				max_distance = max_distance_cl

	return max_distance

def get_mean_neighbors(X, y, topological_radius):
	"""
	Calculate the mean number of neighbors within the topological radius for each point in X.
	"""
	classes = np.unique(y)
	mean_neighbors = 0

	for cl in classes:
		pool_cl = np.where(y == cl)
		X_cl = X[pool_cl]
		
		kd_tree = KDTree(X_cl)
		neighbors_count = [len(kd_tree.query_ball_point(point, r=topological_radius)) - 1 for point in X_cl]

		mean_neighbors += np.sum(neighbors_count)

	return mean_neighbors / len(y)

def get_super_outliers(X, y, topological_radius):
	"""
	Identify super outliers in the dataset based on the topological radius.
	"""
	classes = np.unique(y)
	super_outliers = 0

	for cl in classes:
		pool_cl = np.where(y == cl)
		X_cl = X[pool_cl]
		
		kd_tree = KDTree(X_cl)
		for point_index in range(X_cl.shape[0]):
			point = X_cl[point_index, :]
			indices = kd_tree.query_ball_point(point, r=topological_radius)
			if len(indices) <= 2:
				super_outliers+=1

	return super_outliers


def estimate_delta(X, y, k):
	"""
	Estima un radio r tal que, en promedio, cada instancia tiene k vecinos
	de su misma clase dentro de ese radio.

	Parámetros:
		X: np.ndarray, características (ya escaladas)
		y: np.ndarray, etiquetas (enteras o codificadas)
		k: int, número objetivo de vecinos por instancia. Debe ser menor o igual que el número de instancias de la clase minoritaria.

	Retorna:
		r: float, radio promedio estimado
	"""
	kth_distances = np.zeros(len(y), dtype=float)

	for cl in np.unique(y):
		# Filtrar por clase
		X_cl = X[y == cl]

		# Vecinos más cercanos dentro de la misma clase
		nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
		nbrs.fit(X_cl)
		distances, _ = nbrs.kneighbors(X_cl)

		# Ignorar el primer vecino (distancia 0 a sí mismo)
		kth_distances[y==cl] = distances[:, k]  # k+1 vecinos → índice k es el k-ésimo vecino real
	
	# return median of kth distances
	return np.median(kth_distances)


def phl_scores_k(X, y, k, scoring_version, dimension):
	"""
	Calcula los scores de PHL basados en el número de vecinos de la misma clase
	dentro de un radio estimado.

	Parámetros:
		X: np.ndarray, características (ya escaladas)
		y: np.ndarray, etiquetas (enteras o codificadas)
		k: int, número objetivo de vecinos por instancia

	Retorna:
		outlier_scores: np.ndarray, scores de PHL para cada instancia
	"""
	topological_radius = estimate_delta(X, y, k)
	return phl_scores(X, y, topological_radius, scoring_version, dimension)

def phl_selection_k(X, y, k, perc, scoring_version, dimension, landmark_type):
	"""
	Selecciona instancias representativas basadas en los scores de PHL
	y el número de vecinos de la misma clase dentro de un radio estimado.

	Parámetros:
		X: np.ndarray, características (ya escaladas)
		y: np.ndarray, etiquetas (enteras o codificadas)
		k: int, número objetivo de vecinos por instancia
		perc: float, porcentaje de reducción deseado
		scoring_version: str, versión del scoring ('restrictedDim' o 'multiDim')
		dimension: int, dimensión para el cálculo de PH
		landmark_type: str, tipo de landmark ('representative' o 'vital')

	Retorna:
		X_res: np.ndarray, conjunto reducido de características
		y_res: np.ndarray, etiquetas correspondientes al conjunto reducido
	"""
	outlier_scores = phl_scores_k(X, y, k, scoring_version, dimension)
	return phl_selection_from_scores(X, y, perc, landmark_type, outlier_scores)