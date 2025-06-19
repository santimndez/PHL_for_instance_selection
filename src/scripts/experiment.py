import pandas as pd
import numpy as np
import time
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, accuracy_score

from data_reduction.representativeness import find_epsilon

import argparse

sys.path.append('../')
from my_dataset_reduction import phl_selection, phl_selection_from_scores, phl_scores, get_max_distance, \
                                 srs_selection, clc_selection, drop3_selection, cnn_selection

# argparser
parser = argparse.ArgumentParser(description='Instance selection experiment')
parser.add_argument('-d', '--dataset', type=str, default='../datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx', help='Dataset to use for the experiment')
parser.add_argument('-o', '--output', type=str, default='../results/drybeans/', help='Output folder for the results. ')
parser.add_argument('-t', '--target', type=str, default='Class', help='Target column in the dataset')
parser.add_argument('--profiling', action='store_true', help='Enable profiling mode for quick tests (uses a subset of the dataset)')
parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
args = parser.parse_args()

PROFILING = args.profiling # False para tomar todo el dataset, True para tomar una muestra para hacer pruebas rápidas

dbpath = args.dataset
if not os.path.exists(dbpath):
    raise FileNotFoundError(f"Dataset file not found: {dbpath}")

results_folder = args.output
if PROFILING: 
    results_folder = f'{results_folder[:-1]}_profiling/'

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

target = args.target


SEED = args.seed
np.random.seed(SEED)

# Load the dataset
if dbpath.endswith('.xlsx') or dbpath.endswith('.xls'):
    # Read the Excel file
    df = pd.read_excel(dbpath, )
else:
    # Read the CSV file
    df = pd.read_csv(dbpath)

# Check if the target column exists in the DataFrame
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in the dataset.")

if PROFILING and len(df) > 1000:
    df = df.sample(1000, random_state=SEED)

# Train test split
X = df.drop(columns=[target])
X = np.array(X)

# Convert target to numeric if it's not already
if not pd.api.types.is_integer_dtype(df[target]):
    le = LabelEncoder()
    y = le.fit_transform(df[target])
else:
    y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Classification models
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
xgb = XGBClassifier(n_estimators=100, random_state=SEED)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DATA REDUCTION
percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
metrics = ['reduction_ratio', 'representativeness', 'accuracy', 'f1', 'training_time', 'reduction_time']

# PHL Hyperparameter experiments
phl_results = pd.DataFrame(columns=['model', 'reduction_method', 'mode', 'dimensions', 'max_dimension', 'percentage', 'delta'] + metrics)

# deltas = [0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 1.0]
deltas = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
max_distance = get_max_distance(X_train_scaled, y_train)

modes = ['representative', 'vital'] # if not PROFILING else ['representative']
dimensions = [(0, 'restrictedDim')] # (1, 'restrictedDim') # (2, 'multiDim')] # if not PROFILING else [(1, 'restrictedDim')]

def reduce(X,y,perc,method):
    if method == 'SRS':
        X_red, y_red = srs_selection(X,y,perc)
    if method == 'CLC':
        X_red, y_red = clc_selection(X,y,perc)
    if method == 'PHL':
        X_red, y_red = phl_selection(X, y, 0.05, perc, 'restrictedDim', 2, 'representative')
    return X_red, y_red

models = {'KNN': knn, 'RF': rf, 'XGB': xgb}
reduction_methods = {'SRS': lambda X,y,perc: srs_selection(X,y,perc), 
                     'CLC': lambda X,y,perc: clc_selection(X,y,perc), 
                     'PHL': lambda X,y,perc: phl_selection(X,y,perc=perc, topological_radius=0.1*max_distance, scoring_version='restrictedDim', dimension=0, landmark_type='representative')}

reduction_methods_without_perc = {'CNN': lambda X,y: cnn_selection(X,y), 
                                'DROP3': lambda X,y: drop3_selection(X,y)}

all_reduction_methods = reduction_methods | reduction_methods_without_perc

for model_name in models.keys():
        for dimension, scoring_version in dimensions:
            for delta in deltas:
                # Get score of each instance
                t0 = time.time()
                outlier_scores = phl_scores(X_train_scaled, y_train, delta*max_distance, scoring_version, dimension)
                score_time = time.time()-t0
                for mode in modes:
                    for percentage in percentages:
                        # Reduce the dataset
                        t0 = time.time()
                        X_red, y_red = phl_selection_from_scores(X_train_scaled, y_train, 
                                                                 perc = percentage, 
                                                                 landmark_type=mode, 
                                                                 outlier_scores=outlier_scores)
                        reduction_time = time.time() - t0

                        # Fit the model
                        t0 = time.time()
                        model = models[model_name]
                        model.fit(X_red, y_red)
                        training_time = time.time() - t0

                        # Evaluate the model
                        y_pred_test = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred_test)
                        f1 = f1_score(y_test, y_pred_test, average='weighted')

                        # Calculate representativeness
                        epsilon = find_epsilon(X_train_scaled, y_train, X_red, y_red)

                        # Store the results
                        phl_results = phl_results.append({
                            'model': model_name,
                            'reduction_method': f'PHL',
                            'mode': mode,
                            'dimensions': scoring_version,
                            'max_dimension': dimension,
                            'percentage': percentage,
                            'delta': delta,
                            'reduction_ratio': len(y_red) / len(y_train),
                            'representativeness': epsilon,
                            'accuracy': accuracy,
                            'f1': f1,
                            'training_time': training_time,
                            'reduction_time': score_time + reduction_time,
                        }, ignore_index=True)
# Save the results
phl_results.to_csv(f'{results_folder}/phl_results.csv', index=False)

# EXPERIMENT OF COMPARISON OF INSTANCE SELECTION METHODS
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

results = pd.DataFrame(columns=['model', 'reduction_method', 'percentage'] + metrics)
for model_name, model in models.items():
        # Fit the model with the full dataset
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        t = time.time() - t0

        # Evaluate the model
        y_pred_test = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')

        epsilon = 1
        # Store the results
        results = results.append({
            'model': model_name,
            'reduction_method': 'None',
            'percentage': 1.0,
            'reduction_ratio': 1.0,
            'representativeness': 0,
            'accuracy': accuracy,
            'f1': f1,
            'training_time': t,
            'reduction_time': 0
        }, ignore_index=True)

        # Fit the model with reduced dataset
        for reduction_method, reduce in reduction_methods.items():
            for percentage in percentages:
                # Reduce the dataset
                t0 = time.time()
                X_red, y_red = reduce(X_train_scaled, y_train, percentage)
                reduction_time = time.time() - t0
                
                # Fit the model
                t0 = time.time()
                model = models[model_name]
                model.fit(X_red, y_red)
                training_time = time.time() - t0

                # Evaluate the model
                y_pred_test = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred_test)
                f1 = f1_score(y_test, y_pred_test, average='weighted')

                # Calculate representativeness
                epsilon = find_epsilon(X_train_scaled, y_train, X_red, y_red)

                # Store the results
                results = results.append({
                    'model': model_name,
                    'reduction_method': reduction_method,
                    'percentage': percentage,
                    'reduction_ratio': len(y_red) / len(y_train),
                    'representativeness': epsilon,
                    'accuracy': accuracy,
                    'f1': f1,
                    'training_time': training_time,
                    'reduction_time': reduction_time,
                }, ignore_index=True)

        # Reduce the dataset with methods that do not require percentage
        for reduction_method, reduce in reduction_methods_without_perc.items():
                # Reduce the dataset
                t0 = time.time()
                X_red, y_red = reduce(X_train_scaled, y_train)
                reduction_time = time.time() - t0

                # Fit the model
                t0 = time.time()
                model = models[model_name]
                model.fit(X_red, y_red)
                training_time = time.time() - t0

                # Evaluate the model
                y_pred_test = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred_test)
                f1 = f1_score(y_test, y_pred_test, average='weighted')

                # Calculate representativeness
                epsilon = find_epsilon(X_train_scaled, y_train, X_red, y_red)

                # Store the results
                results = results.append({
                    'model': model_name,
                    'reduction_method': reduction_method,
                    'percentage': 0,
                    'reduction_ratio': len(y_red) / len(y_train),
                    'representativeness': epsilon,
                    'accuracy': accuracy,
                    'f1': f1,
                    'training_time': training_time,
                    'reduction_time': reduction_time,
                }, ignore_index=True)
        
# Save the results
results.to_csv(results_folder + 'results.csv', index=False)