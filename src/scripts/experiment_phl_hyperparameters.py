import pandas as pd
import numpy as np
import time
import os
import sys
import argparse

import tqdm  # For progress bar

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, accuracy_score

from data_reduction.representativeness import find_epsilon



sys.path.append('../')
from my_dataset_reduction import phl_selection_from_scores, phl_scores_k, estimate_delta, \
                                 get_max_distance, get_super_outliers, get_mean_neighbors
from my_dataset_reduction import srs_selection

# argparser
parser = argparse.ArgumentParser(description='Instance selection experiment')
parser.add_argument('-d', '--dataset', type=str, default='../datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx', help='Dataset to use for the experiment')
parser.add_argument('-o', '--output', type=str, default='../results/phl_hyperparameter_experiment/', help='Output folder for the results. ')
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

X_train_full, _, y_train_full, _ = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Get train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full)

# Classification models
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
xgb = XGBClassifier(n_estimators=100, random_state=SEED)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# DATA REDUCTION
percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
metrics = ['reduction_ratio', 'representativeness', 'accuracy', 'f1', 'training_time', 'reduction_time']

# PHL Hyperparameter experiments
phl_results = pd.DataFrame(columns=['model', 'reduction_method', 'mode', 'dimensions', 'max_dimension', 'percentage', 'k'] + metrics) # Store results of PHL hyperparameter experiments
super_outliers_df = pd.DataFrame(columns=['k', 'delta', 'super_outliers', 'mean_neighbors']) # Store super outliers and mean neighbors for each delta

k_values = [[1, 2, 3, 5, 7, 10, 15], [1, 2, 3, 5, 7, 10, 15], [1, 2, 3, 5]] #, [2, 5, 10]]
modes = ['representative', 'vital'] # if not PROFILING else ['representative']
dimensions = [(0, 'restrictedDim'), (1, 'restrictedDim') , (2, 'multiDim')]

models = {'KNN': knn, 'RF': rf, 'XGB': xgb}

# Get super outliers and mean neighbors for the current delta
max_distance = get_max_distance(X_train_scaled, y_train)
distinct_k_values = sorted(set(k for sublist in k_values for k in sublist)) # get distinct values from k_values

for k in distinct_k_values:
    delta = estimate_delta(X_train_scaled, y_train, k)
    super_outliers = get_super_outliers(X_train_scaled, y_train, delta)
    mean_neighbors = get_mean_neighbors(X_train_scaled, y_train, delta)
    super_outliers_df = super_outliers_df.append({
        'k': k,
        'delta': delta,
        'super_outliers': super_outliers,
        'mean_neighbors': mean_neighbors
    }, ignore_index=True)

super_outliers_df.to_csv(f'{results_folder}/neighbors.csv', index=False)
print(f"Super outliers and mean neighbors saved to {results_folder}neighbors.csv")

# Train models with full dataset to get baseline results
baseline_results = pd.DataFrame(columns=['model', 'accuracy', 'f1', 'training_time']) # Store results of baseline models
for model_name, model in models.items():
    # Fit the model
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - t0

    # Evaluate the model
    y_pred_test = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred_test)
    f1 = f1_score(y_val, y_pred_test, average='weighted')

    # Store the results for full dataset
    baseline_results = baseline_results.append({
        'model': model_name,
        'accuracy': accuracy,
        'f1': f1,
        'training_time': training_time,
    }, ignore_index=True)
baseline_results.to_csv(f'{results_folder}baseline_results.csv', index=False)
print(f"Baseline results saved to {results_folder}baseline_results.csv")

# Train models with Stratified Random Sampling to get baseline results
SRS_REPS = 5 # Number of repetitions for SRS to get a more stable estimate
print("Reduction with SRS (baseline): ")
srs_results = pd.DataFrame(columns=['model', 'percentage', 'rep'] + metrics) # Store results of SRS
for percentage in tqdm.tqdm(percentages, desc="Percentage", leave=False):
    for rep in tqdm.tqdm(range(SRS_REPS), desc="SRS Repetitions", leave=False):
        t0 = time.time()
        X_red, y_red = srs_selection(X_train_scaled, y_train, perc=percentage)
        reduction_time = time.time() - t0
        for model_name, model in tqdm.tqdm(models.items(), desc="Model", leave=False):
            # Fit the model
            t0 = time.time()
            model.fit(X_red, y_red)
            training_time = time.time() - t0

            # Evaluate the model
            y_pred_test = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred_test)
            f1 = f1_score(y_val, y_pred_test, average='weighted')

            # Calculate representativeness
            epsilon = find_epsilon(X_train_scaled, y_train, X_red, y_red)

            # Store the results for SRS
            srs_results = srs_results.append({
                'model': model_name,
                'percentage': percentage,
                'rep': rep,
                'reduction_ratio': len(y_red) / len(y_train),
                'representativeness': epsilon,
                'accuracy': accuracy,
                'f1': f1,
                'training_time': training_time,
                'reduction_time': reduction_time,
            }, ignore_index=True)
# Save mean results to phl_results
srs_mean_results = srs_results.groupby(['model', 'percentage']).mean().reset_index()
srs_mean_results.to_csv(f'{results_folder}srs_results.csv', index=False)
print(f"SRS results saved to {results_folder}srs_results.csv")

print("Reduction with PHL: ")
for (dimension, scoring_version), k_list in tqdm.tqdm(zip(dimensions, k_values), desc="Dimension", leave=False):
    for k in tqdm.tqdm(k_list, desc="K", leave=False):
        # Get score of each instance
        t0 = time.time()
        outlier_scores = phl_scores_k(X_train_scaled, y_train, k, scoring_version, dimension)
        score_time = time.time()-t0
        for mode in tqdm.tqdm(modes, desc="Mode", leave=False):
            for percentage in tqdm.tqdm(percentages, desc="Percentage", leave=False):
                # Reduce the dataset
                t0 = time.time()
                X_red, y_red = phl_selection_from_scores(X_train_scaled, y_train,
                                                        perc=percentage,
                                                        landmark_type=mode,
                                                        outlier_scores=outlier_scores)
                reduction_time = time.time() - t0
                for model_name, model in tqdm.tqdm(models.items(), desc="Model", leave=False):

                    # Fit the model
                    t0 = time.time()
                    model.fit(X_red, y_red)
                    training_time = time.time() - t0

                    # Evaluate the model
                    y_pred_test = model.predict(X_val_scaled)
                    accuracy = accuracy_score(y_val, y_pred_test)
                    f1 = f1_score(y_val, y_pred_test, average='weighted')

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
                        'k': k,
                        'reduction_ratio': len(y_red) / len(y_train),
                        'representativeness': epsilon,
                        'accuracy': accuracy,
                        'f1': f1,
                        'training_time': training_time,
                        'reduction_time': score_time + reduction_time,
                    }, ignore_index=True)

# Save the results
phl_results.to_csv(f'{results_folder}/phl_results.csv', index=False)
print(f"PHL hyperparameter experiment results saved to {results_folder}phl_results.csv")