import pandas as pd
import numpy as np
import time
import os
import sys
import argparse

from tqdm import tqdm  # For progress bar

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, accuracy_score

from data_reduction.representativeness import find_epsilon

# Uncomment to hide FutureWarnings
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('../')
from my_dataset_reduction import phl_selection_k, phl_selection_from_scores, phl_scores_k, \
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

# Data reduction parameters
percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
metrics = ['reduction_ratio', 'representativeness', 'accuracy', 'f1', 'training_time', 'reduction_time']

# PHL parameters
SCORING_VERSION = 'restrictedDim'

models = {'KNN': knn, 'RF': rf, 'XGB': xgb}

SRS_REPS = 5 # Number of repetitions for SRS selection

reduction_methods = {'CLC': lambda X,y,perc: clc_selection(X,y,perc)}


phl_methods = {('restrictedDim', 0, 3, 'representative'),
               ('restrictedDim', 1, 3, 'representative'),
              ('restrictedDim', 0, 5, 'vital')}

reduction_methods_without_perc = {'CNN': lambda X,y: cnn_selection(X,y), 
                                'DROP3': lambda X,y: drop3_selection(X,y)}

all_reduction_methods = reduction_methods | reduction_methods_without_perc

# EXPERIMENT OF COMPARISON OF INSTANCE SELECTION METHODS

results = pd.DataFrame(columns=['model', 'reduction_method', 'percentage'] + metrics)

# Fit the model with the full dataset
for model_name, model in tqdm(models.items(), desc="Fitting models with full dataset"):
    # Fit the model with the full dataset
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    t = time.time() - t0

    # Evaluate the model
    y_pred_test = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='weighted')

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

# Save temporary results
results.to_csv(results_folder + 'results.csv', index=False)

# Reduce the dataset with SRS method
srs_results = pd.DataFrame(columns=['percentage'] + metrics)
for percentage in tqdm(percentages, desc="Reducing dataset with SRS method", leave=False):
    for _ in tqdm(range(SRS_REPS), desc="SRS repetition", leave=False):
        # Reduce the dataset
        t0 = time.time()
        X_red, y_red = srs_selection(X_train_scaled, y_train, percentage)
        reduction_time = time.time() - t0
        for model_name, model in tqdm(models.items(), desc="Models", leave=False):
            # Fit the model
            t0 = time.time()
            model.fit(X_red, y_red)
            training_time = time.time() - t0

            # Evaluate the model
            y_pred_test = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test, average='weighted')

            # Calculate representativeness
            epsilon = find_epsilon(X_train_scaled, y_train, X_red, y_red)

            # Store the results
            srs_results = srs_results.append({
                'percentage': percentage,
                'reduction_ratio': len(y_red) / len(y_train),
                'representativeness': epsilon,
                'accuracy': accuracy,
                'f1': f1,
                'training_time': training_time,
                'reduction_time': reduction_time,
            }, ignore_index=True)

srs_mean_results = srs_results.groupby('percentage').mean().reset_index()  
results.append(srs_mean_results.assign(model=model_name, reduction_method='SRS'), ignore_index=True)
# Save temporary results
results.to_csv(results_folder + 'results.csv', index=False)

# Reduce the dataset with methods that do not require percentage
if len(y_train)<50000:
    for reduction_method, reduce in tqdm(reduction_methods_without_perc.items(), desc="Reducing dataset with methods without percentage"):
        # Reduce the dataset
        t0 = time.time()
        X_red, y_red = reduce(X_train_scaled, y_train)
        reduction_time = time.time() - t0    
        for model_name, model in tqdm(models.items(), desc="Reduction methods without percentage", leave=False):
            # Fit the model
            t0 = time.time()
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

# Save temporary results
results.to_csv(results_folder + 'results.csv', index=False)

# Reduce the dataset with methods that require percentage
for reduction_method, reduce in tqdm(reduction_methods.items(), desc="Reduction methods with percentage", leave=False):
    for percentage in tqdm(percentages, desc="Percentages", leave=False):
        # Reduce the dataset
        t0 = time.time()
        X_red, y_red = reduce(X_train_scaled, y_train, percentage)
        reduction_time = time.time() - t0
        
        for model_name, model in models.items():
            # Fit the model
            t0 = time.time()
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

# Reduce the dataset with PHL methods
for phl_method in tqdm(phl_methods, desc="PHL methods"):
    # Get outlier scores
    t0 = time.time()
    scores = phl_scores_k(X_train_scaled, y_train, 
                          k=phl_method[1], 
                          scoring_version=phl_method[0], 
                          dimension=phl_method[1])
    score_time = time.time() - t0
    for percentage in tqdm(percentages, desc="Percentages", leave=False):
        # Select instances based on scores
        t0 = time.time()
        X_red, y_red = phl_selection_from_scores(X_train_scaled, y_train, percentage,
                                                 landmark_type=phl_method[3],
                                                 outlier_scores=scores)
        reduction_time = time.time() - t0
        
        for model_name, model in models.items():
            # Fit the model
            t0 = time.time()
            model.fit(X_red, y_red)
            training_time = time.time() - t0

            # Evaluate the model
            y_pred_test = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test, average='weighted')

            # Calculate representativeness
            epsilon = find_epsilon(X_train_scaled, y_train, X_red, y_red)

            # Store the results
            reduction_method = f'PHL_{"R" if phl_method[3] == "representative" else "V"}{phl_method[1] if phl_method[0] == "restrictedDim" else "".join([str(i) for i in range(phl_method[1]+1)])}_k={phl_method[2]}'
            results = results.append({
                'model': model_name,
                'reduction_method': reduction_method,
                'percentage': percentage,
                'reduction_ratio': len(y_red) / len(y_train),
                'representativeness': epsilon,
                'accuracy': accuracy,
                'f1': f1,
                'training_time': training_time,
                'reduction_time': reduction_time + score_time,
            }, ignore_index=True)
# Save the results
results.to_csv(results_folder + 'results.csv', index=False)