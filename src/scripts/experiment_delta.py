import pandas as pd
import numpy as np
import os
import sys
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

sys.path.append('../')
from my_dataset_reduction import get_super_outliers, get_mean_neighbors, get_max_distance

DATASETS = [
    '../datasets/pima.csv',
    '../datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx',
    # '../datasets/diabetes_health_indicators/diabetes_binary_5050split_health_indicators_BRFSS2015.csv',
    # '../datasets/diabetes_health_indicators/diabetes_binary_health_indicators_BRFSS2015.csv',
]

TARGETS = [
    'Outcome',  # Pima Indians Diabetes Database
    'Class',    # Dry Bean Dataset
    # 'Diabetes_binary',  # Diabetes Health Indicators Dataset (5050 split)
    # 'Diabetes_binary',  # Diabetes Health Indicators Dataset (full)
]

# argparser
parser = argparse.ArgumentParser(description='Instance selection experiment')
parser.add_argument('-d', '--datasets', nargs='+', type=str, default=DATASETS, help='Datasets to use for the experiment')
parser.add_argument('-t', '--targets', nargs='+', type=str, default=TARGETS, help='Target column in the dataset')
parser.add_argument('-o', '--output', type=str, default='../results', help='Output folder for the results. ')
parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
args = parser.parse_args()

datasets = args.datasets
targets = args.targets
results_folder = args.output

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

SEED = args.seed
np.random.seed(SEED)


deltas = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]

super_outliers_df = pd.DataFrame(columns=['dataset', 'n_samples', 'max_distance', 'delta', 'super_outliers', 'mean_neighbors']) # Store super outliers and mean neighbors for each delta
# Get super outliers and mean neighbors for the current delta
for dbpath, target in zip(datasets, targets):
    print(f"Processing dataset: {dbpath}")
    if not os.path.exists(dbpath):
        raise FileNotFoundError(f"Dataset file not found: {dbpath}")

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

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    max_distance = get_max_distance(X_train_scaled, y_train)

    for delta in deltas:
        print(f"Delta = {delta}")
        super_outliers = get_super_outliers(X_train_scaled, y_train, delta*max_distance)
        mean_neighbors = get_mean_neighbors(X_train_scaled, y_train, delta*max_distance)
        super_outliers_df = super_outliers_df.append({
            'dataset': dbpath,
            'n_samples': len(X_train_scaled),
            'max_distance': max_distance,
            'delta': delta,
            'super_outliers': super_outliers,
            'mean_neighbors': mean_neighbors
        }, ignore_index=True)

super_outliers_df.to_csv(f'{results_folder}/neighbors.csv', index=False)
print(f"Super outliers and mean neighbors saved to {results_folder}/neighbors.csv")