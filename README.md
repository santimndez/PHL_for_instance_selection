# Application of Topological Data Analysis to the problem of instance selection for classification

This repository contains the python implementation of the experiments of comparison of instance selection methods from the Bachelor's Thesis "Aplicación del Análisis Topológico de Datos al problema de selección de instancias para clasificación" by Santiago Méndez García.

## Structure

- src
  - my_dataset_reduction: python module with instance selection for classification methods.
    - __init__.py: exportation of functions.
    - imblearn_wrapper.py: implementation of SRS, CNN and CLC using the Imbalanced-learn library.
    - drop3.py: DROP3 implementation adapting the code of [2].
    - phl.py: PHL implementation derived from [1].
    - divide_and_conquer.py: divide and conquer approach for instance selection in large datasets.
  - notebooks: experiment analysis python notebooks.
    - SGAI_vs_MyPHL.ipynb: compares PHL implementation with the implementation in [1].
    - phl_experiment_analysis.ipynb: analysis of PHL hyperparameter tuning experiment.
    - experiment_analysis.ipynb: analysis of the experiment of comparison of instance selection methods.
  - scripts: experiment scripts:
    - experiment_delta.py: experiment to analyze the parameter delta of PHL.
    - experiment_phl_hyperparameters.py: PHL hyperparameter tuning experiment.
    - experiment_reduction.py: experiment of comparison of instance selection methods.
    - run_experiments.py: script to run several experiments.
  - results: results folder.

## Original Work and Citation

Used datasets:

- Pima Indians Diabetes Database. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database. Accessed 1 july 2025.
- Dry Bean: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset. Accessed 1 july 2025.
- Diabetes Health Indicators Dataset: Diabetes binary 5050split. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv. Accessed 1 july 2025.
- Diabetes Health Indicators Dataset: Diabetes binary. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv. Accessed 1 july 2025.

The following files are derived from other work (read the file header for details).
:

- src/my_dataset_reduction/phl.py:

[1] Javier Perera Lago, Eduardo Paluzo Hidalgo y Víctor Toscano Durán. Repositorio Survey Green AI. https://github.com/Cimagroup/SurveyGreenAI/. Accessed July 1, 2025. Mar. de 2024. doi: 10.5281/zenodo.10844558.

- src/my_dataset_reduction/drop3.py:

[2] Washington Cunha et al. A Comparative Survey of Instance Selection Methods ap-
plied to NonNeural and Transformer-Based Text Classification. https://github.com/waashk/instanceselection. Accessed July 1, 2025. 2023.

If you use this code in academic research, please cite the original work accordingly.