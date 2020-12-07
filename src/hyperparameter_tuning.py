print('Importing packages...', end="")
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
print('ok.\n')

BASE_DIR = '~/ML-AZ/Projeto - Churn'
DATA_DIR = os.path.join(BASE_DIR, 'input')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print('Loading data...', end="")
X_train_over_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_train_over_scaled.csv'))
y_train_over = pd.read_csv(os.path.join(DATA_DIR, 'y_train_over.csv'))
X_test_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
print('ok.\n')

print('Loading models...', end="")
models = pd.read_pickle(os.path.join(MODEL_DIR, 'oversampled_models.pkl'))
print('ok.\n')

xgb = models['models'][2][1]
rf = models['models'][0][1]

from sklearn.model_selection import RandomizedSearchCV

print('Tunning XGB...', end="")
params_xgb = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

random_xgb = RandomizedSearchCV(xgb, 
                                param_distributions=params_xgb, 
                                n_iter=10, 
                                scoring='recall', 
                                cv=5, 
                                verbose=3, 
                                random_state=1001 )

random_xgb.fit(X_train_over_scaled, y_train_over)

print('\n XGB Best Estimator: {}'.format(random_xgb.best_estimator_))
print('\n XGB Best Hyperparameters: {}'.format(random_xgb.best_params_))
print('ok.\n')

print('Tunning Random Forest...', end="")
params_rf = { 
    'n_estimators': [25, 50, 100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

random_rf = RandomizedSearchCV(rf, 
                                param_distributions=params_rf, 
                                n_iter=10, 
                                scoring='recall', 
                                cv=5, 
                                verbose=3, 
                                random_state=1001 )

random_rf.fit(X_train_over_scaled, y_train_over)

print('\n RF Best Estimator: {}'.format(random_rf.best_estimator_))
print('\n RF Best Hyperparameters: {}'.format(random_rf.best_params_))
print('ok.')

best_models = [
    ('Best Random Forest', random_rf.best_estimator_),
    ('Best XGB Classifier', random_xgb.best_estimator_)
]

from utils import trainModels, makePredictions

best_model_results = trainModels(best_models,
                                X_train_over_scaled[models['features']], 
                                y_train_over, 
                                X_test_scaled[models['features']], 
                                y_test, 
                                verbose=True)

print('Saving models...', end="")
best_model_data = pd.Series({
    'oe_features': models['oe_features'],
    'ohe_features': models['ohe_features'],
    'scaling_features': models['scaling_features'],
    'ohe': models['ohe'],
    'oe': models['oe'],
    'scaler_train': models['scaler_train'],
    'scaler_test': models['scaler_test'],
    'models': best_models,
    'models_results': best_model_results,
    'features': models['features']
})

best_model_data.to_pickle(os.path.join(MODEL_DIR, 'tunned_models.pkl'))
print('ok.')