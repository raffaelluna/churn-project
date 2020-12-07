print('Importing packages...', end="")
import os
import pandas as pd
import numpy as np
print('ok.\n')

BASE_DIR = '~/ML-AZ/Projeto - Churn'
DATA_DIR = os.path.join(BASE_DIR, 'input')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print('Loading data...', end="")
X_train_over_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_train_over_scaled.csv'))
y_train_over = pd.read_csv(os.path.join(DATA_DIR, 'y_train_over.csv'))
X_test_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
print('ok.\n')

print('Loading model...', end="")
ensembled_model = pd.read_pickle(os.path.join(MODEL_DIR, 'ensembled_model.pkl'))
model = ensembled_model['models']
print('ok.\n')

print('Generating Age Scaler...', end="")
from sklearn.preprocessing import StandardScaler
age_scaler = StandardScaler()
age_scaler.fit(X_test['Age'].values.reshape(-1,1))
print('ok.\n')

#Train model with most important features. Check notebook for more infos.
important_features = ['Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender', 'Age', 'IsActiveMember', 'NumOfProducts']

model.fit(X_train_over_scaled[important_features], y_train_over)

from utils import trainModels, makePredictions

recall, accu, roc_auc = makePredictions(model, X_test_scaled[important_features], y_test, verbose=True)

final_results = {'Reduced Voting Classifier': {'recall': recall, 'Accuracy': accu, 'ROC-AUC':roc_auc}}

print('Saving models...', end="")
reduced_model_data = pd.Series({
    'ohe': ensembled_model['ohe'],
    'oe': ensembled_model['oe'],
    'scaler_train': ensembled_model['scaler_train'],
    'scaler_test': age_scaler,
    'models': model,
    'models_results': final_results,
    'features': important_features
})

reduced_model_data.to_pickle(os.path.join(MODEL_DIR, 'reduced_model.pkl'))
print('ok.')