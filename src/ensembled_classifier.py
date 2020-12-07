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
oversampled_models = pd.read_pickle(os.path.join(MODEL_DIR, 'oversampled_models.pkl'))
tunned_models = pd.read_pickle(os.path.join(MODEL_DIR, 'tunned_models.pkl'))
print('ok.\n')

chosen_models = [
    ('Random Forest', tunned_models['models'][0][1]),
    ('Logistic Regressor', oversampled_models['models'][1][1]),
    ('XGB Classifier', tunned_models['models'][1][1])
]

vclf = VotingClassifier(estimators=chosen_models, voting='soft', weights=[6,5,4])
vclf = vclf.fit(X_train_over_scaled, y_train_over)

from utils import trainModels, makePredictions

recall, accu, roc_auc = makePredictions(vclf, X_test_scaled, y_test, verbose=True)

"""stack_clf = StackingClassifier(chosen_models,
                               final_estimator=tunned_models['models'][0][1],
                               cv=2) 
stack_clf.fit(X_train_over_scaled, y_train_over)

makePredictions(stack_clf, X_test_scaled, y_test, verbose=True)"""

final_results = {'Voting Classifier': {'recall': recall, 'Accuracy': accu, 'ROC-AUC':roc_auc}}

print('Saving models...', end="")
ensembled_model_data = pd.Series({
    'oe_features': tunned_models['oe_features'],
    'ohe_features': tunned_models['ohe_features'],
    'scaling_features': tunned_models['scaling_features'],
    'ohe': tunned_models['ohe'],
    'oe': tunned_models['oe'],
    'scaler_train': tunned_models['scaler_train'],
    'scaler_test': tunned_models['scaler_test'],
    'models': vclf,
    'models_results': final_results,
    'features': tunned_models['features']
})

ensembled_model_data.to_pickle(os.path.join(MODEL_DIR, 'ensembled_model.pkl'))
print('ok.')