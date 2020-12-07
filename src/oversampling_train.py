print('Importing packages...', end="")
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

print('ok.\n')

BASE_DIR = '~/ML-AZ/Projeto - Churn'
DATA_DIR = os.path.join(BASE_DIR, 'input')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print('Loading models...', end="")
models = pd.read_pickle(os.path.join(MODEL_DIR, 'first_models.pkl'))
print('ok.\n')

#Loading dataset
print('Loading dataset...', end="")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_out.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
print('ok.\n')

#Variable encoders
print('Enconding Variables...', end="")
X_train[models['oe_features']] = models['oe'].transform(X_train[models['oe_features']])
X_test[models['oe_features']] = models['oe'].transform(X_test[models['oe_features']])

ohe_df = pd.DataFrame(models['ohe'].transform(X_train[models['ohe_features']]), 
                    columns=models['ohe'].get_feature_names(models['ohe_features']))

ohe_df_test = pd.DataFrame(models['ohe'].transform(X_test[models['ohe_features']]), 
                    columns=models['ohe'].get_feature_names(models['ohe_features']))

df_train = pd.concat([ohe_df, X_train.drop(models['ohe_features'], axis=1)], axis=1)
df_test = pd.concat([ohe_df_test, X_test.drop(models['ohe_features'], axis=1)], axis=1)
print('ok.\n')

#Oversampling with SMOTE technique
print('Oversampling training data...',end="")
from imblearn.over_sampling import SMOTE, ADASYN
smk = SMOTE()
#adasyn = ADASYN()

X_train_over, y_train_over = smk.fit_sample(df_train, y_train)
#X_train_over, y_train_over = adasyn.fit_sample(tmp, y_train)

X_train_over.shape, y_train_over.shape
print('ok.\n')

#Variable scaling
print('Scaling numerical variables...', end="")
scaler_train = StandardScaler()
X_train_over_scaled = X_train_over.copy()
X_train_over_scaled[models['scaling_features']] = scaler_train.fit_transform(X_train_over_scaled[models['scaling_features']])

X_test_scaled = df_test.copy()
X_test_scaled[models['scaling_features']] = models['scaler_test'].transform(X_test_scaled[models['scaling_features']])

X_train_over_scaled.to_csv(os.path.join(DATA_DIR, 'X_train_over_scaled.csv'), sep=',', index=False)
y_train_over.to_csv(os.path.join(DATA_DIR, 'y_train_over.csv'), sep=',', index=False)
X_test_scaled.to_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'), sep=',', index=False)
print('ok.\n')

#Modelling
print('Initializing models...', end="")
clf_rf = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=1994)
clf_lr = LogisticRegression(random_state=1994, class_weight='balanced')
clf_xgb = XGBClassifier(learning_rate=0.02,n_estimators=600,objective='binary:logistic', random_state=1)
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_svm = SVC(random_state=1994)
print('ok.')

over_models = [
    ('Random Forest', clf_rf),
    ('Logistic Regressor', clf_lr),
    ('XGB Classifier', clf_xgb),
    ('K-Nearest Neighbours', clf_knn),
    ('Support Vector Classifier', clf_svm)
]

from utils import trainModels, makePredictions

over_model_results = trainModels(over_models,
                                X_train_over_scaled[models['features']], 
                                y_train_over, 
                                X_test_scaled[models['features']], 
                                y_test, 
                                verbose=True)

#print(model_results)

#Saving the model
print('Saving models...', end="")
over_model_data = pd.Series({
    'oe_features': models['oe_features'],
    'ohe_features': models['ohe_features'],
    'scaling_features': models['scaling_features'],
    'ohe': models['ohe'],
    'oe': models['oe'],
    'scaler_train': scaler_train,
    'scaler_test': models['scaler_test'],
    'models': over_models,
    'models_results': over_model_results,
    'features': models['features']
})

over_model_data.to_pickle(os.path.join(MODEL_DIR, 'oversampled_models.pkl'))
print('ok.')
