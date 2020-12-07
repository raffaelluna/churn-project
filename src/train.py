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
print('ok.')

BASE_DIR = '~/ML-AZ/Projeto - Churn'
DATA_DIR = os.path.join(BASE_DIR, 'input')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

#Loading dataset
print('Loading dataset...', end="")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_out.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
print('ok.\n')

#Feature Encoding and Scaling
oe_features = ['Gender']
ohe_features = ['Geography']
scaling_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

print('Preprocessing...', end="")
oe = OrdinalEncoder()
oe.fit(X_train[oe_features])
X_train[oe_features] = oe.transform(X_train[oe_features])
X_test[oe_features] = oe.transform(X_test[oe_features])

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe.fit(X_train[ohe_features])
ohe_df = pd.DataFrame(ohe.transform(X_train[ohe_features]), columns=ohe.get_feature_names(ohe_features))
ohe_df_test = pd.DataFrame(ohe.transform(X_test[ohe_features]), columns=ohe.get_feature_names(ohe_features))

df_train = pd.concat([ohe_df, X_train.drop(ohe_features, axis=1)], axis=1)
df_test = pd.concat([ohe_df_test, X_test.drop(ohe_features, axis=1)], axis=1)

features_list = df_train.columns.tolist()

scaler_train = StandardScaler()
X_train_scaled = df_train.copy()
X_train_scaled[scaling_features] = scaler_train.fit_transform(X_train_scaled[scaling_features])

scaler_test = StandardScaler()
X_test_scaled = df_test.copy()
X_test_scaled[scaling_features] = scaler_test.fit_transform(X_test_scaled[scaling_features])
print('ok.')

#Modelling
print('Initializing models...', end="")
clf_rf = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=1994)
clf_lr = LogisticRegression(random_state=1994, class_weight='balanced')
clf_xgb = XGBClassifier(learning_rate=0.02,n_estimators=600,objective='binary:logistic', random_state=1994)
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_svm = SVC(random_state=1994)
print('ok.')

models = [
    ('Random Forest', clf_rf),
    ('Logistic Regressor', clf_lr),
    ('XGB Classifier', clf_xgb),
    ('K-Nearest Neighbours', clf_knn),
    ('Support Vector Classifier', clf_svm)
]

from utils import trainModels, makePredictions

model_results = trainModels(models, 
                            X_train_scaled[features_list], 
                            y_train, 
                            X_test_scaled[features_list], 
                            y_test, 
                            verbose=True)

#print(model_results)

#Saving the model
print('Saving models...', end="")
model_data = pd.Series({
    'oe_features': oe_features,
    'ohe_features': ohe_features,
    'scaling_features': scaling_features,
    'ohe': ohe,
    'oe': oe,
    'scaler_train': scaler_train,
    'scaler_test': scaler_test,
    'models': models,
    'models_results': model_results,
    'features': features_list
})

model_data.to_pickle(os.path.join(MODEL_DIR, 'first_models.pkl'))
print('ok.')
# The results aren't good. Better try some oversampling technique. Check oversampling_train.py