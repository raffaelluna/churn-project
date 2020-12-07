import os
import pandas as pd
import numpy as np

BASE_DIR = '~/ML-AZ/Projeto - Churn'
DATA_DIR = os.path.join(BASE_DIR, 'input')

#Loading dataset
print('Loading dataset...', end="")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
print('ok.\n')

#Outlier handling

#Handling Credit Score outliers
IQR_creditScore = X_train['CreditScore'].quantile(0.75) - X_train['CreditScore'].quantile(0.25)

cs_iqr_inf_limit = X_train['CreditScore'].quantile(0.25) - (IQR_creditScore * 1.5)
cs_iqr_sup_limit = X_train['CreditScore'].quantile(0.75) + (IQR_creditScore * 1.5)

print('Credit Score 1st Quartile - 1.5 * DIR = {}'.format(cs_iqr_inf_limit))
print('Credit Score 3rd Quartile + 1.5 * DIR = {}\n'.format(cs_iqr_sup_limit))

print('There are {} clients with more than 920.5 Credit Score'.format(
    len(X_train[X_train['CreditScore'] > cs_iqr_sup_limit])))

print('There are {} clients with less than 380.5 Credit Score\n'.format(
    len(X_train[X_train['CreditScore'] < cs_iqr_inf_limit])))

print('Handling Credit Score Outlier...', end="")
X_train['CreditScore'] = np.where(X_train['CreditScore'] < cs_iqr_inf_limit, 
                                  cs_iqr_inf_limit, X_train['CreditScore'])
print('ok.\n')

print('There are {} clients with less than 380.5 Credit Score\n'.format(
    len(X_train[X_train['CreditScore'] < cs_iqr_inf_limit])))

#Handling Age outliers
IQR_age = X_train['Age'].quantile(0.75) - X_train['Age'].quantile(0.25)

age_iqr_inf_limit = X_train['Age'].quantile(0.25) - (IQR_age * 1.5)
age_iqr_sup_limit = X_train['Age'].quantile(0.75) + (IQR_age * 1.5)

print('Age 1st Quartile - 1.5 * DIR = {}'.format(age_iqr_inf_limit))
print('Age 3rd Quartile + 1.5 * DIR = {}\n'.format(age_iqr_sup_limit))

print('There are {} clients with more than 62 years'.format(
    len(X_train[X_train['Age'] > age_iqr_sup_limit])))

print('There are {} clients with less than 14 years\n'.format(
    len(X_train[X_train['Age'] < age_iqr_inf_limit])))

print('Handling Age Outlier...', end="")
X_train['Age'] = np.where(X_train['Age'] > age_iqr_sup_limit, 
                                  age_iqr_sup_limit, X_train['Age'])
print('ok.\n')

print('There are {} clients with more than 62 years\n'.format(
    len(X_train[X_train['Age'] < age_iqr_inf_limit])))

X_train.to_csv(os.path.join(DATA_DIR, 'X_train_out.csv'), sep=',', index=False)