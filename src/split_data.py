import os
import pandas as pd

from sklearn.model_selection import train_test_split

BASE_DIR = '~/ML-AZ/Projeto - Churn'
DATA_DIR = os.path.join(BASE_DIR, 'input')

df = pd.read_csv(os.path.join(DATA_DIR, 'churn_raw.csv'))

X_train, X_test, y_train, y_test = train_test_split(
                                        df.drop(['RowNumber','CustomerId','Surname', 'Exited'], axis=1), 
                                        df['Exited'], 
                                        test_size=0.15,  
                                        random_state=0) 

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print('X_train Shape: ', X_train.shape)
print('X_test Shape: ', X_test.shape)
print('y_train Shape: ', y_train.shape)
print('y_test Shape: ', y_test.shape)

X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), sep=',', index=False)
X_test.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), sep=',', index=False)
y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), sep=',', index=False)
y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), sep=',', index=False)