print('Importing packages...', end="")
import os
import pandas as pd
print('ok.\n')

BASE_DIR = '~/ML-AZ/Projeto - Churn'
DATA_DIR = os.path.join(BASE_DIR, 'input')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print('Loading model...', end="")
reduced_model = pd.read_pickle(os.path.join(MODEL_DIR, 'reduced_model.pkl'))
print('ok.\n')

model = reduced_model['models']

Age = int(input("Entre com a idade do cliente: "))
Gender = input("Entre com o gênero, Male ou Female, do cliente: ")
Geography = input("France, Germany ou Spain? ")
NumOfProducts = int(input("Quantidade, de 1 a 4, de produtos que o cliente possui: "))
IsActiveMember = int(input("0 para cliente inativo, 1 para cliente ativo: "))

data = {
    'Geography': [Geography],
    'Gender': [Gender],
    'Age': [Age],
    'IsActiveMember': [IsActiveMember],
    'NumOfProducts': [NumOfProducts]
}

data = pd.DataFrame(data, columns=['Geography', 'Gender', 'Age', 'IsActiveMember', 'NumOfProducts'])

model_features = reduced_model['features']

oe_feature = ['Gender']
ohe_feature = ['Geography']
scaling_feature = ['Age']

data[oe_feature] = reduced_model['oe'].transform(data[oe_feature])
data[scaling_feature] = reduced_model['scaler_test'].transform(data[scaling_feature])

ohe_df = pd.DataFrame(reduced_model['ohe'].transform(data[ohe_feature]), 
                    columns=reduced_model['ohe'].get_feature_names(ohe_feature))

data_to_predict = pd.concat([ohe_df, data.drop(ohe_feature, axis=1)], axis=1)

y_pred = model.predict_proba(data_to_predict)

print('Probabilidade de Churn do cliente: ', y_pred[0,1])