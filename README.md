# Reduzindo a fuga de clientes com Machine Learning

A fuga de clientes é um problema que tem a capacidade de arruinar uma empresa e pode ser ocasionado por diversos fatores, como, por exemplo, preços mais interessantes em outras empresas, pouco engajamento, baixa qualidade dos produtos ofertados, entre outros. Por isso, é muito importante estar antenado à evasão de clientes e sempre buscar formas de combatê-la.

Esse artigo trouxe a elaboração de um classificador que prevê a probabilidade de um cliente entrar em Churn ou não. Passou por uma análise exploratória dos dados, engenharia de atributos, construção e otimização de modelos unitários, até a combinação desses modelos, gerando o classificador final. Esse modelo final possui 82% de acurácia, com bom desempenho na previsão das classes minoritárias, estando pronto para ser colocado em produção, entregando a probabilidade de fuga para base de dados não vistas pelo modelo ([predict.py](./src/predict.py)), e com espaço para ser ainda mais otimizado.

O modelo obtido pode atuar facilmente como um instrumento de predição de fuga de clientes, auxiliando gestores a detectar a probabilidade de um cliente romper seu contrato com a empresa, permitindo a ele elaborar estratégias de incentivo à permanência e combate à fuga de clientes, consequentemente reduzindo a queda de faturamento decorrente desse problema.

Clique [aqui](./notebooks/projeto-churn.ipynb) para acessar o notebook com todas as explicações, visualizações e resultados do projeto. Caso tenha dificuldades, também pode acessá-lo [aqui](https://nbviewer.jupyter.org/github/raffaelluna/churn-project/blob/main/projeto-churn.ipynb).

## Estrutura do Projeto

Este projeto está estruturado da seguinte forma:

* [input](./input): pasta que contém os dados de treino e de teste e a base de dados crua (churn_raw.csv)
* [models](./models): pasta que contém os modelos treinados e seus metadados no formato pickle .pkl
* [notebooks](./notebooks): pasta que contém o notebook com as visualizações produzidas e detalhamento do projeto
* [output](./output): pasta que contém a base de dados escoradas com a probabilidade de fuga do cliente
* [src](./src): pasta que contém todos os scripts python de pré-processamento, modelagem, treinamento e avaliação dos modelos

## Etapas do Projeto

Os scripts python do projeto foram executados nesta ordem:

1. Divisão do conjunto de teste e de treino: [split_data.py](./src/split_data.py)
2. Tratamento de outlier do conjunto de treino: [outlier_handling.py](./src/outlier_handling.py)
3. Pré-processamento e treinamento dos modelos preliminares: [train.py](./src/train.py)
4. Aplicação da técnica SMOTE, pré-processamento e treinamento dos modelos: [train_oversampling.py](./src/train_oversampling.py)
5. Otimização dos hiperparâmetros dos melhores modelos obtidos na etapa anterior: [hyperparameter_tuning.py](./src/hyperparameter_tuning.py)
6. Combinação dos melhores classificadores obtidos nas etapas anteriores: [train_ensembled_classifier.py](./src/train_ensembled_classifier.py)
7. Treinamento do modelo combinado com base de dados de atributos reduzidos para fins de produção: [train_reduced_model.py](./src/train_reduced_model.py)
8. Algoritmo que entrega a probabilidade de o cliente entrar em "Fuga", dado suas características: [predict.py](./src/predict.py)
