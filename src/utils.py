import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, roc_curve, precision_score, recall_score

def diagnostic_plots(df, feature):
    
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 3, 1)
    sns.distplot(df[feature], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[feature], dist="norm", plot=plt)
    plt.ylabel('RM quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[feature])
    plt.title('Boxplot')
    
    #plt.tight_layout()
    plt.show();

def makePredictions(model, X_test, y_test, verbose):
    
    result = model.predict(X_test)
    
    score = f1_score(y_test, result)
    conf_matrix = confusion_matrix(y_test, result)
    roc_auc = roc_auc_score(y_test, result)
    recall = recall_score(y_test, result)
    accu = accuracy_score(y_test, result)
    
    if verbose:
        print('Model F1-Score: {}\n'.format(score))
        print('Model Accuracy: {}\n'.format(accu))
        print('Model Recall: {}\n'.format(recall))
        print('ROC-AUC: {}\n'.format(roc_auc))
        print('Confusion Matrix:\n {}\n'.format(conf_matrix))

    return recall, accu, roc_auc

def trainModels(models, X_train, y_train, X_test, y_test, verbose=True):
    
    model_results = {}
    for model in models:

        if verbose:

            print('-' * 30)
            print('Training model: {}'.format(model[0]))
            print('Model Results:\n')

        model[1].fit(X_train, y_train)
        recall, accu, roc_auc = makePredictions(model[1], X_test, y_test, verbose)

        model_results[model[0]] = {'recall': recall, 'accuracy': accu, 'roc_auc': roc_auc}
    
    return model_results