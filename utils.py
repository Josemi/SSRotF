import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score, log_loss
from imblearn.metrics import geometric_mean_score
from copy import copy
import os
from sklearn.preprocessing import LabelEncoder


def get_labeled_data(X,y):
    """
    Function that returns the labeled data
    Parameters
    ---------- 
    X: array-like, shape (n_samples, n_features)
        The input samples.
    y: array-like, shape (n_samples,)
        The target values.
    Returns
    -------
    X: array-like, shape (n_samples, n_features)
        The input samples.
    y: array-like, shape (n_samples,)
        The target values.
    """
    df = pd.DataFrame(copy(X))
    df['y'] = pd.Series(y)

    #Remove -1 values on y
    df = df[df['y'] != -1]

    return df.drop('y', axis=1), df['y']

def get_result_uci(model, rep, kfold, lp, y_test, y_pred, y_pred_proba, data):
    """
    Function that calculates and saves results in the file
    Parameters
    ----------
    model: string
        Name of the model
    rep: int
        Repetition number
    kfold: int
        Kfold number
    lp: int
        Label proportion
    y_test: array-like, shape (n_samples,)
        The target values.
    y_pred: array-like, shape (n_samples,)
        The predicted values.
    file_name: string
        Name of the file where the metrics will be saved
    Returns
    -------
    None
    """
    acc = accuracy_score(y_test, y_pred)
    g = geometric_mean_score(y_test, y_pred, average="macro")
    fmacro = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    return [data, rep, kfold, model, lp, acc, g, fmacro, mcc]
    

def load_uci_datasets(uci_type):
    """Get datasets from files.
    Parameters
    ----------
    uci_type : collection
        List of datasets to load.
    Returns
    -------
    dict
        Dictionary with the datasets.
    """
    print("DEBUG -", "Loading datasets...")
    datasets = {}
    route = "" + uci_type
    data_it = os.listdir(route)
    for file in data_it:
        dataset = pd.read_csv(
            os.path.join(route, file), header=None
        )
        columns = []
        for i, tp in enumerate(dataset.dtypes):
            if not np.issubdtype(tp, np.number) and i != dataset.shape[1] - 1:
                columns.append(i)
                dataset[i] = dataset[i].astype("|S")
        y = dataset.iloc[:, -1]
        if np.issubdtype(y, np.number):
            y = y + 2
        X = dataset.iloc[:, :-1]
        if len(columns) > 0:
            elements = [X[X.columns.difference(columns)]]
            for col in columns:
                elements.append(pd.get_dummies(X[col]))
            concatenated_data = pd.concat(elements, axis=1)
            X = concatenated_data

        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

        y = pd.Series(y)
        
        datasets[file.split(".")[0]] = (X, y)
    print("DEBUG -", "Done")
    return datasets
