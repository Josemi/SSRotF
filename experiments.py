import os
import pandas as pd
import utils
import sys
import numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "50"

#limit RAM 475GB
import resource
def limit_memory(bytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (bytes, hard))

limit_memory(475*(2**30))

from sslearn.wrapper import SelfTraining
from sslearn.model_selection import artificial_ssl_dataset

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from multiprocessing import Pool

from SSLTree import SSLTree
from rotation import RotationForestClassifier
from RotationForestSSL import RotationForestSSLClassifier
from RandomForestSSL import RandomForestSSL


import warnings
warnings.filterwarnings("ignore")

#Configuration
REP = 5
KFOLD = 2
RANDOM_STATE = np.random.RandomState(33)
RANDOM_STATE_PARTITION = np.random.RandomState(333)

#Models
#SSL models
ssl_models = {"RFSSLTree": RandomForestSSL(n_estimators=100, estimator=SSLTree(max_depth=100, w=0.85, max_features="sqrt")),
              "SSLTree" : SSLTree(max_depth=100, w=0.85),
              "RotSSL75": RotationForestSSLClassifier(n_estimators=100, random_state=RANDOM_STATE, base_estimator=SSLTree(max_depth=100, w=0.75)),
              "RotSSL85": RotationForestSSLClassifier(n_estimators=100, random_state=RANDOM_STATE, base_estimator=SSLTree(max_depth=100, w=0.85)),
              "RotSSLRT85": RotationForestSSLClassifier(n_estimators=100, random_state=RANDOM_STATE, base_estimator=SSLTree(max_depth=100, w=0.85, max_features="sqrt")),
              "SelfTraining(RotF)" : SelfTraining(base_estimator=RotationForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
              "SelfTraining(DT)" : SelfTraining(base_estimator=DecisionTreeClassifier(random_state=RANDOM_STATE)),
              "SelfTraining(RF)" : SelfTraining(base_estimator=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))
              }

#SL models
sl_models = {"DT": DecisionTreeClassifier(random_state=RANDOM_STATE),
             "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
             "RotF": RotationForestClassifier(n_estimators=100, random_state=RANDOM_STATE)}


#Label proportion
label_prop = np.arange(0.05,0.2,0.05)
aux = np.arange(0.3,1,0.1)
label_prop = np.concatenate((label_prop,aux, [0.99]))

for lp in range(len(label_prop)):
    label_prop[lp] = round(label_prop[lp],3)

#Data
uci_type = ""


SAVE = ["DATA", "REP","KFOLD","MODEL","LP","ACC", "Gmacro", "Fmacro","MCC"]

datasets = utils.load_uci_datasets(uci_type)

RESULTS_FILE = ""

print("Experiment with data: ", uci_type, " with " + str(REP) + " repetitions and " + str(KFOLD) + " folds", flush=True)

args = []


def job(r, k, lp, X_train_ssl, y_train_ssl, X_test, y_test, ssl_m, sl_m, dataset):
    X_train_supervised, y_train_supervised = utils.get_labeled_data(X_train_ssl, y_train_ssl)

    res = []

    for ssl_m in ssl_models:
        model = ssl_models[ssl_m]
        model.fit(X_train_ssl, y_train_ssl)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        res.append(utils.get_result_uci(ssl_m,r,k,lp,y_test,y_pred, y_pred_proba, dataset))
        del model

    for sl_m in sl_models:
        model = sl_models[sl_m]
        model.fit(X_train_supervised, y_train_supervised)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        res.append(utils.get_result_uci(sl_m,r,k,lp,y_test,y_pred, y_pred_proba, dataset))
        del model

    res = pd.DataFrame(res, columns = SAVE)
    res.to_csv("Results/UCI/UCI_v2/aux/aux_"+ dataset + "_" + str(lp) + "_" + str(r) + "_" + str(k) + ".csv", index=False)
    del res
    del X_train_supervised
    del y_train_supervised

#Repetitions
for r in range(REP):
    #print("-> Repetition: ", str(r), flush=True)
    for dataset in datasets:
        X, y = datasets[dataset]

        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE_PARTITION)
        k=0

        #Folds
        for train_index, test_index in skf.split(X,y):
            #print("--> Fold: ", str(k), flush=True)

            #Split train/test data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            #Label proportion
            for lp in label_prop:
                lp = round(lp,3)
                #print("---> Label proportion: ", lp, flush=True)


                #Get one instance per class
                index = []
                base_X = []
                base_y = []

                for i in y_train.unique():
                    base_y.append(i)
                    aux_y = y_train[y_train == i]
                    aux_y = aux_y.sample(random_state=RANDOM_STATE_PARTITION).index
                    base_X.append(X_train.loc[aux_y].values[0])
                    index.append(aux_y[0])

                #X_train = X_train.drop(index)
                X_train.reset_index()
                #y_train = y_train.drop(index)
                y_train.reset_index()

                #SSL data
                X_train_ssl, y_train_ssl, X_unlabeled, y_unlabeled = artificial_ssl_dataset(X_train, y_train, lp, random_state=RANDOM_STATE_PARTITION)

                #Add one instance per class
                X_train_ssl = np.concatenate([X_train_ssl, np.array(base_X)])
                y_train_ssl = np.concatenate([y_train_ssl, np.array(base_y)])

                args.append((r, k, lp, X_train_ssl, y_train_ssl, X_test, y_test, ssl_models, sl_models, dataset))
            
            k+=1

with Pool(None) as pool:
    pool.starmap(job, args, chunksize=1)
    pool.close()

df_res = pd.DataFrame(columns = SAVE)

for r in range(REP):
    for dataset in datasets:
        for k in range(KFOLD):
            for lp in label_prop:
                pd_aux = pd.read_csv("")
                df_res = pd.concat([df_res, pd_aux])

df_res.to_csv(RESULTS_FILE, index=False)
print("Results saved in: ", RESULTS_FILE, flush=True)
