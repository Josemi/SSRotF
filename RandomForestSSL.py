"""
RandomForestSSL.py

Author: David Martínez Acha
Email: dmacha@ubu.es / achacbmb3@gmail.com
Description: Random Forest Classifier with SSLTree (handles both labeled and unlabeled data)
"""

from copy import deepcopy
from scipy.stats import mode
import numpy as np
from sklearn.utils import check_array


class RandomForestSSL:
    def __init__(self, estimator=None, n_estimators=100, random_state=None, n_jobs=None):
        self.n_jobs_ = n_jobs
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.trees = []
        self._random_state = np.random.RandomState(random_state)

    def fit(self, X, y):
        self.trees = []
        self.classes_ = np.unique(y)
        self.classes_ = self.classes_[self.classes_ != -1]
        self.n_classes_ = self.classes_.shape[0]

        self._class_dictionary = {}
        for c in range(len(self.classes_)):
            self._class_dictionary[self.classes_[c]] = c

        labeled_indices = np.where(y != -1)[0]
        unlabeled_indices = np.where(y == -1)[0]

        for b in range(self.n_estimators):
            labeled_sample = self._random_state.choice(labeled_indices, size=len(labeled_indices), replace=True)
            unlabeled_sample = self._random_state.choice(unlabeled_indices, size=len(unlabeled_indices), replace=True)

            sample = np.concatenate((labeled_sample, unlabeled_sample))
            self._random_state.shuffle(sample)

            X_train_b = X[sample]
            y_train_b = y[sample]

            tree = deepcopy(self.estimator)
            tree.fit(X_train_b, y_train_b)
            self.trees.append(tree)

    def predict(self, X):
        y_test_hats = np.empty((len(self.trees), len(X)))
        for i, tree in enumerate(self.trees):
            y_test_hats[i] = tree.predict(X)

        y_test_hats_mode, _ = mode(y_test_hats, axis=0)

        return y_test_hats_mode.flatten()
    
    def predict_proba(self, X):
        X = check_array(X)
        #X = X[:, self._useful_atts]
        #X = (X - self._min) / self._ptp

        y_probas = []
        for i in range(self.n_estimators):
            y_probas.append(self._predict_proba_for_estimator(X, self.trees[i]))

        # y_probas = [self._predict_proba_for_estimator(X, self.trees_[i]) for i in range(self.n_estimators) ]

   
        output = np.sum(y_probas, axis=0) / (
            np.ones(len(self.classes_)) * self.n_estimators
        )

        return output
    
    def _predict_proba_for_estimator(self, X, estimator): 
        probas = estimator.predict_proba(X)
        if probas.shape[1] != len(self.classes_):
            new_probas = np.zeros((probas.shape[0], len(self.classes_)))
            for i, cls in enumerate(estimator.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = probas[:, i]
            probas = new_probas
        

        return probas