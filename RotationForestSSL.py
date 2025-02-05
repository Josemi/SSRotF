"""
Semi-Supervised Rotation Forest (SSRotF)
Author: José Miguel Ramírez-Sanz and Álvar Arnaiz-González
Idea: José Miguel Ramírez-Sanz, David Martínez-Acha, Álvar Arnaiz-González, César García-Osorio and Juan José Rodríguez
Email: jmrsanz@ubu.es


Based on the implementation of the original Rotation Forest modifications made by José Luis Garrido-Labrador
Implementation based on:
https://github.com/alan-turing-institute/sktime/blob/cc91ba9591aa88cba3874365782951745cd5ad6d/sktime/classification/sklearn/_rotation_forest.py
"""

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, MetaEstimatorMixin, is_classifier
from abc import abstractmethod, ABCMeta
from sklearn.base import clone as skclone
from sklearn.utils import check_X_y, check_random_state, check_array
from sklearn.decomposition import PCA

import sys
from SSLTree import SSLTree


class RotationTransformerSSL(TransformerMixin):

    def __init__(self, min_group=3, max_group=3, remove_proportion=0.5, remove_proportion_ssl=0.9, n_jobs=None, random_state=None):
        """
        Tranformer that rotates the features of a dataset.

        Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006).
        Rotation forest: A new classifier ensemble method.
        IEEE transactions on pattern analysis and machine intelligence,
        28(10), 1619-1630.

        Parameters
        ----------
        min_group : int, optional
            Minimum size of a group of attributes, by default 3
        max_group : int, optional
            Maximum size of a group of attributes, by default 3
        remove_proportion : float, optional
            Proportion of instances to be removed, by default 0.5
        remove_proportion_ssl : float, optional
            Proportion of instances to be removed for the semi-supervised learning, by default 0.9
        n_jobs : int, optional
            The number of jobs to run in parallel for both `fit` and `predict`.
            `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """

        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.remove_proportion_ssl = remove_proportion_ssl
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the transformer according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,), default=None
            The target values (class labels) if supervised.
            If None, unsupervised learning is assumed.

        Returns
        -------
        self : RotationTranformer
            Returns self.
        """        
        X = check_array(X)
        self.n_features_ = X.shape[1]
        if y is not None:
            self.classes_ = np.unique(y)
            self.classes_ = self.classes_[self.classes_ != -1]
            self.n_classes_ = self.classes_.shape[0]
            X_cls_split = [X[np.where(y == i)] for i in self.classes_]
            #Add the unlabeled data
            X_cls_split.append(X[np.where(y == -1)])

        else:
            X_cls_split = None
        self.n_jobs_ = effective_n_jobs(self.n_jobs)
        self.random_state_ = check_random_state(self.random_state)
        
        self.groups_ = self._generate_groups(self.random_state_)
        self.pcas_ = self._generate_pcas(
            X_cls_split, self.groups_, self.random_state_)

        return self

    def transform(self, X):
        """
        Transform the data according to the fitted transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_t : array-like of shape (n_samples, n_features)
            The transformed samples.
        """
        X_t = np.concatenate(
            [self.pcas_[i].transform(X[:, group]) for i, group in enumerate(self.groups_)], axis=1
        )
        return X_t

    def _generate_pcas(self, X_cls_split, groups, rng):
        pcas = []
        for group in groups:
            classes = rng.choice(
                range(self.n_classes_),
                size=rng.randint(1, self.n_classes_ + 1),
                replace=False,
            )
            
            #Get X_cls_split for the group
            X_cls_split_group = [x[:,group] for x in X_cls_split]

            #Calculate centroids for X_labeled
            centroids = {}
            for classe in self.classes_:
                centroids[classe] = np.mean(X_cls_split_group[classe], axis=0)

            #Calculate pseudo-labels for X_unlabeled based on distance to centroids, can be taken out of the loop
            pseudo_y = []
            for xu in X_cls_split_group[-1]:
                min_dis = float('inf')
                min_class = -1
                for classe in self.classes_: 
                    dis = np.linalg.norm(xu - centroids[classe])
                    if min_dis > dis:
                        min_dis = dis
                        min_class = classe
                pseudo_y.append(min_class)

            X_cls_split_group_unlabeled = [X_cls_split_group[-1][np.where(pseudo_y == i)] for i in self.classes_]   

            X_t = []
            X_t_ssl = []

            if X_cls_split is not None:
                # randomly add the classes with the randomly selected attributes.
                for cls_idx in classes:
                    for instance in X_cls_split_group[cls_idx]:
                        X_t.append(instance)
                    for instance in X_cls_split_group_unlabeled[cls_idx]:
                        X_t_ssl.append(instance)
            X_t = np.array(X_t)
            X_t_ssl = np.array(X_t_ssl)
            sample_ind = rng.choice(
                X_t.shape[0],
                max(1, int(X_t.shape[0] * (1-self.remove_proportion))),
                replace=False,
            )
            X_t = X_t[sample_ind]


            #print("X_t_ssl: ", X_t_ssl.shape)
            if X_t_ssl.shape[0] > 0:
                sample_ind_ssl = rng.choice(
                    X_t_ssl.shape[0],
                    max(1, int(X_t_ssl.shape[0] * (1-self.remove_proportion_ssl))),
                    replace=False,
                )
                X_t_ssl = X_t_ssl[sample_ind_ssl]
                X_t = np.concatenate((X_t, X_t_ssl), axis=0)
            #else:
            #    print("X_t_ssl: ", X_t_ssl.shape, " X_t: ", X_t.shape, "classes: ", classes, "group: ", group)

                

            #print("X_t:", X_t.shape, " X_t_ssl: ", X_t_ssl.shape)
            # try to fit the PCA if it fails, remake it, and add 10 random data
            # instances.
            while True:
                # ignore err state on PCA because we account if it fails.
                with np.errstate(divide="ignore", invalid="ignore"):
                    # differences between os occasionally. seems to happen when there
                    # are low amounts of cases in the fit
                    pca = PCA(random_state=rng.randint(1, 255)).fit(X_t)

                if not np.isnan(pca.explained_variance_ratio_).all():
                    break
                X_t = np.concatenate(
                    (X_t, rng.random_sample((10, X_t.shape[1]))), axis=0
                )

            pcas.append(pca)
        return pcas

    def _generate_groups(self, rng):
        """Generate random groups of subspaces. The size of each group is randomly selected between
        min_group and max_group. If the number of features is not divisible by the size of the group,
        the last group will have repeated random attributes added to it.

        Parameters
        ----------
        rng : RandomState
            Random state.

        Returns
        -------
        list
            List of groups of subspaces.
        """        
        # FROM: https://github.com/alan-turing-institute/sktime/blob/cc91ba9591aa88cba3874365782951745cd5ad6d/sktime/classification/sklearn/_rotation_forest.py#L488
        permutation = rng.permutation((np.arange(0, self.n_features_)))

        # select the size of each group.
        group_size_count = np.zeros(self.max_group - self.min_group + 1)
        n_attributes = 0
        n_groups = 0
        while n_attributes < self.n_features_:
            n = rng.randint(group_size_count.shape[0])
            group_size_count[n] += 1
            n_attributes += self.min_group + n
            n_groups += 1

        groups = []
        current_attribute = 0
        current_size = 0
        for i in range(0, n_groups):
            while group_size_count[current_size] == 0:
                current_size += 1
            group_size_count[current_size] -= 1

            n = self.min_group + current_size
            groups.append(np.zeros(n, dtype=int))
            for k in range(0, n):
                if current_attribute < permutation.shape[0]:
                    groups[i][k] = permutation[current_attribute]
                else:
                    groups[i][k] = permutation[rng.randint(
                        permutation.shape[0])]
                current_attribute += 1

        return groups


class RotationTreeSSL(BaseEstimator, MetaEstimatorMixin):
    ___metaclass__ = ABCMeta

    def __init__(self, base_estimator, min_group=3, max_group=3, remove_proportion=0.5, remove_proportion_ssl=0.9, n_jobs=None, random_state=None):
        """
        This is a rotation tree. It is a tree that uses a rotation forest to generate the ensemble. 
        Do not use this class directly, use the derived classes instead.
        
        Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006).
        Rotation forest: A new classifier ensemble method.
        IEEE transactions on pattern analysis and machine intelligence,
        28(10), 1619-1630.

        Parameters
        ----------
        base_estimator : BaseEstimator
            Estimator to use for the rotation tree.
        min_group : int, optional
            Minimum size of a group of attributes, by default 3
        max_group : int, optional
            Maximum size of a group of attributes, by default 3
        remove_proportion : float, optional
            Proportion of instances to be removed, by default 0.5
        remove_proportion_ssl : float, optional
            Proportion of instances to be removed for the semi-supervised learning, by default 0.9
        n_jobs : int, optional
            The number of jobs to run in parallel for both `fit` and `predict`.
            `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        self.base_estimator = base_estimator
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.remove_proportion_ssl = remove_proportion_ssl
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, **kwards):
        """
        Fit the rotation tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : RotationTreeSSL
            Fitted estimator.
        """        
        X, y = check_X_y(X, y)
        self.n_features_ = X.shape[1]
        self.random_state_ = check_random_state(self.random_state)
        self.rotation_ = RotationTransformerSSL(
            min_group=self.min_group,
            max_group=self.max_group,
            n_jobs=self.n_jobs,
            remove_proportion=self.remove_proportion,
            remove_proportion_ssl=self.remove_proportion_ssl,
            random_state=self.random_state_)

        self.rotation_.fit(X, y)
        X_labeled = X[y != y.dtype.type(-1)]
        y_labeled = y[y != y.dtype.type(-1)]

        X_transformed = self.rotation_.transform(X_labeled)


        self.estimator_ = skclone(self.base_estimator).fit(
            X_transformed, y_labeled, **kwards)

        return self

    @abstractmethod
    def predict(self, X):
        pass


class RotationTreeSSLClassifier(RotationTreeSSL, ClassifierMixin):

    def __init__(self, base_estimator=SSLTree(max_depth=100), min_group=3, max_group=3, remove_proportion=0.5, remove_proportion_ssl=0.9, random_state=None):
        super(RotationTreeSSLClassifier, self).__init__(
            base_estimator=base_estimator,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            remove_proportion_ssl=remove_proportion_ssl,
            random_state=random_state,
        )
        self._estimator_type = "classifier"

    def fit(self, X, y, **kwards):
        super(RotationTreeSSLClassifier, self).fit(X, y, **kwards)
        #self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """        
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the classes corresponds to that in the attribute `classes_`.
        """
        X = check_array(X)
        X_transformed = self.rotation_.transform(X)
        return self.estimator_.predict_proba(X_transformed)


class RotationTreeSSLRegressor(RotationTreeSSL, RegressorMixin):

    def __init__(self, base_estimator=SSLTree(max_depth=100), min_group=3, max_group=3, remove_proportion=0.5, remove_proportion_ssl=0.9, random_state=None):
        super(RotationTreeSSLRegressor, self).__init__(
            base_estimator=base_estimator,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            remove_proportion_ssl=remove_proportion_ssl,
            random_state=random_state,
        )

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = check_array(X)
        X_transformed = self.rotation_.transform(X)
        return self.estimator_.predict(X_transformed)


class RotationForestSSL(BaseEstimator):
    __metaclass__ = ABCMeta
    
    def __init__(self, base_estimator, n_estimators=100, min_group=3, max_group=3, remove_proportion=0.5, remove_proportion_ssl=0.9, random_state=None, n_jobs=None):
        """
        This is a rotation forest.
        Do not use this class directly, use the derived classes instead.
        
        Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006).
        Rotation forest: A new classifier ensemble method.
        IEEE transactions on pattern analysis and machine intelligence,
        28(10), 1619-1630.

        Parameters
        ----------        
        base_estimator : BaseEstimator
            Estimator to use for the rotation tree.
        n_estimators : int, optional
            number of trees, by default 100
        min_group : int, optional
            Minimum size of a group of attributes, by default 3
        max_group : int, optional
            Maximum size of a group of attributes, by default 3
        remove_proportion : float, optional
            Proportion of instances to be removed, by default 0.5
        remove_proportion_ssl : float, optional
            Proportion of instances to be removed for the semi-supervised learning, by default 0.9
        n_jobs : int, optional
            The number of jobs to run in parallel for both `fit` and `predict`.
            `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.remove_proportion_ssl = remove_proportion_ssl
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y,**kwards):
        rs = check_random_state(self.random_state)
        X, y = check_X_y(X, y)
        if is_classifier(self):
            self.classes_ = np.unique(y)
            if -1 in np.unique(y):
                self.classes_ = self.classes_[self.classes_ != -1]
            self._class_dictionary = {}
            for index, classVal in enumerate(self.classes_):
                self._class_dictionary[classVal] = index
            self.base_ = RotationTreeSSLClassifier(
                self.base_estimator,
                self.min_group,
                self.max_group,
                self.remove_proportion,
                self.remove_proportion_ssl
            )
        else:
            self.base_ = RotationTreeSSLRegressor(
                self.base_estimator,
                self.min_group,
                self.max_group,
                self.remove_proportion,
                self.remove_proportion_ssl
            )

        # Remove useless attributes
        self._useful_atts = ~np.all(X[1:] == X[:-1], axis=0)
        X = X[:, self._useful_atts]
        # Normalize attributes
        self._min = X.min(axis=0)
        self._ptp = X.max(axis=0) - self._min
        X = (X - self._min) / self._ptp

        self.n_jobs_ = min(effective_n_jobs(self.n_jobs), self.n_estimators)
        self.trees_ = Parallel(n_jobs=self.n_jobs_)(
            delayed(self._fit_estimator)(
                skclone(self.base_),
                X,
                y,
                rs.randint(np.iinfo(np.int32).max),
                **kwards
            )
            for i in range(self.n_estimators)
        )
        return self

    def _fit_estimator(self, estimator, X, y, random_state, **kwards):
        random_state = check_random_state(random_state)

        to_set = {}
        for key in sorted(estimator.get_params(deep=True)):
            if key == "random_state" or key.endswith("__random_state"):
                to_set[key] = random_state.randint(np.iinfo(np.int32).max)

        if to_set:
            estimator.set_params(**to_set)

        return estimator.fit(X, y, **kwards)

    def predict(self, X):
        if is_classifier(self):
            rng = check_random_state(self.random_state)

            return np.array(
                [
                    self.classes_[
                        int(rng.choice(np.flatnonzero(prob == prob.max())))]
                    for prob in self.predict_proba(X)
                ]
            )
        else:
            return self.predict_proba(X)

    def predict_proba(self, X):
        X = check_array(X)
        X = X[:, self._useful_atts]
        X = (X - self._min) / self._ptp

        y_probas = Parallel(n_jobs=self.n_jobs_)(
            delayed(self._predict_proba_for_estimator)(
                X,
                self.trees_[i],
            )
            for i in range(self.n_estimators)
        )

        # y_probas = [self._predict_proba_for_estimator(X, self.trees_[i]) for i in range(self.n_estimators) ]

        if is_classifier(self):
            output = np.sum(y_probas, axis=0) / (
                np.ones(len(self.classes_)) * self.n_estimators
            )
        else:
            output = np.mean(y_probas, axis=0)
        return output

    def _predict_proba_for_estimator(self, X, estimator):

        if is_classifier(self):
            probas = estimator.predict_proba(X)
            if probas.shape[1] != len(self.classes_):
                new_probas = np.zeros((probas.shape[0], len(self.classes_)))
                for i, cls in enumerate(estimator.classes_):
                    cls_idx = self._class_dictionary[cls]
                    new_probas[:, cls_idx] = probas[:, i]
                probas = new_probas
        else:
            probas = estimator.predict(X)

        return probas


class RotationForestSSLClassifier(RotationForestSSL, ClassifierMixin):
    def __init__(self, base_estimator=SSLTree(max_depth=100), n_estimators=100, min_group=3, max_group=3, remove_proportion=0.5, remove_proportion_ssl=0.9, random_state=None, n_jobs=None):
        super(RotationForestSSLClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            remove_proportion_ssl=remove_proportion_ssl,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._estimator_type = "classifier"


    def fit(self, X, y, **kwards):
        super(RotationForestSSLClassifier, self).fit(X, y, **kwards)
        #self.classes_ = np.unique(y)
        return self


class RotationForestSSLRegressor(RotationForestSSL, RegressorMixin):

    def __init__(self, base_estimator=SSLTree(max_depth=100), n_estimators=100, min_group=3, max_group=3, remove_proportion=0.5, remove_proportion_ssl=0.9, random_state=None, n_jobs=None):
        super(RotationForestSSLRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            remove_proportion_ssl=remove_proportion_ssl,
            random_state=random_state,
            n_jobs=n_jobs,
        )
