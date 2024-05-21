from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
import numpy as np

# HW18
SEED = 42


class RandomForestClassifierCustom(BaseEstimator):
    """
    Custom implementation of a Random Forest Classifier.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

    max_features : int, default=None
        The number of features to consider when looking for the best split.

    random_state : int, default=SEED
        Controls the randomness of the bootstrapping of the samples used when building trees.
    """

    def __init__(
        self, n_estimators=10, max_depth=None, max_features=None, random_state=SEED
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feat_ids_by_tree = []

    def _train_single_tree(self, i, X, y):
        """
        Trains a single decision tree.

        Parameters
        ----------
        i : int
            Index of the tree being trained.

        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        tuple
            A tuple containing the feature indices used and the trained decision tree.
        """
        np.random.seed(self.random_state + i)
        n_features = np.shape(X)[1]
        n_objects = np.shape(X)[0]
        features_indices = np.random.choice(
            n_features, self.max_features, replace=False
        )
        train_indices = np.random.choice(n_objects, n_objects, replace=True)
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth, max_features=self.max_features
        )
        tree.fit(X[train_indices][:, features_indices], y[train_indices])
        return features_indices, tree

    def fit(self, X, y, n_jobs=-1):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        n_jobs : int, default=-1
            The number of jobs to run in parallel. `-1` means using all processors.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.classes_ = sorted(np.unique(y))
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._train_single_tree)(i, X, y) for i in range(self.n_estimators)
        )
        self.feat_ids_by_tree, self.trees = zip(*results)
        return self

    def _predict_proba_single_tree(self, tree, features, X):
        """
        Predict class probabilities for a single tree.

        Parameters
        ----------
        tree : DecisionTreeClassifier
            The decision tree used for prediction.

        features : array-like of shape (n_features,)
            The feature indices used by the tree.

        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        array of shape (n_samples, n_classes)
            The class probabilities predicted by the tree.
        """
        return tree.predict_proba(X[:, features])

    def predict_proba(self, X, n_jobs=-1):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as the mean predicted class probabilities of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        n_jobs : int, default=-1
            The number of jobs to run in parallel. `-1` means using all processors.

        Returns
        -------
        array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        probas = np.zeros((X.shape[0], len(self.classes_)))
        tree_probas = Parallel(n_jobs=n_jobs)(
            delayed(self._predict_proba_single_tree)(tree, features, X)
            for tree, features in zip(self.trees, self.feat_ids_by_tree)
        )
        for y_pred in tree_probas:
            probas += y_pred
        probas /= self.n_estimators
        return probas

    def predict(self, X, n_jobs=-1):
        """
        Predict class for X.

        The predicted class of an input sample is computed as the class with the highest mean predicted class probability of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        n_jobs : int, default=-1
            The number of jobs to run in parallel. `-1` means using all processors.

        Returns
        -------
        array of shape (n_samples,)
            The predicted classes.
        """
        probas = self.predict_proba(X, n_jobs=n_jobs)
        predictions = np.argmax(probas, axis=1)
        return predictions
