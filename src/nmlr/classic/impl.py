"""
This file contains classic regression model implementations for experiments in
our recent research work.

The all model classes are based on `CLRModel` abstract class for easier
implementation and usage in experiments. They just "box" the SKLearn model
instances and the weights can be easily extracted by accessing the `model` field
in class.

All model hyperparameters can be passed to the "box" constructor.
"""

import sys

sys.path.append("..")

from typing import Union
from abc import ABC
from sklearn import linear_model, neighbors, ensemble, tree, svm, dummy
import numpy as np
import copy
from nmlr.common import CommonModel


class ClassicModel(CommonModel, ABC):
    model: Union[
        dummy.DummyRegressor,
        linear_model.LinearRegression,
        linear_model.Lasso,
        linear_model.Ridge,
        neighbors.KNeighborsRegressor,
        ensemble.RandomForestRegressor,
        tree.DecisionTreeRegressor,
        svm.SVR,
    ]

    def __init__(
        self,
        name: str,
        model: Union[
            dummy.DummyRegressor,
            linear_model.LinearRegression,
            linear_model.Lasso,
            linear_model.Ridge,
            neighbors.KNeighborsRegressor,
            ensemble.RandomForestRegressor,
            tree.DecisionTreeRegressor,
            svm.SVR,
        ],
    ):
        super().__init__(name)
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def __deepcopy__(self, memo):
        return ClassicModel(
            copy.deepcopy(self.name, memo), copy.deepcopy(self.model, memo)
        )


class Dummy(ClassicModel):
    def __init__(self, **kwargs):
        super().__init__("dummy", dummy.DummyRegressor(**kwargs))


class LinearRegression(ClassicModel):
    def __init__(self, **kwargs):
        super().__init__("linear_regression", linear_model.LinearRegression(**kwargs))


class Lasso(ClassicModel):
    def __init__(self, **kwargs):
        super().__init__("lasso", linear_model.Lasso(**kwargs))


class Ridge(ClassicModel):
    def __init__(self, **kwargs):
        super().__init__("ridge", linear_model.Ridge(**kwargs))


class KNN(ClassicModel):
    def __init__(self, **kwargs):
        super().__init__("knn", neighbors.KNeighborsRegressor(**kwargs))


class RandomForest(ClassicModel):
    def __init__(self, **kwargs):
        super().__init__("random_forest", ensemble.RandomForestRegressor(**kwargs))


class DecisionTree(ClassicModel):
    def __init__(self, **kwargs):
        super().__init__("decision_tree", tree.DecisionTreeRegressor(**kwargs))


class SVR(ClassicModel):
    def __init__(self, **kwargs):
        super().__init__("svr", svm.SVR(**kwargs))
