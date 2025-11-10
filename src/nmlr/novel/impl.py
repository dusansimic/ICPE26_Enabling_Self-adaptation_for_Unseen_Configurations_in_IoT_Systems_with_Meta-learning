from abc import ABC, abstractmethod
from typing import Any, Union, Sequence, Tuple
import numpy as np
from sklearn import linear_model
from nmlr.common import CommonModel
import copy


class NovelModel(CommonModel, ABC):
    """
    Abstract class for implementing base functionalities, common methods and various static methods
    for use with novel MAML methods.
    """

    # temporary model instance
    model: Union[
        linear_model.LinearRegression,
        linear_model.Lasso,
        linear_model.Ridge,
    ]

    # actulally the model parameters (these are changed while training and adapting)
    w: Any
    intercept: Any

    # hyperparameters
    alpha_reg: float
    alpha: float
    fit_intercept: bool
    max_iter: int

    # learning configurations
    learning_rate: float
    n_epochs: int

    def __init__(
        self,
        name: str,
        model: Union[
            linear_model.LinearRegression,
            linear_model.Lasso,
            linear_model.Ridge,
        ],
        alpha_reg: float,
        alpha: float,
        fit_intercept: bool,
        max_iter: int,
        learning_rate=0.01,
        n_epochs=10,
    ):
        super().__init__(name)
        self.model = model
        self.alpha_reg = alpha_reg
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.w = None
        self.intercept = None

    @abstractmethod
    def initialize(self, meta_train_tasks):
        """
        Abstract method for initializing/training the meta-model.
        """
        pass

    def adapt(self, A_train: np.ndarray, b_train: np.ndarray):
        """
        Adapt the current model (efectivelly weights and intercept) based on the
        A_trian and b_train dataset portion.
        """
        w = self.w.copy()
        intercept = self.intercept

        for _ in range(self.n_epochs):
            grad_w = A_train.T @ (A_train @ w + intercept - b_train)
            grad_intercept = np.sum(A_train @ w + intercept - b_train)

            w -= self.learning_rate * grad_w
            intercept -= self.learning_rate * grad_intercept

        self.w = w
        self.intercept = intercept

    def _calculate_A_i_hat(
        self, task: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Calculates A_i_hat for task i"""
        A_i_tr, A_i_ts, _, _ = task
        return A_i_ts @ (np.eye(A_i_ts.shape[1]) - self.alpha * A_i_tr.T @ A_i_tr)

    def _calculate_b_i_hat(
        self, task: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Calculates b_i_hat for task i"""
        A_i_tr, A_i_ts, b_i_tr, b_i_ts = task
        # NOTE: np.linalg.multi_dot is more efficient for multiple matrix multiplications
        # return b_i_ts - self.alpha * A_i_ts @ A_i_tr.T @ b_i_tr
        return b_i_ts - self.alpha * np.linalg.multi_dot([A_i_ts, A_i_tr.T, b_i_tr])

    def _calculate_A_b(
        self,
        meta_train_tasks: Sequence[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
    ):
        A_hat_list = []
        b_hat_list = []

        for task in meta_train_tasks:
            A_i_hat = self._calculate_A_i_hat(task)
            b_i_hat = self._calculate_b_i_hat(task)

            A_hat_list.append(A_i_hat)
            b_hat_list.append(b_i_hat)

        return np.vstack(A_hat_list), np.concatenate(b_hat_list)

    def predict(self, A):
        """
        Predict regressor attribute for data A with the current model
        """
        return A @ self.w + self.intercept


class LinearRegression(NovelModel):
    def __init__(
        self,
        alpha=0.005,
        fit_intercept=True,
        learning_rate=0.001,
        n_epochs=10,
        skl_args={},
    ):
        super().__init__(
            "linear_regression",
            linear_model.LinearRegression(fit_intercept=fit_intercept, **skl_args),
            alpha_reg=0.0,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=0,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
        )

    def initialize(self, meta_train_tasks):
        A_hat, b_hat = self._calculate_A_b(meta_train_tasks)
        self.model.fit(A_hat, b_hat)
        self.w = self.model.coef_.reshape(-1, 1)
        self.intercept = self.model.intercept_ if self.fit_intercept else 0

    def __deepcopy__(self, memo):
        model = LinearRegression()
        model.name = copy.deepcopy(self.name, memo)
        model.model = copy.deepcopy(self.model, memo)
        model.w = copy.deepcopy(self.w, memo)
        model.intercept = copy.deepcopy(self.intercept, memo)
        model.alpha_reg = copy.deepcopy(self.alpha_reg, memo)
        model.alpha = copy.deepcopy(self.alpha, memo)
        model.fit_intercept = copy.deepcopy(self.fit_intercept, memo)
        model.max_iter = copy.deepcopy(self.max_iter, memo)
        model.learning_rate = copy.deepcopy(self.learning_rate, memo)
        model.n_epochs = copy.deepcopy(self.n_epochs, memo)
        return model


class Ridge(NovelModel):
    def __init__(
        self,
        alpha_reg=1.0,
        alpha=0.005,
        fit_intercept=True,
        learning_rate=0.001,
        n_epochs=10,
        skl_args={},
    ):
        super().__init__(
            "ridge",
            linear_model.Ridge(
                alpha=alpha_reg, fit_intercept=fit_intercept, **skl_args
            ),
            alpha_reg=alpha_reg,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=0,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
        )

    def initialize(self, meta_train_tasks):
        A_hat, b_hat = self._calculate_A_b(meta_train_tasks)
        self.model.fit(A_hat, b_hat)
        self.w = self.model.coef_.reshape(-1, 1)
        self.intercept = self.model.intercept_ if self.fit_intercept else 0

    def __deepcopy__(self, memo):
        model = Ridge()
        model.name = copy.deepcopy(self.name, memo)
        model.model = copy.deepcopy(self.model, memo)
        model.w = copy.deepcopy(self.w, memo)
        model.intercept = copy.deepcopy(self.intercept, memo)
        model.alpha_reg = copy.deepcopy(self.alpha_reg, memo)
        model.alpha = copy.deepcopy(self.alpha, memo)
        model.fit_intercept = copy.deepcopy(self.fit_intercept, memo)
        model.max_iter = copy.deepcopy(self.max_iter, memo)
        model.learning_rate = copy.deepcopy(self.learning_rate, memo)
        model.n_epochs = copy.deepcopy(self.n_epochs, memo)
        return model


class Lasso(NovelModel):
    def __init__(
        self,
        alpha_reg=0.01,
        alpha=0.005,
        fit_intercept=True,
        max_iter=1000,
        learning_rate=0.001,
        n_epochs=10,
        skl_args={},
    ):
        super().__init__(
            "lasso",
            linear_model.Lasso(
                alpha=alpha_reg,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                **skl_args,
            ),
            alpha_reg=alpha_reg,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
        )

    def initialize(self, meta_train_tasks):
        A_hat, b_hat = self._calculate_A_b(meta_train_tasks)
        self.model.fit(A_hat, b_hat.ravel())
        self.w = self.model.coef_.reshape(-1, 1)
        self.intercept = self.model.intercept_ if self.fit_intercept else 0

    def __deepcopy__(self, memo):
        model = Lasso()
        model.name = copy.deepcopy(self.name, memo)
        model.model = copy.deepcopy(self.model, memo)
        model.w = copy.deepcopy(self.w, memo)
        model.intercept = copy.deepcopy(self.intercept, memo)
        model.alpha_reg = copy.deepcopy(self.alpha_reg, memo)
        model.alpha = copy.deepcopy(self.alpha, memo)
        model.fit_intercept = copy.deepcopy(self.fit_intercept, memo)
        model.max_iter = copy.deepcopy(self.max_iter, memo)
        model.learning_rate = copy.deepcopy(self.learning_rate, memo)
        model.n_epochs = copy.deepcopy(self.n_epochs, memo)
        return model
