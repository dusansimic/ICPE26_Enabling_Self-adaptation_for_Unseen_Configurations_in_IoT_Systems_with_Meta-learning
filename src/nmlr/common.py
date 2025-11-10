from typing import TypedDict, Sequence
import numpy as np
from abc import ABC, abstractmethod
import pyRAPL


class MetricDict(TypedDict):
    mse: float
    mae: float
    r2: float


class EvalResultDict(TypedDict):
    per_task: Sequence[MetricDict]
    cumulative: MetricDict


class OldTask:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    def __init__(
        self,
        name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.name = name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class Task:
    name: str
    X: np.ndarray
    y: np.ndarray

    def __init__(self, name: str, X: np.ndarray, y: np.ndarray) -> None:
        self.name = name
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        self.X = X
        self.y = y

    @property
    def num_samples(self) -> int:
        return self.X.shape[0]


class CommonModel(ABC):
    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class ExperimentContext(ABC):
    model: CommonModel
    n_tasks: int
    time_measurement: Sequence[float]
    energy_measurement: Sequence[pyRAPL.Result]
    metrics: Sequence[EvalResultDict]


def select_regressor_column(tasks: Sequence[OldTask], column: int):
    if column >= tasks[0].y_train.shape[1] or column < 0:
        raise Exception("selected regressor column out of bounds")

    for task in tasks:
        task.y_train = task.y_train[:, column]
        task.y_test = task.y_test[:, column]
