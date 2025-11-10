import sys

sys.path.append("..")

from typing import Sequence, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from nmlr.common import MetricDict, CommonModel


class Evaluator:
    eval_tasks: Sequence[Tuple[np.ndarray, np.ndarray]]

    def __init__(
        self,
        eval_tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
    ):
        self.eval_tasks = eval_tasks

    def _calculate_metrics(
        self, model: CommonModel, A_test: np.ndarray, b_test: np.ndarray
    ) -> MetricDict:
        b_pred = model.predict(A_test)
        # print("b_test.min = {}, b_test.max = {}".format(np.min(b_test), np.max(b_test)))
        # print("b_pred.min = {}, b_pred.max = {}".format(np.min(b_pred), np.max(b_pred)))
        return {
            "mse": mean_squared_error(b_test, b_pred),
            "mae": mean_absolute_error(b_test, b_pred),
            "r2": r2_score(b_test, b_pred),
        }

    def _current_error(
        self, model: CommonModel, A_test: np.ndarray, b_test: np.ndarray
    ) -> Sequence[float]:
        b_pred = model.predict(A_test)
        b_error = b_test - b_pred
        return {
            "mean": np.mean(b_error, axis=0),
            "std": np.std(b_error, axis=0, ddof=1),
            "var": np.var(b_error, axis=0, ddof=1),
            "min": np.min(b_error, axis=0),
            "max": np.max(b_error, axis=0),
            "median": np.median(b_error, axis=0),
            "b_error": b_error,
        }

    def _current_error_precentile(
        self, model: CommonModel, A_test: np.ndarray, b_test: np.ndarray
    ) -> Sequence[float]:
        b_pred = model.predict(A_test)
        b_error = np.abs(b_test - b_pred)
        b_error.sort(axis=0)
        end_idx_5 = int(len(b_error) * 0.05)
        end_idx_25 = int(len(b_error) * 0.25)
        end_idx_50 = int(len(b_error) * 0.5)
        end_idx_75 = int(len(b_error) * 0.75)
        end_idx_95 = int(len(b_error) * 0.95)
        end_idx_100 = len(b_error)
        return {
            "5%": np.mean(b_error[:end_idx_5]),
            "25%": np.mean(b_error[:end_idx_25]),
            "50%": np.mean(b_error[:end_idx_50]),
            "75%": np.mean(b_error[:end_idx_75]),
            "95%": np.mean(b_error[:end_idx_95]),
            "100%": np.mean(b_error[:end_idx_100]),
        }

    def evaluate(
        self, model: CommonModel, last_task: int
    ) -> Tuple[Sequence[MetricDict], MetricDict]:
        per_task_metrics = []
        A_cumulative = []
        b_cumulative = []

        for A_test, b_test in self.eval_tasks[: last_task + 1]:
            per_task_metrics.append(self._calculate_metrics(model, A_test, b_test))
            A_cumulative.append(A_test)
            b_cumulative.append(b_test)

        # A_test, b_test = self.eval_tasks[last_task]
        # current_error = self._current_error(model, A_test, b_test)
        # for key, value in current_error.items():
        #     print(f"Current {key}: {value}")
        # current_error = self._current_error_precentile(model, A_test, b_test)
        # print("{},{},{},{},{},{},{}".format(last_task, *current_error.values()))

        A_cumulative = np.vstack(A_cumulative)
        b_cumulative = np.concatenate(b_cumulative)
        cumulative_metrics = self._calculate_metrics(model, A_cumulative, b_cumulative)

        return per_task_metrics, cumulative_metrics
