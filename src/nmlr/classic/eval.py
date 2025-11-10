from typing import Sequence, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from nmlr.classic.impl import ClassicModel
from nmlr.common import MetricDict


class Evaluator:
    eval_tasks: Sequence[Tuple[np.ndarray, np.ndarray]]

    def __init__(
        self,
        eval_tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
    ):
        self.eval_tasks = eval_tasks

    def _calculate_metrics(
        self, model: ClassicModel, A_test: np.ndarray, b_test: np.ndarray
    ) -> MetricDict:
        b_pred = model.predict(A_test)
        return {
            "mse": mean_squared_error(b_test, b_pred),
            "mae": mean_absolute_error(b_test, b_pred),
            "r2": r2_score(b_test, b_pred),
        }

    def evaluate(self, model: ClassicModel) -> Tuple[Sequence[MetricDict], MetricDict]:
        per_task_metrics = []
        A_cumulative = []
        b_cumulative = []

        for A_test, b_test in self.eval_tasks:
            per_task_metrics.append(self._calculate_metrics(model, A_test, b_test))
            A_cumulative.append(A_test)
            b_cumulative.append(b_test)

        A_cumulative = np.vstack(A_cumulative)
        b_cumulative = np.concatenate(b_cumulative)
        cumulative_metrics = self._calculate_metrics(model, A_cumulative, b_cumulative)

        return per_task_metrics, cumulative_metrics
