import sys

import mlflow

sys.path.append("..")

from nmlr.classic.impl import ClassicModel
import numpy as np
from typing import Sequence, Tuple
from nmlr.loggers import _Logger
from nmlr.eval import Evaluator
from nmlr.classic.utils import stack_tasks
import time
import copy
from nmlr.common import EvalResultDict, ExperimentContext
import pyRAPL


class Experiment(ExperimentContext):
    model: ClassicModel
    train_tasks: Sequence[Tuple[np.ndarray, np.ndarray]]
    eval_tasks: Sequence[Tuple[np.ndarray, np.ndarray]]

    def __init__(
        self,
        model: ClassicModel,
        train_tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
        eval_tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
    ):
        self.model = model
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.n_tasks = len(train_tasks)

    def benchmark(
        self,
        time_measurement=True,
        energy_measurement=False,
        evaluate=True,
        loggers: Sequence[_Logger] = [],
    ):
        current_model = copy.deepcopy(self.model)

        self.time_measurement = []

        if evaluate:
            self.metrics = []
            evaluator = Evaluator(self.eval_tasks)

        pyRAPL.setup()
        meter = pyRAPL.Measurement("nmlr_experiment")
        meter.begin()

        for last_task in range(1, self.n_tasks + 1):
            X_train, y_train = stack_tasks(self.train_tasks[:last_task])

            process_start = time.process_time_ns()

            current_model.fit(X_train, y_train)

            process_end = time.process_time_ns()
            self.time_measurement.append(process_end - process_start)

            if evaluate:
                per_task_metrics, cumulative_metrics = evaluator.evaluate(
                    current_model, last_task - 1
                )
                self.metrics.append(
                    {"per_task": per_task_metrics, "cumulative": cumulative_metrics}
                )

            for logger in loggers:
                logger.log_experiment_context(self)

        meter.end()
        # print(meter.result)

        # mlflow.end_run()
