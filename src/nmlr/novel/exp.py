import mlflow
from nmlr.novel.impl import NovelModel
import numpy as np
from typing import Sequence, Tuple
from nmlr.loggers import _Logger
from nmlr.eval import Evaluator
import time
import copy
from nmlr.common import ExperimentContext
import pyRAPL


class Experiment(ExperimentContext):
    model: NovelModel
    meta_train_tasks: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    meta_adapt_tasks: Sequence[Tuple[np.ndarray, np.ndarray]]
    meta_eval_tasks: Sequence[Tuple[np.ndarray, np.ndarray]]

    def __init__(
        self,
        model: NovelModel,
        meta_train_tasks: Sequence[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
        meta_adapt_tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
        meta_eval_tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
    ):
        self.model = model
        self.meta_train_tasks = meta_train_tasks
        self.meta_adapt_tasks = meta_adapt_tasks
        self.meta_eval_tasks = meta_eval_tasks
        self.n_tasks = len(meta_adapt_tasks) + 1

    def benchmark(
        self,
        time_measurement=True,
        energy_measurement=False,
        evaluate=True,
        loggers: Sequence[_Logger] = [],
    ):
        current_model = copy.deepcopy(self.model)

        self.time_measurement = []
        self.energy_measurement = []
        self.metrics = []
        evaluator = Evaluator(self.meta_eval_tasks)

        # pyRAPL.setup()
        # meter = pyRAPL.Measurement("nmlr_experiment")

        # meter.begin()
        process_start = time.process_time_ns()
        current_model.initialize(self.meta_train_tasks)
        process_end = time.process_time_ns()

        # if time_measurement:
        self.time_measurement.append(process_end - process_start)

        # if not energy_measurement and evaluate:
        per_task_metrics, cumulative_metrics = evaluator.evaluate(
            current_model, len(self.meta_train_tasks)
        )
        self.metrics.append(
            {"per_task": per_task_metrics, "cumulative": cumulative_metrics}
        )

        # if not energy_measurement:
        for logger in loggers:
            logger.log_experiment_context(self)

        for i, (A_tr, b_tr) in enumerate(self.meta_adapt_tasks):
            process_start = time.process_time_ns()
            current_model.adapt(A_tr, b_tr)
            process_end = time.process_time_ns()

            # if time_measurement:
            self.time_measurement.append(process_end - process_start)

            if evaluate:
                per_task_metrics, cumulative_metrics = evaluator.evaluate(
                    current_model, i
                )
                self.metrics.append(
                    {"per_task": per_task_metrics, "cumulative": cumulative_metrics}
                )

            # if not energy_measurement:
            for logger in loggers:
                logger.log_experiment_context(self)

        # if energy_measurement:
        # meter.end()
        # print(meter.result)
        # print(self.energy_measurement)

        # mlflow.end_run()

        return current_model
