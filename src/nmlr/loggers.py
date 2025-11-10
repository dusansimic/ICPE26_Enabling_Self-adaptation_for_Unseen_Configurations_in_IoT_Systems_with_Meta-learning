import sys
from abc import ABC, abstractmethod
from typing import List, TextIO
import json
from nmlr.common import EvalResultDict, ExperimentContext
import mlflow
from rich.progress import (
    Progress,
    Task,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm


class _Logger(ABC):
    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def log_experiment_context(self, context: ExperimentContext):
        pass


class _FileLogger(_Logger, ABC):
    log_file: str

    def __init__(self, name: str, log_file: str):
        super().__init__(name)
        self.log_file = log_file


class _StreamLogger(_Logger, ABC):
    stream: TextIO

    def __init__(self, name: str, stream: TextIO):
        super().__init__(name)
        self.stream = stream


class JSONEvalutorLogger(_FileLogger):
    def __init__(self, log_file: str):
        super().__init__("JSON evaluator logger", log_file)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("[]")

    def log_experiment_context(self, context):
        per_task_metrics = context.metrics[-1]["per_task"]
        cumulative_metrics = context.metrics[-1]["cumulative"]

        with open(self.log_file, "r", encoding="utf-8") as f:
            eval_results: List[EvalResultDict] = json.load(f)

        eval_results.append(
            {"per_task": per_task_metrics, "cumulative": cumulative_metrics}
        )

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(eval_results, f)


class ConsoleEvaluatorLogger(_StreamLogger):
    def __init__(self):
        super().__init__("Console evaluator logger", sys.stdout)

    def log_experiment_context(self, context):
        cumulative_metrics = context.metrics[-1]["cumulative"]
        self.stream.write(f"  --- {self.name} ---\n")
        self.stream.write(f"model: {context.model.name}\n")
        self.stream.write("cumulative (mse): " + str(cumulative_metrics["mse"]) + "\n")
        self.stream.write("cumulative (mae): " + str(cumulative_metrics["mae"]) + "\n")
        self.stream.write("cumulative (r^2): " + str(cumulative_metrics["r2"]) + "\n")
        self.stream.write("\n")


class ConsoleEvaluatorCurrentTaskPerformanceLogger(_StreamLogger):
    model_variant: str

    def __init__(self, model_variant: str):
        super().__init__(
            "Console evaluator logger for current task performance", sys.stdout
        )

        self.model_variant = model_variant

    def log_experiment_context(self, context):
        current_task = len(context.metrics)
        if self.model_variant == "novel":
            current_task -= 1

        cumulative_metrics = context.metrics[-1]["per_task"][current_task]
        self.stream.write(f"  --- {self.name} ---\n")
        self.stream.write(f"model: {context.model.name}\n")
        self.stream.write(f"task: {current_task}\n")
        self.stream.write("cumulative (mse): " + str(cumulative_metrics["mse"]) + "\n")
        self.stream.write("cumulative (mae): " + str(cumulative_metrics["mae"]) + "\n")
        self.stream.write("cumulative (r^2): " + str(cumulative_metrics["r2"]) + "\n")
        self.stream.write("\n")


class ConsoleEvaluatorSimpleLogger(_StreamLogger):
    def __init__(self):
        super().__init__("Console evaluator simple logger", sys.stdout)

    def log_experiment_context(self, context):
        cumulative_metrics = context.metrics[-1]["cumulative"]
        self.stream.write(
            f"model: {context.model.name}; mse: "
            + str(cumulative_metrics["mse"])
            + "; mae: "
            + str(cumulative_metrics["mae"])
            + "; r^2: "
            + str(cumulative_metrics["r2"])
            + "\n"
        )


class MLFlowEvaluatorLogger(_Logger):
    def __init__(self, tracking_uri: str, experiment: str, run: str):
        super().__init__("MLFlow evaluator logger")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run)

    def log_experiment_context(self, context):
        cumulative_metrics = context.metrics[-1]["cumulative"]
        mlflow.log_metrics(cumulative_metrics, step=len(context.metrics))


# timing loggers


class CSVTimingLogger(_FileLogger):
    def __init__(self, log_file):
        super().__init__("CSV timing logger", log_file)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("task,process_time_ns\n")

    def log_experiment_context(self, context):
        task_id = len(context.time_measurement)
        time_measurement = context.time_measurement[-1]
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{task_id},{time_measurement}\n")


class ConsoleTimingLogger(_StreamLogger):
    def __init__(self):
        super().__init__("Console timing logger", sys.stdout)

    def log_experiment_context(self, context):
        self.stream.write(f"  --- {self.name} ---\n")
        self.stream.write(f"model: {context.model.name}\n")
        self.stream.write(f"time: {context.time_measurement[-1]}\n")
        self.stream.write(f"\n")


# progress loggers


class ConsoleClassicTaskProgressLogger(_StreamLogger):
    compact: bool
    n_tasks: int

    def __init__(self, compact=False):
        super().__init__("Console classic task progress logger", sys.stdout)
        self.compact = compact
        self.n_tasks = 0

    def log_experiment_context(self, context):
        self.n_tasks += 1
        if self.compact:
            self.stream.write(f"tasks: {self.n_tasks}/{context.n_tasks}\n")
        else:
            self.stream.write(f"  --- {self.name} ---\n")
            self.stream.write(f"complete: {self.n_tasks}\n")
            self.stream.write(f"total tasks: {context.n_tasks}\n")
            self.stream.write(f"\n")


class TaskProgressBar(_StreamLogger):
    progress: tqdm
    n_tasks: int = -1

    def __init__(self):
        super().__init__("Task progress bar", sys.stdout)

    def log_experiment_context(self, context):
        if self.n_tasks == -1:
            self.n_tasks = context.n_tasks
            self.progress = tqdm(
                total=self.n_tasks,
                desc="Tasks",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        else:
            if context.n_tasks != self.n_tasks:
                raise ValueError(
                    "The number of tasks in the context does not match the initial number of tasks."
                )

        self.progress.update(1)
