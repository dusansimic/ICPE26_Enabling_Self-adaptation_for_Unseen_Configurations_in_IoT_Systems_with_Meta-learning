"""
This file contains utility functions for experiments in our recent research
work.

- `Task` class implements a data class for storing portion of a dataset which
  corresponds to one learning task.
- `dict_to_task` function is used for converting loaded list of dictionaries to
  a list of `Task` instances.
- `load_tasks` function loads a already prepackaged dataset which is ready for
  usage and is split by tasks.
- `stack_tasks` function stacks multiple `Task` instances into one. This is used
  specifically for classic model implementations.
- `fit_aggregate` function trains a model on a list of tasks which is stacked
  before training.
"""

import sys

sys.path.append("..")

from typing import Sequence, Tuple
import numpy as np
import pickle
from nmlr.common import OldTask, Task


def __dict_to_task(dictionary: dict) -> OldTask:
    return OldTask(**dictionary)


def load_tasks(file: str) -> Sequence[OldTask]:
    tasks: Sequence[dict] = pickle.load(open(file, "rb"))
    tasks: Sequence[OldTask] = [__dict_to_task(task) for task in tasks]
    return tasks


def get_train_and_eval_tasks(
    tasks: Sequence[OldTask],
) -> Tuple[
    Sequence[Tuple[np.ndarray, np.ndarray]], Sequence[Tuple[np.ndarray, np.ndarray]]
]:
    train_tasks = []
    eval_tasks = []
    for task in tasks:
        train_tasks.append((task.X_train, task.y_train))
        eval_tasks.append((task.X_test, task.y_test))
    return train_tasks, eval_tasks


def stack_tasks(
    task_list: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    X_train, y_train = [], []
    for X_train_task, y_train_task in task_list:
        X_train.append(X_train_task)
        y_train.append(y_train_task)

    return np.vstack(X_train), np.concatenate(y_train)


def get_train_and_eval_tasks_new(
    tasks: Sequence[Task], test_size: float = 0.2
) -> Tuple[
    Sequence[Tuple[np.ndarray, np.ndarray]], Sequence[Tuple[np.ndarray, np.ndarray]]
]:
    train_tasks = []
    eval_tasks = []
    for task in tasks:
        end_idx_train = int(task.num_samples * (1 - test_size))
        train_tasks.append((task.X[:end_idx_train], task.y[:end_idx_train]))
        eval_tasks.append((task.X[end_idx_train:], task.y[end_idx_train:]))
    return train_tasks, eval_tasks
