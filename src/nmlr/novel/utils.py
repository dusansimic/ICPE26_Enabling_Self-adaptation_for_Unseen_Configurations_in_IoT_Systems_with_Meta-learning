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


def get_meta_train_tasks(
    tasks: Sequence[OldTask], meta_train_size: float, p_su: float, k: int
) -> Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """returns list of meta-train tasks where each element of the list contains data for meta-train in shape
    (X_support, X_query, y_support, y_query) = (A_i_tr, A_i_ts, b_i_tr, b_i_ts)"""
    meta_train_tasks = []
    for task in tasks[:k]:
        X_train, y_train = task.X_train, task.y_train
        idxs = np.arange(len(X_train))
        idxs = np.random.permutation(idxs)
        start_idx_support = 0
        end_idx_support = start_idx_query = int(len(X_train) * meta_train_size * p_su)
        end_idx_query = int(len(X_train) * meta_train_size)

        meta_train_tasks.append(
            (
                X_train[idxs[start_idx_support:end_idx_support]],
                X_train[idxs[start_idx_query:end_idx_query]],
                y_train[idxs[start_idx_support:end_idx_support]],
                y_train[idxs[start_idx_query:end_idx_query]],
            )
        )

    return meta_train_tasks


def get_meta_adapt_and_eval_tasks(
    tasks: Sequence[OldTask], meta_train_size: float, meta_adapt_size: float
) -> Tuple[
    Sequence[Tuple[np.ndarray, np.ndarray]], Sequence[Tuple[np.ndarray, np.ndarray]]
]:
    meta_adapt_tasks = []
    meta_eval_tasks = []
    for task in tasks:
        X_train, X_test, y_train, y_test = (
            task.X_train,
            task.X_test,
            task.y_train,
            task.y_test,
        )

        idxs = np.arange(len(X_train))
        idxs = np.random.permutation(idxs)

        if meta_train_size + meta_adapt_size <= 1:
            start_idx = int(len(X_train) * meta_train_size)
            end_idx = int(len(X_train) * (meta_train_size + meta_adapt_size))

            X_adapt = X_train[idxs[start_idx:end_idx]]
            y_adapt = y_train[idxs[start_idx:end_idx]]
        else:
            start_idx = int(len(X_train) * meta_train_size)
            end_idx = int(len(X_train) * (meta_adapt_size - (1 - meta_train_size)))

            X_adapt1 = X_train[idxs[start_idx:]]
            X_adapt2 = X_train[idxs[:end_idx]]
            X_adapt = np.concatenate((X_adapt1, X_adapt2), axis=0)

            y_adapt1 = y_train[idxs[start_idx:]]
            y_adapt2 = y_train[idxs[:end_idx]]
            y_adapt = np.concatenate((y_adapt1, y_adapt2), axis=0)

        meta_adapt_tasks.append((X_adapt, y_adapt.reshape(-1, 1)))
        meta_eval_tasks.append((X_test, y_test.reshape(-1, 1)))

    return meta_adapt_tasks, meta_eval_tasks


def get_meta_train_adapt_and_eval_tasks(
    tasks: Sequence[OldTask], meta_train_size: float, meta_adapt_size: float
) -> Tuple[
    Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    Sequence[Tuple[np.ndarray, np.ndarray]],
    Sequence[Tuple[np.ndarray, np.ndarray]],
]:
    end_train_idx = int(len(tasks) * meta_train_size)
    end_adapt_idx = int(len(tasks) * (meta_train_size + meta_adapt_size))
    meta_train_tasks = []
    meta_adapt_tasks = []
    meta_eval_tasks = []

    for task in tasks[:end_train_idx]:
        X, y = (
            task.X_train,
            task.y_train,
        )
        meta_train_tasks.append((X, X, y, y))

    for task in tasks[end_train_idx:end_adapt_idx]:
        X, y = (
            task.X_train,
            task.y_train,
        )
        meta_adapt_tasks.append((X, y.reshape(-1, 1)))

    for task in tasks[end_adapt_idx:]:
        X, y = (
            task.X_train,
            task.y_train,
        )
        meta_eval_tasks.append((X, y.reshape(-1, 1)))

    return meta_train_tasks, meta_adapt_tasks, meta_eval_tasks


def get_meta_train_adapt_and_eval_full(
    tasks: Sequence[OldTask], test_size: float
) -> Tuple[
    Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    Sequence[Tuple[np.ndarray, np.ndarray]],
    Sequence[Tuple[np.ndarray, np.ndarray]],
]:
    meta_train_tasks = []
    meta_adapt_tasks = []
    meta_eval_tasks = []

    for task in tasks:
        X_train, X_test, y_train, y_test = (
            task.X_train,
            task.X_test,
            task.y_train,
            task.y_test,
        )
        meta_train_tasks.append((X_train, X_train, y_train, y_train))
        meta_adapt_tasks.append((X_train, y_train.reshape(-1, 1)))
        meta_eval_tasks.append((X_test, y_test.reshape(-1, 1)))

    return meta_train_tasks, meta_adapt_tasks, meta_eval_tasks


def get_meta_train_adapt_and_eval_tasks_samplewise(
    tasks: Sequence[Task],
    n_init_tasks: int,
    p_init: float,
    p_finetune: float,
    p_support: float,
) -> Tuple[
    Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    Sequence[Tuple[np.ndarray, np.ndarray]],
    Sequence[Tuple[np.ndarray, np.ndarray]],
]:
    """
    Splits tasks into meta-train, meta-adapt, and meta-eval sets based on the provided proportions.

    The first `n_init_tasks` tasks are used for meta-training, while all tasks are used for meta-adaptation and evaluation.

    A task is split into meta-train and fine-tune sets, where:
    - meta-train set contains a proportion `p_init` of the task samples.
    - meta-adapt set contains a proportion `p_finetune` of the task samples.

    Fine-tune set is further divided into support and query sets:
    - Support set contains a proportion `p_support` of the samples.
    """
    if p_init + p_finetune > 1:
        raise ValueError("Total percentage of init and fine-tune tasks is over 100%")

    meta_train_tasks = []
    meta_adapt_tasks = []
    meta_eval_tasks = []

    for i, task in enumerate(tasks):
        init_end_idx = int(task.num_samples * p_init)
        finetune_end_idx = int(task.num_samples * (p_init + p_finetune))

        X, y = task.X, task.y

        if i < n_init_tasks:
            X_train = X[:init_end_idx]
            y_train = y[:init_end_idx]

            support_end_idx = int(X_train.shape[0] * p_support)
            meta_train_tasks.append(
                (
                    X_train[:support_end_idx],
                    X_train[support_end_idx:],
                    y_train[:support_end_idx],
                    y_train[support_end_idx:],
                )
            )

        X_finetune = X[init_end_idx:finetune_end_idx]
        y_finetune = y[init_end_idx:finetune_end_idx]

        support_end_idx = int(X_finetune.shape[0] * p_support)
        meta_adapt_tasks.append(
            (
                X_finetune[:support_end_idx],
                y_finetune[:support_end_idx].reshape(-1, 1),
            )
        )

        X_eval = X[finetune_end_idx:]
        y_eval = y[finetune_end_idx:]

        meta_eval_tasks.append(
            (
                X_eval,
                y_eval.reshape(-1, 1),
            )
        )

    return meta_train_tasks, meta_adapt_tasks, meta_eval_tasks


def get_meta_train_adapt_and_eval_tasks_taskwise(
    tasks: Sequence[Task],
    p_init: float,
    p_adapt: float,
) -> Tuple[
    Sequence[Tuple[np.ndarray, np.ndarray]],
    Sequence[Tuple[np.ndarray, np.ndarray]],
    Sequence[Tuple[np.ndarray, np.ndarray]],
]:
    """
    Splits tasks into meta-train, meta-adapt, and meta-eval sets based on the provided proportions.

    A task is split into meta-train, meta-adapt and meta-eval sets, where:
    - meta-train set contains a proportion `p_init` of the tasks.
    - meta-adapt set contains a proportion `p_adapt` of the tasks.
    - meta-eval set contains the remaining tasks.
    """
    if p_init + p_adapt > 1:
        raise ValueError("Total percentage of init and adapt tasks is over 100%")

    if p_init + p_adapt == 1:
        raise ValueError(
            "Total percentage of init and adapt tasks is 100%, no eval tasks"
        )

    num_tasks = len(tasks)
    init_end_task = int(num_tasks * p_init)
    adapt_end_task = int(num_tasks * (p_init + p_adapt))

    meta_train_tasks = [tasks[:init_end_task]]
    meta_adapt_tasks = tasks[:init_end_task:adapt_end_task]
    meta_eval_tasks = tasks[adapt_end_task:]

    return meta_train_tasks, meta_adapt_tasks, meta_eval_tasks
