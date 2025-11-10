import os
from nmlr.common import Task
from typing import Sequence, Literal, Tuple, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def _load_csv_dataset(
    filename: str,
    data_attributes: Sequence[str],
    ignore_attributes: Sequence[str],
    target_attributes: Sequence[str],
) -> pd.DataFrame:
    df = pd.read_csv(filename)
    if ignore_attributes is not None:
        return df.drop(columns=ignore_attributes, errors="ignore")
    else:
        return df[data_attributes + target_attributes]


def _select_upload_or_download(
    df: pd.DataFrame, upload_or_download: Literal["upload", "download"]
) -> pd.DataFrame:
    if upload_or_download == "upload":
        return df[df["throughput_UL"] != 0]
    else:
        return df[df["throughput_DL"] != 0]


def load_eab_dataset(
    filename: str,
    upload_or_download: Literal["upload", "download"],
    target_attribute: Literal[
        "throughput_DL", "throughput_UL", "delay_mean_DL", "delay_mean_UL"
    ],
    data_attributes: Sequence[str] = None,
    ignore_attributes: Sequence[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if upload_or_download not in ["upload", "download"]:
        raise ValueError("upload_or_download must be 'upload' or 'download'")

    if ignore_attributes is not None and not isinstance(ignore_attributes, Sequence):
        raise ValueError("ignore_attributes must be a sequence of strings")

    if data_attributes is not None and not isinstance(data_attributes, Sequence):
        raise ValueError("data_attributes must be a sequence of strings")

    if ignore_attributes is None and data_attributes is None:
        ignore_attributes = [
            "Unnamed: 0",
            "Unnamed: 0.1",
            "Unnamed: 0.2",
            "ANOMALY",
            "unix",
            "timestamp",
            "room_id",
            "room_derived",
        ]

    target_attributes = [
        "throughput_DL",
        "throughput_UL",
        "delay_mean_DL",
        "delay_mean_UL",
    ]
    if target_attribute not in target_attributes:
        raise ValueError(
            f"target_attribute must be one of {target_attributes}, got {target_attribute}"
        )

    df = _load_csv_dataset(
        filename, data_attributes, ignore_attributes, target_attributes
    )
    df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
    df = _select_upload_or_download(df, upload_or_download)

    data_selection_attributes = list(set(df.columns) - set(target_attributes))

    data = df[data_selection_attributes].values
    target = df[target_attribute].values

    return data, target


def scale_eab_data(data: np.ndarray):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data


def load_edict_data_scale_and_create_tasks(
    filename: str,
) -> Sequence[Task]:
    """Load a dataset from a CSV file and return it as a dictionary."""
    df = pd.read_csv(filename)
    df.replace({"latency": ""}, np.nan, inplace=True)
    df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
    data = df[df["latency"].notna() & df["latency"] != 0]

    applicationCategory_list = data["applicationCategory"].unique().tolist()
    task_list = data["task"].unique().tolist()

    task_column = "task"
    categorical_columns = ["applicationCategory"]
    y_column = "latency"
    standard_scaler_columns = list(
        set(data.columns.tolist())
        - (set(categorical_columns) | set([task_column, y_column]))
    )

    list_of_categorical_values = [applicationCategory_list]

    def create_transformers(
        data: pd.DataFrame,
        standard_scaler_columns: Sequence[str],
        categorical_columns: Sequence[str],
        y_column: str,
        list_of_categorical_values: Sequence[Sequence[Any]],
    ) -> Tuple[ColumnTransformer, StandardScaler]:
        X = data.drop(columns=[y_column])
        y = data[y_column]

        ct = ColumnTransformer(
            transformers=[
                ("numerical", StandardScaler(), standard_scaler_columns),
                (
                    "categorical",
                    OneHotEncoder(
                        sparse_output=False,
                        drop="first",
                        categories=list_of_categorical_values,
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )
        ct.fit(X)

        y_scaler = StandardScaler()
        y_scaler.fit(y.values.reshape(-1, 1))

        return ct, y_scaler

    ct, y_scaler = create_transformers(
        data,
        standard_scaler_columns,
        categorical_columns,
        y_column,
        list_of_categorical_values,
    )

    y = data[y_column]
    scaled = y_scaler.transform(y.values.reshape(-1, 1)).ravel()

    tasks = []

    for task in task_list:
        task_data = data[data["task"].isin([task])]

        X = task_data.drop(columns=[task_column, y_column])
        X_columns = X.columns.to_list()

        y = task_data[y_column]

        X_df = pd.DataFrame(X, columns=X_columns)
        X_df = ct.transform(X_df)
        y_df = pd.DataFrame(y, columns=["latency"])
        y_df = y_scaler.transform(y_df)

        tasks.append(Task(name=task, X=X_df, y=y_df.ravel()))

    return tasks


def load_edict11_prescaled_data(filename: str) -> Sequence[Task]:
    tasks = pickle.load(open(filename, "rb"))

    task_instances = []

    for i, (X, y) in enumerate(tasks):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)

        task_instances.append(Task(name=f"task_{i}", X=X, y=y.ravel()))

    return task_instances


def split_tasks_synthetic(
    data: np.ndarray,
    target: np.ndarray,
    n_tasks: int,
) -> Sequence[Task]:
    data_parts = np.array_split(data, n_tasks)
    target_parts = np.array_split(target, n_tasks)

    tasks = []

    for n in range(n_tasks):
        task = Task(name=f"task_{n}", X=data_parts[n], y=target_parts[n])
        tasks.append(task)

    return tasks


def shuffle_task_samples(tasks: Sequence[Task]) -> Sequence[Task]:
    """Shuffle the samples in each task."""
    shuffled_tasks = []
    for task in tasks:
        idxs = np.arange(task.num_samples)
        np.random.shuffle(idxs)
        shuffled_X = task.X[idxs]
        shuffled_y = task.y[idxs]
        shuffled_tasks.append(Task(name=task.name, X=shuffled_X, y=shuffled_y))
    return shuffled_tasks


def load_sag_dataset(
    directory: str,
    target_attribute: Sequence[str] = None,
    data_attributes: Sequence[str] = None,
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

    files = [
        f
        for f in os.listdir(directory)
        if f.endswith(".csv") and f.startswith("output_")
    ]
    files.sort()

    if target_attribute is None:
        target_attribute = [
            'node_network_transmit_bytes_total{app="other-apps",device="eth1",instance="172.16.3.86:9100",job="otel-collector"}'
        ]

    if data_attributes is None:
        data_attributes = [
            'variance_delta_time_ratio{exported_job="devx/delta-time-monitoring",instance="172.16.3.242:8889",job="otel-collector",label1="prometheus-exporter"}',
            'delta_time_milliseconds{exported_job="devx/delta-time-monitoring",instance="172.16.3.242:8889",job="otel-collector",label1="prometheus-exporter"}',
            'latest_delta_time_milliseconds{exported_job="devx/delta-time-monitoring",instance="172.16.3.242:8889",job="otel-collector",label1="prometheus-exporter"}',
            'max_delta_time_milliseconds{exported_job="devx/delta-time-monitoring",instance="172.16.3.242:8889",job="otel-collector",label1="prometheus-exporter"}',
            'min_delta_time_milliseconds{exported_job="devx/delta-time-monitoring",instance="172.16.3.242:8889",job="otel-collector",label1="prometheus-exporter"}',
            'packet_count_ratio{exported_job="devx/delta-time-monitoring",instance="172.16.3.242:8889",job="otel-collector",label1="prometheus-exporter"}',
        ]

    ndarray_list = []

    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        df = df.dropna(subset=target_attribute + data_attributes)

        X = df[data_attributes].values
        y = df[target_attribute].values

        ndarray_list.append((X, y))

    return ndarray_list


def ndarray_list_to_tasks(
    ndarray_list: Sequence[Tuple[np.ndarray, np.ndarray]],
    task_names: Sequence[str] = None,
) -> Sequence[Task]:
    tasks = []
    for i, (X, y) in enumerate(ndarray_list):
        if task_names is None:
            name = f"task_{i}"
        else:
            name = task_names[i]
        tasks.append(Task(name=name, X=X, y=y))
    return tasks


def load_eab_2_dataset(
    filename: str,
    target_attribute: Literal["throughput (MB/s)", "Latency(ms)"],
    data_attributes: Sequence[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_attribute not in ["throughput (MB/s)", "Latency(ms)"]:
        raise ValueError("target_attribute must be 'throughput' or 'Latency'")

    df = pd.read_excel(filename)

    if data_attributes is None:
        data_attributes = [
            "header.stamp.sec",
            "header.stamp.nanosec",
            "pose.pose.position.x",
            "pose.pose.position.y",
            "pose.pose.position.z",
            "pose.pose.orientation.x",
            "pose.pose.orientation.y",
            "pose.pose.orientation.z",
            "pose.pose.orientation.w",
            "twist.twist.linear.x",
            "twist.twist.linear.y",
            "twist.twist.linear.z",
            "twist.twist.angular.x",
            "twist.twist.angular.y",
            "twist.twist.angular.z",
            "battery_percentage",
        ]

    X = df[data_attributes].values
    y = df[target_attribute].values

    return X, y


def transform_sliding_window(
    tasks: Sequence[Task],
    window_size: int,
    step_size: int = 1,
) -> Sequence[Task]:
    transformed_tasks: Sequence[Task] = []
    for task in tasks:
        X, y = task.X, task.y
        num_samples = X.shape[0]
        windows = []

        for start in range(0, num_samples - window_size + 1, step_size):
            end = start + window_size
            X_window = X[start:end].flatten()
            y_window = y[start:end].reshape(-1, 1)
            # concatenate X_window and y_window
            X_window = np.concatenate((X_window, y_window[:-1].ravel()), axis=0)
            y_window = y_window[-1]  # use the last value as the target
            windows.append((X_window, y_window))

        # Turn windows into X and y arrays
        X_windows = np.array([w[0] for w in windows])
        y_windows = np.array([w[1] for w in windows]).ravel()

        transformed_tasks.append(Task(name=task.name, X=X_windows, y=y_windows))

    return transformed_tasks
