import sys
import time
from .utils import load_tasks, stack_tasks, get_train_and_eval_tasks
from .impl import (
    Dummy,
    LinearRegression,
    Lasso,
    Ridge,
    KNN,
    RandomForest,
    DecisionTree,
    SVR,
    ClassicModel,
)
from nmlr.eval import Evaluator

tasks_file = sys.argv[1].strip()
tasks = load_tasks(tasks_file)
train_tasks, eval_tasks = get_train_and_eval_tasks(tasks)
n_tasks = len(tasks)

record_measurements = False
evaluate_model = False

models = [
    Dummy,
    LinearRegression,
    Lasso,
    Ridge,
    KNN,
    RandomForest,
    DecisionTree,
    SVR,
]

iters = 5

evaluator = Evaluator(eval_tasks)

for Model in models:
    for iter in range(1, iters + 1):
        model: ClassicModel = Model()
        task_train_times = []

        for last_task in range(1, n_tasks + 1):
            X_train, y_train = stack_tasks(train_tasks[:last_task])

            process_start = time.process_time_ns()

            model.fit(X_train, y_train)

            process_end = time.process_time_ns()
            task_train_times.append(process_end - process_start)
            print(f"  ----- time measurement -----")
            print(f"model: {model.name}")
            print(f"iteration: {iter}")
            print(f"task: {last_task}")
            print(f"measurement (ns): {process_end - process_start}")
            print()

            if evaluate_model:
                _, metrics = evaluator.evaluate(model)
                print(f"  ----- evaluation metrics -----")
                print(f"model: {model.name}")
                print(f"iteration: {iter}")
                print(f"task: {last_task}")
                print(f"cumulative (mse): {metrics["mse"]}")
                print(f"cumulative (mae): {metrics["mae"]}")
                print(f"cumulative (r^2): {metrics["r2"]}")
                print()

        if record_measurements:
            with open(
                f"results/classic_result_{model.name}_{str(iter).zfill(2)}.csv",
                "w",
                encoding="utf-8",
            ) as f:
                f.write("task,process_time_ns\n")
                for i, task_train_time in enumerate(task_train_times):
                    f.write(str(i + 1) + "," + str(task_train_time) + "\n")
