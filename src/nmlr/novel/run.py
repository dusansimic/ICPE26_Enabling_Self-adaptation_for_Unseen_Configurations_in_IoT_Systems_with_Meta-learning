import sys
import time
from .utils import load_tasks, get_meta_train_tasks, get_meta_adapt_and_eval_tasks
from .impl import LinearRegression, Lasso, Ridge, NovelModel
from nmlr.eval import Evaluator

tasks_file = sys.argv[1].strip()
tasks = load_tasks(tasks_file)

p_init = 0.8
p_ft = 0.05
p_su = 0.8
p_qu = 0.2

record_measurements = False
print_measurements = True
evaluate_model = True

ks = [10, 50, 80]

for k in ks:
    print("running for k = " + str(k))
    meta_train_tasks = get_meta_train_tasks(tasks, p_init, p_su, k)
    meta_adapt_tasks, meta_eval_tasks = get_meta_adapt_and_eval_tasks(
        tasks, p_init, p_ft
    )

    evaluator = Evaluator(meta_eval_tasks)

    models = [
        LinearRegression,
        Lasso,
        Ridge,
    ]

    iters = 5

    for Model in models:
        for iter in range(1, iters + 1):
            model: NovelModel = Model()
            task_train_times = []

            process_start = time.process_time_ns()
            model.initialize(meta_train_tasks)
            process_end = time.process_time_ns()
            task_train_times.append(process_end - process_start)

            if print_measurements:
                print(f"  ----- time measurement -----")
                print(f"model: {model.name}")
                print(f"iteration: {iter}")
                print(f"task: meta-model train")
                print(f"measurement (ns): {process_end - process_start}")
                print()

            for i, (A_tr, b_tr) in enumerate(meta_adapt_tasks):
                process_start = time.process_time_ns()

                model.adapt(A_tr, b_tr)

                process_end = time.process_time_ns()
                task_train_times.append(process_end - process_start)

                if print_measurements:
                    print(f"  ----- time measurement -----")
                    print(f"model: {model.name}")
                    print(f"iteration: {iter}")
                    print(f"task: {i + 1}")
                    print(f"measurement (ns): {process_end - process_start}")
                    print()

                if evaluate_model:
                    _, metrics = evaluator.evaluate(model)
                    print(f"  ----- evaluation metrics -----")
                    print(f"model: {model.name}")
                    print(f"iteration: {iter}")
                    print(f"task: {i + 1}")
                    print(f"cumulative (mse): {metrics["mse"]}")
                    print(f"cumulative (mae): {metrics["mae"]}")
                    print(f"cumulative (r^2): {metrics["r2"]}")
                    print()

            if record_measurements:
                with open(
                    f"results/novel_result_{model.name}_k-{str(k)}_{str(iter).zfill(2)}.csv",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("task,process_time_ns\n")
                    for i, task_train_time in enumerate(task_train_times):
                        if i == 0:
                            f.write("meta," + str(task_train_time) + "\n")
                        else:
                            f.write(str(i) + "," + str(task_train_time) + "\n")
