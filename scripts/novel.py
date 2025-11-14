import argparse
from nmlr.novel.impl import LinearRegression, Lasso, Ridge
from nmlr.novel.exp import Experiment
from nmlr.novel.utils import get_meta_train_adapt_and_eval_tasks_samplewise
from nmlr.data import load_edict11_prescaled_data, shuffle_task_samples
from nmlr.loggers import (
    ConsoleEvaluatorSimpleLogger,
    JSONEvalutorLogger,
    CSVTimingLogger,
)

# Predefined variables
DATASET_FILE = "../data/edict11_scaled_tasks.pkl"
VALID_MODEL_NAMES = ["lnr", "lss", "rdg"]

# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-D", "--dataset", help="dataset file path", required=False, default=DATASET_FILE)
parser.add_argument("-M", "--model", help="model short name", required=True, choices=VALID_MODEL_NAMES)
parser.add_argument("-E", "--epochs", help="number of epochs", required=True, type=int)
parser.add_argument("-L", "--learning-rate", help="learning rate", required=True, type=float)

args = parser.parse_args()

n_epochs = args.epochs
learning_rate = args.learning_rate
model_name = args.model

print("Loading data...")
tasks = load_edict11_prescaled_data(args.dataset)
print("Shuffling data...")
tasks = shuffle_task_samples(tasks)

print("Splitting data...")
meta_train, meta_adapt, meta_eval = get_meta_train_adapt_and_eval_tasks_samplewise(
    tasks,
    n_init_tasks=30,
    p_init=0.8,
    p_finetune=0.05,
    p_support=0.8,
)

if model_name == "lnr":
    model = LinearRegression(learning_rate=learning_rate, n_epochs=n_epochs)
elif model_name == "lss":
    model = Lasso(learning_rate=learning_rate, n_epochs=n_epochs, alpha=0.1)
elif model_name == "rdg":
    model = Ridge(learning_rate=learning_rate, n_epochs=n_epochs, alpha=0.1)
else:
    raise ValueError(f"Unknown model name: {model_name}")

exp = Experiment(model, meta_train, meta_adapt, meta_eval)
exp.benchmark(
    time_measurement=False,
    energy_measurement=False,
    loggers=[
        ConsoleEvaluatorSimpleLogger(),
        JSONEvalutorLogger(f"results/novel_{model_name}_eval.json"),
        CSVTimingLogger(f"results/novel_{model_name}_timing.csv"),
    ],
)
