import argparse
from nmlr.classic.impl import LinearRegression, Lasso, Ridge, DecisionTree, RandomForest, SVR, KNN
from nmlr.classic.exp import Experiment
from nmlr.classic.utils import get_train_and_eval_tasks_new
from nmlr.data import load_edict11_prescaled_data, shuffle_task_samples
from nmlr.loggers import ConsoleClassicTaskProgressLogger, JSONEvalutorLogger, CSVTimingLogger

# Predefined values
DATASET_FILE = "../data/edict11_scaled_tasks.pkl"
VALID_MODEL_NAMES = ["lnr", "lss", "rdg", "dtr", "rfr", "svr"]

# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-D", "--dataset", help="dataset file path", required=False, default=DATASET_FILE)
parser.add_argument("-M", "--model", help="model short name", required=True, choices=VALID_MODEL_NAMES)

args = parser.parse_args()

print("Loading data...")
tasks = load_edict11_prescaled_data(args.dataset)
print("Shuffling data...")
tasks = shuffle_task_samples(tasks)

print("Splitting data...")
train, test = get_train_and_eval_tasks_new(tasks)

models = {
    "lnr": LinearRegression,
    "lss": Lasso,
    "rdg": Ridge,
    "dtr": DecisionTree,
    "rfr": RandomForest,
    "svr": SVR,
    "knn": KNN,
}

Model = models[args.model]
model = Model()

exp = Experiment(model, train, test)

exp.benchmark(
    time_measurement=False,
    energy_measurement=False,
    loggers=[
        ConsoleClassicTaskProgressLogger(compact=True),
        JSONEvalutorLogger(f"results/baseline_{model.name}_eval.json"),
        CSVTimingLogger(f"results/baseline_{model.name}_timing.csv"),
    ],
)
