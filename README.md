# Enabling Self-adaptation for Unseen Configurations in IoT Systems with Meta-learning

> [!NOTE]
> This code repository contains MAML model implementations referenced in the ICPE26 paper (future reference).

## Usage

To use this implementation of baseline and novel MAML model implementations, you can follow this usage guide.

The example scripts for running the experiments are located in the [scripts](./scripts/) directory where you can find two files:

1. [baseline.py](./scripts/baseline.py) which is an exmaple of how to run the baseline models,
2. [novel.py](./scripts/novel.py) which is an example of how to run the novel models.

These scripts are also implemented as commands with the `--help` command so you can have a simple reference to what parameters are needed for experiment usage.

Both scripts contain two identical parameters:

1. `--dataset` which is not a required option and sepcifies a path to the dataset file which is used for the experiment (default path is `../data/edict11_scaled_tasks.pkl`),
2. `--model` which is a required option and specifies the model that will be used for the experiment.

When running the script with the `--help` option you can see which available models can you use for the experiment at hand.

Options specific to the novel model implementation are:

1. `--epochs` which specifies the number of epochs in model training,
2. `--learning-rate` which specifies the learning rate of a model.

Both of these options specific to novel models are required.

## Example

Example of how to run a baseline model script and novel models script are show below.

```sh
$ cd scripts
$ python baseline.py -M lnr
```

```sh
$ cd scripts
$ python novel.py -M lnr -E 1000 -L 0.00000001
```
