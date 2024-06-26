# experiments

`experiments` folder contains code of experiments. Each file in the experiment folder represents a certain type of
benchmark specific to a project. Such experiment can be instantiated with a certain dataset and a certain algorithm.

You should create a new `.py` file for your experiment,  
inherent from any suitable base classes in `experiments/exp_base.py`,
and then register your new experiment in `experiments/__init__.py`.

You run an experiment by running `python -m main [options]` in the root directory of the
project. You should not log any data in this folder, but storing them under `outputs` under root project
directory.

This folder is only intend to contain formal experiments. For debug code and unit tests, put them under `debug` folder.
For scripts that's not meant to be an experiment please use `scripts` folder.
