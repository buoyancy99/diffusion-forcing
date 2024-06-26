The `datasets` folder is used to contain dataset code or environment code.
Don't store actual data like images here! For those, please use the `data` folder instead of `datasets`.

Create a folder to create your own pytorch dataset definition. Then, update the `__init__.py`
at every level to register all datasets.

Each dataset class takes in a DictConfig file `cfg` in its `__init__`, which allows you to pass in arguments via configuration file in `configurations/dataset` or [command line override](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).
