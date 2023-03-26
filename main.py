"""
Entry filename: main.py

Code in this file inherts from https://github.com/hugochan/IDGL as it provides a flexible way 
to configure parameters and inspect model performance. Great thanks to the author.
"""
import argparse
import yaml
import numpy as np
from collections import defaultdict, OrderedDict

from model import MyHandler
from model import BaselineHandler
from utils.func import print_config


def main(handler, config):
    model = handler(config)
    if config['semi_training']:
        metrics = model.exec_semi_sl()
    elif config['test']:
        metrics = model.exec_test()
    else:
        metrics = model.exec()
    print('[INFO] Metrics:', metrics)

def multi_run_main(handler, config):
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    configs = grid(config)
    for cnf in configs:
        print('\n')
        for k in hyperparams:
            cnf['save_path'] += '-{}_{}'.format(k, cnf[k])
        print(cnf['save_path'])
        model = handler(cnf)
        if config['semi_training']:
            metrics = model.exec_semi_sl()
        elif config['test']:
            metrics = model.exec_test()
        else:
            metrics = model.exec()
        print('[INFO] Metrics:', metrics)

    # print('[INFO] Average C-Index: {}'.format(np.mean(scores['ci'])))
    # print('[INFO] Std C-Index: {}'.format(np.std(scores['ci'])))

    # print('[INFO] Average Loss: {}'.format(np.mean(scores['loss'])))
    # print('[INFO] Std Loss: {}'.format(np.std(scores['loss'])))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', required=True, type=str, help='path to the config file')
    parser.add_argument('--handler', '-d', required=True, type=str, help='model handler (adv or base)')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args

def get_config(config_path="config/config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    print_config(config)
    if cfg['handler'] == 'adv':
        handler = MyHandler
    elif cfg['handler'] == 'base':
        handler = BaselineHandler
    else:
        handler = None
    if cfg['multi_run']:
        multi_run_main(handler, config)
    else:
        main(handler, config)
