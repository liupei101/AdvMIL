from typing import Union
import sys
import os.path as osp
import torch
from torch import Tensor
import torch.distributions as dist
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def add_prefix_to_filename(path, prefix=''):
    dir_name, file_name = osp.split(path)
    file_name = prefix + '_' + file_name
    return osp.join(dir_name, file_name)

def get_kfold_pids(pids, num_fold=5, keep_pids=None, random_state=42):
    kfold_pids = []
    cur_pids = [] if keep_pids is None else keep_pids
    if num_fold <= 1:
        kfold_pids.append(cur_pids + pids)
    else:
        kfold = KFold(n_splits=num_fold, shuffle=True, random_state=random_state)
        X = np.ones((len(pids), 1))
        for _, fold_index in kfold.split(X):
            kfold_pids.append(cur_pids + [pids[_i] for _i in fold_index])
    return kfold_pids

def get_label_mask(t, c, bins):
    n = t.shape[0]
    z = (torch.arange(bins).view(1, -1) * torch.ones((n, 1))).to(t.device)
    label = torch.where(c.to(torch.bool), z > t, z == t).to(torch.float)
    label_mask = (z <= t).to(torch.int) # we ignore the location whose value is greater than t
    return label, label_mask

def get_patient_data(df:pd.DataFrame, at_column='patient_id'):
    df_gps = df.groupby('patient_id').groups
    df_idx = [i[0] for i in df_gps.values()]
    pat_df = df.loc[df_idx, :]
    pat_df = pat_df.reset_index(drop=True)
    return pat_df

def compute_discrete_label(df:pd.DataFrame, column_t='t', column_e='e', bins=4):
    # merge all T and E
    min_t, max_t = df[column_t].min(), df[column_t].max()
    df.loc[:, 'y_c'] = 1 - df.loc[:, column_e] 

    # get patient data to generate their discrete time
    pat_df = get_patient_data(df)

    # qcut for patients
    df_evt    = pat_df[pat_df[column_e] == 1]
    _, qbins  = pd.qcut(df_evt[column_t], q=bins, retbins=True, labels=False)
    qbins[0]  = min_t - 1e-5
    qbins[-1] = max_t + 1e-5

    # cut for original data
    discrete_labels, qbins = pd.cut(df[column_t], bins=qbins, retbins=True, labels=False, right=False, include_lowest=True)
    df.loc[:, 'y_t'] = discrete_labels.values.astype(int)

    return df, ['y_t', 'y_c']

def sampling_data(data, num:Union[int,float]):
    total = len(data)
    if isinstance(num, float):
        assert num < 1.0 and num > 0.0
        num = int(total * num)
    assert num < total
    idxs = np.random.permutation(total)
    idxs_sampled = idxs[:num]
    idxs_left = idxs[num:]
    data_sampled = [data[i] for i in idxs_sampled]
    data_left = [data[i] for i in idxs_left]
    return data_sampled, data_left

def rename_keys(d, prefix_name, sep='/'):
    newd = dict()
    for k, v in d.items():
        newd[prefix_name + sep + k] = v
    return newd

def collect_tensor(collector, real, fake):
    if real is not None:
        if collector['real'] is None:
            collector['real'] = real
        else:
            collector['real'] = torch.cat([collector['real'], real], dim=0)

    if fake is not None:
        if collector['fake'] is None:
            collector['fake'] = fake
        else:
            collector['fake'] = torch.cat([collector['fake'], fake], dim=0)
    
    return collector

def agg_tensor(collector, data):
    for k in data.keys():
        if k not in collector or collector[k] is None:
            collector[k] = data[k]
        else:
            collector[k] = torch.cat([collector[k], data[k]], dim=0)
    return collector

def sparse_key(d, prefixes:str=''):
    if prefixes == '':
        return d
    else:
        ret = dict()
        for k in d.keys():
            if k.startswith(prefixes):
                new_key = k.split(prefixes)[1]
                if len(new_key) < 2:
                    continue
                ret[new_key[1:]] = d[k]
        return ret

def sparse_str(s, sep='-', dtype=int):
    if type(s) != str:
        return [s]
    else:
        return [dtype(_) for _ in s.split(sep)]

def generate_noise(*dims, to_device='cpu', distribution='uniform'):
    assert distribution in ['uniform', 'gaussian']
    ones = torch.ones(dims[1:], dtype=torch.float32)
    if distribution == 'uniform':
        m = dist.uniform.Uniform(0*ones, ones)
    elif distribution == 'gaussian':
        m = dist.normal.Normal(0*ones, ones)
    else:
        raise NotImplementedError(f'{distribution} has not implemented yet.')
    data = m.sample([dims[0]]).to(to_device)
    return data

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('[setup] seed: {}'.format(seed))

def setup_device(no_cuda, cuda_id, verbose=True):
    device = 'cpu'
    if not no_cuda and torch.cuda.is_available():
        device = 'cuda' if cuda_id < 0 else 'cuda:{}'.format(cuda_id)
    if verbose:
        print('[setup] device: {}'.format(device))

    return device

# worker_init_fn = seed_worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# generator = g
def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def print_config(config, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL CONFIGURATION ****************", file=f)
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val), file=f)
    print("**************** MODEL CONFIGURATION ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def print_metrics(metrics, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL METRICS ****************", file=f)
    for key in sorted(metrics.keys()):
        val = metrics[key]
        for v in val:
            cur_key = key + '/' + v[0]
            keystr  = "{}".format(cur_key) + (" " * (20 - len(cur_key)))
            valstr  = "{}".format(v[1])
            if isinstance(v[1], list):
                valstr = "{}, avg/std = {:.5f}/{:.5f}".format(valstr, np.mean(v[1]), np.std(v[1]))
            print("{} -->   {}".format(keystr, valstr), file=f)
    print("**************** MODEL METRICS ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def plot_time_kde(y, y_hat):
    """
    y is with shape of [n, 2]; each row has t and e
    y_hat is with shape of [n, 1]; each row is an estimation of time
    """
    y = y.squeeze().numpy()
    t, e = y[:, 0], y[:, 1]
    y_hat = y_hat.squeeze().numpy()
    # KDE plotting
    fig, axis = plt.subplots(1, 3, figsize=(12, 3), tight_layout=True)
    axis[0].hist(t, bins=100, density=True, label='real_time')
    axis[0].hist(y_hat, bins=100, density=True, label='pred_time')
    axis[0].set_title('All samples')
    axis[0].legend()

    axis[1].hist(t[e==1], bins=100, density=True, label='real_time')
    axis[1].hist(y_hat[e==1], bins=100, density=True, label='pred_time')
    axis[1].set_title('Event samples')
    axis[1].legend()

    axis[2].hist(t[e==0], bins=100, density=True, label='real_time')
    axis[2].hist(y_hat[e==0], bins=100, density=True, label='pred_time')
    axis[2].set_title('Censored samples')
    axis[2].legend()

    return fig

def coord_discretization(wsi_coord: Tensor):
    """
    Coordinate Discretization.
    If the value of coordinates is too large (such as 100,000), it will need super large space 
    when computing the positional embedding of patch.
    """
    x, y = wsi_coord[:, 0].tolist(), wsi_coord[:, 1].tolist()
    sorted_x, sorted_y = sorted(list(set(x))), sorted(list(set(y))) # remove duplicates and then sort
    xmap, ymap = {v:i for i, v in enumerate(sorted_x)}, {v:i for i, v in enumerate(sorted_y)}
    nx, ny = [xmap[v] for v in x], [ymap[v] for v in y]
    res = torch.tensor([nx, ny], dtype=wsi_coord[0].dtype, device=wsi_coord[0].device)
    return res.T

def to_relative_coord(wsi_coord: Tensor):
    ref_xy, _ = torch.min(wsi_coord, dim=-2)
    top_xy, _ = torch.max(wsi_coord, dim=-2)
    rect = top_xy - ref_xy
    ncoord = wsi_coord - ref_xy
    return ncoord, ref_xy, rect

def rearrange_coord(wsi_coords, offset_coord=[1, 0], discretization=False):
    """
    wsi_coord (list(torch.Tensor)): list of all patch coordinates of one WSI.
    offset_coord (list): it is set as [1, 0] by default, which means putting WSIs horizontally.
    """
    assert isinstance(wsi_coords, list)
    ret = []
    off_coord = torch.tensor([offset_coord], dtype=wsi_coords[0].dtype, device=wsi_coords[0].device)
    top_coord = -1 * off_coord
    for coord in wsi_coords:
        if discretization:
            coord = coord_discretization(coord)
        new_coord, ref_coord, rect = to_relative_coord(coord)
        new_coord = top_coord + off_coord + new_coord
        top_coord = top_coord + off_coord + rect
        ret.append(new_coord)
    return ret

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, start_epoch=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            start_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.start_epoch = start_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_checkpoint = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss):

        self.save_checkpoint = False

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.update_score(val_loss)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.start_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_score(val_loss)
            self.counter = 0

    def if_stop(self, **kws):
        return self.early_stop

    def if_save_checkpoint(self, **kws):
        return self.save_checkpoint

    def update_score(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        self.save_checkpoint = True