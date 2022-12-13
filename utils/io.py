import os.path as osp
import torch
from torch import Tensor
import numpy as np
import pandas as pd
import random
import h5py

from .func import compute_discrete_label


def retrieve_from_table(patient_ids, table_path, ret=None, level='slide', shuffle=False, 
    processing_table=None, pid_column='patient_id', time_format='origin', time_bins=4):
    assert level in ['slide', 'patient']
    assert time_format in ['origin', 'ratio', 'quantile']
    if ret is None:
        if level == 'patient':
            ret = ['pid', 'pid2sid', 'pid2label'] # for patient-level task
        else:
            ret = ['sid', 'sid2pid', 'sid2label'] # for slide-level task
    for r in ret:
        assert r in ['pid', 'sid', 'pid2sid', 'sid2pid', 'pid2label', 'sid2label']

    df = pd.read_csv(table_path, dtype={pid_column: str})
    assert_columns = [pid_column, 'pathology_id', 't', 'e']
    for c in assert_columns:
        assert c in df.columns
    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    if shuffle:
        patient_ids = random.shuffle(patient_ids)

    pid2loc = dict()
    max_time = 0.0
    for i in df.index:
        max_time = max(max_time, df.loc[i, 't'])
        _p = df.loc[i, 'patient_id']
        if _p in patient_ids:
            if _p in pid2loc:
                pid2loc[_p].append(i)
            else:
                pid2loc[_p] = [i]

    # process time format
    if time_format == 'ratio':
        df.loc[:, 't'] = 1.0 * df.loc[:, 't'] / max_time
    elif time_format == 'quantile':
        df, new_columns = compute_discrete_label(df, bins=time_bins)
        assert_columns  = [pid_column, 'pathology_id'] + new_columns
    else:
        pass

    pid, sid = list(), list()
    pid2sid, pid2label, sid2pid, sid2label = dict(), dict(), dict(), dict()
    for p in patient_ids:
        if p not in pid2loc:
            print('[Warning] Patient ID {} not found in table {}.'.format(p, table_path))
        pid.append(p)
        for _i in pid2loc[p]:
            _pid, _sid, _t, _ind = df.loc[_i, assert_columns].to_list()
            if _pid in pid2sid:
                pid2sid[_pid].append(_sid)
            else:
                pid2sid[_pid] = [_sid]
            if _pid not in pid2label:
                pid2label[_pid] = (_t, _ind)

            sid.append(_sid)
            sid2pid[_sid] = _pid
            sid2label = (_t, _ind)

    res = []
    for r in ret:
        res.append(eval(r))
    return res

def read_patch_feature(path: str, dtype:str='torch'):
    r"""Read node features from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']
    ext = osp.splitext(path)[1]

    if ext == '.h5':
        with h5py.File(path, 'r') as hf:
            nfeats = hf['features'][:]
    elif ext == '.pt':
        nfeats = torch.load(path, map_location=torch.device('cpu'))
    else:
        raise ValueError(f'Not support {ext}')

    if isinstance(nfeats, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(nfeats)
    elif isinstance(nfeats, Tensor) and dtype == 'numpy':
        return nfeats.numpy()
    else:
        return nfeats

def read_patch_coord(path: str, dtype:str='torch'):
    r"""Read patch coordinates from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']

    with h5py.File(path, 'r') as hf:
        coords = hf['coords'][:]

    if isinstance(coords, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(coords)
    else:
        return coords 

def read_datasplit_npz(path: str):
    data_npz = np.load(path)
    
    pids_train = [str(s) for s in data_npz['train_patients']]
    pids_val   = [str(s) for s in data_npz['val_patients']]
    if 'test_patients' in data_npz:
        pids_test = [str(s) for s in data_npz['test_patients']]
    else:
        pids_test = None
    return pids_train, pids_val, pids_test

def read_maxt_from_table(path: str, at_column='t'):
    df = pd.read_csv(path)
    return df[at_column].max()

def save_prediction(patient_id, y_true, y_pred, save_path):
    r"""Save surival prediction.

    Args:
        y_true (Tensor or ndarray): true labels, typically with shape [N, 2].
        y_pred (Tensor or ndarray): predicted values, typically with shape [N, 1].
        save_path (string): path to save.
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()
    assert len(patient_id) == len(y_true)
    assert len(patient_id) == len(y_pred)
    
    if y_pred.shape[1] == 1: # continuous model
        y_pred = np.squeeze(y_pred)
        y_true = np.squeeze(y_true)
        t, e = y_true[:, 0], y_true[:, 1]
        df = pd.DataFrame(
            {'patient_id': patient_id, 't': t, 'e': e, 'pred_t': y_pred}, 
            columns=['patient_id', 't', 'e', 'pred_t']
        )
    else:
        bins = y_pred.shape[1]
        y_t, y_e = y_true[:, [0]], 1 - y_true[:, [1]]
        survival = np.cumprod(1 - y_pred, axis=1)
        risk = np.sum(survival, axis=1, keepdims=True)
        arr = np.concatenate((y_t, y_e, risk, survival), axis=1) # [B, 3+BINS]
        df = pd.DataFrame(arr, columns=['t', 'e', 'risk'] + ['surf_%d' % (_ + 1) for _ in range(bins)])
        df.insert(0, 'patient_id', patient_id)
    df.to_csv(save_path, index=False)
