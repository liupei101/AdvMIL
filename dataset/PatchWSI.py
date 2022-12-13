from typing import Union
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
import torch_geometric

from utils.io import retrieve_from_table, read_patch_feature, read_patch_coord
from utils.func import rearrange_coord, sampling_data

#######################################################################
# WSI dataset class: it supports for loading cluster-, graph-, and 
# patch-based WSI data at patient-level.
#######################################################################

class WSIPatch(Dataset):
    r"""A patch dataset class
    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        label_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self, patient_ids: list, patch_path: str, label_path: str, mode:str,
        read_format:str='pt', time_format='ratio', time_bins=4, ratio_sampling:Union[None,float,int]=None, **kws):
        super(WSIPatch, self).__init__()
        if ratio_sampling is not None:
            print("[dataset] Sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] Sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))

        info = ['pid', 'pid2sid', 'pid2label']
        self.pids, self.pid2sid, self.pid2label = retrieve_from_table(
            patient_ids, label_path, ret=info, time_format=time_format, time_bins=time_bins)
        self.read_path = patch_path
        assert mode in ['patch', 'cluster', 'graph']
        self.mode = mode
        self.read_format = read_format
        self.kws = kws
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        if self.mode == 'graph':
            assert 'graph_path' in kws
        self.summary()

    def summary(self):
        print(f"Dataset WSIPatch for {self.mode}: avaiable patients count {self.__len__()}")

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid   = self.pids[index]
        sids  = self.pid2sid[pid]
        label = self.pid2label[pid]
        # get all data from one patient
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)

        if self.mode == 'patch':
            feats = []
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                feats.append(read_patch_feature(full_path, dtype='torch'))

            feats = torch.cat(feats, dim=0).to(torch.float)
            return index, (feats, torch.Tensor([0])), label

        elif self.mode == 'cluster':
            cids = np.load(osp.join(self.kws['cluster_path'], '{}.npy'.format(pid)))
            feats = []
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                feats.append(read_patch_feature(full_path, dtype='torch'))
            feats = torch.cat(feats, dim=0).to(torch.float)
            cids = torch.Tensor(cids)
            assert cids.shape[0] == feats.shape[0]
            return index, (feats, cids), label

        elif self.mode == 'graph':
            feats, graphs = [], []
            from .GraphBatchWSI import GraphBatch
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                feats.append(read_patch_feature(full_path, dtype='torch'))
                full_graph = osp.join(self.kws['graph_path'],  sid + '.pt')
                graphs.append(torch.load(full_graph))
            feats = torch.cat(feats, dim=0).to(torch.float)
            graphs = GraphBatch.from_data_list(graphs, update_cat_dims={'edge_latent': 1})
            assert isinstance(graphs, torch_geometric.data.Batch)
            return index, (feats, graphs), label

        else:
            pass
