"""
This script is used to prepare Graphs for the graph-based PatchGCN,
which follows PatchGCN's original paper.

The second step of graph construction.
"""
import sys
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import h5py
import torch
from torch_geometric.data import Data as geomData
from itertools import chain
import nmslib


#PATHIDS_TO_PROCESS = ['10848'] # 10848 pathology_id respective to the patient with ID 128599 
PATHIDS_TO_PROCESS = []
ROOT_DIR = '/hdd/liup/data/WSI/{}/processed'
COOR_READ_DIR = 'hier-x20-tiles-s256/patches'
FEAT_READ_DIR = 'feat-x20-RN50-B-color_norm/pt_files'
SAVE_DIR = 'wsigraph-x20-features'


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

def pt2graph(fpath_h5, fpath_pt, fpath_save, radius=9):
    # read coords
    wsi_h5 = h5py.File(fpath_h5, "r")
    coords = wsi_h5['coords'][:]
    wsi_h5.close()
    # read feats
    features = torch.load(fpath_pt).numpy()
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]
    
    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),dtype=int)
    edge_spatial = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)
    
    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),dtype=int)
    edge_latent = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)

    G = geomData(x = torch.Tensor(features),
                 edge_index=edge_spatial,
                 edge_latent=edge_latent,
                 centroid=torch.Tensor(coords))    
    torch.save(G, fpath_save)

def main(h5_path, pt_path, save_path):
    """
    h5_path: path to coordinates of patches.
    pt_path: path to features of patches.
    save_path: path to save graph files.
    """
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        path_id = h5_fname[:-3]
        if len(PATHIDS_TO_PROCESS) > 0 and path_id not in PATHIDS_TO_PROCESS:
            pbar.set_description('%s - Skipping Graph' % (path_id))
            continue

        pbar.set_description('%s - Creating Graph' % (path_id))

        if h5_fname[-2:] != 'h5':
            print('invalid h5 file {}, skipped'.format(h5_fname))
            continue

        try:
            fpath_h5 = osp.join(h5_path, path_id + '.h5')
            fpath_pt = osp.join(pt_path, path_id + '.pt')
            fpath_save = osp.join(save_path, path_id + '.pt')
            pt2graph(fpath_h5, fpath_pt, fpath_save)
        except OSError:
            pbar.set_description('%s - Broken H5' % (path_id))
            print(h5_fname, 'Broken')


# python3 patchgcn_graph_s2.py NLST
if __name__ == '__main__':
    dataset_name = sys.argv[1]
    data_dir = ROOT_DIR.format(dataset_name)
    dir_coords = osp.join(data_dir, COOR_READ_DIR)
    dir_feats = osp.join(data_dir, FEAT_READ_DIR)
    dir_save = osp.join(data_dir, SAVE_DIR)
    os.makedirs(dir_save, exist_ok=True)
    
    main(dir_coords, dir_feats, dir_save)
