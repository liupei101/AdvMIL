"""
This script is used to prepare clusters for the cluster-based DeepAttnMISL,
which follows DeepAttnMISL's original paper.

Cluster patches from the slides of a patient.
"""
import sys
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

PIDS_TO_PROCESS = []
path_csv = './table/{}_path_full.csv' # path to the table with patient_id and pathology_id
ROOT_DIR = '/data/{}/processed' # root dir of data
READ_DIR = 'feat-l1-RN50-B/pt_files' # path to patch features
SAVE_DIR = 'patch-l1-cluster{}-ids' # directory name to output and save in ROOT_DIR

def main(dir_read, dir_save, csv_path, num_clusters):
    df = pd.read_csv(csv_path, dtype={'patient_id': str, 'pathology_id': str}) # patient_id/pathology_id
    df_gps = df.groupby('patient_id').groups

    cnt = len(df_gps)
    cnt_i = 0
    for pid in df_gps.keys():
        if len(PIDS_TO_PROCESS) > 0 and pid not in PIDS_TO_PROCESS:
            print("skipped %s" % pid)
            continue

        pid_df_idx = df_gps[pid]
        
        path_ids = []
        for i in pid_df_idx:
            path_pathid = osp.join(dir_read, '{}.pt'.format(df.loc[i, 'pathology_id']))
            if not osp.exists(path_pathid):
                continue
            path_ids.append(path_pathid)
        
        if len(path_ids) == 0:
            print('pt files have not been found, skipped patient {}'.format(pid))
            continue

        feats = [torch.load(p).numpy() for p in path_ids]
        feats = np.concatenate(feats, axis=0)
        if len(feats) < num_clusters:
            print('length is less than the number of clusters, skipped')

        cluster_assign = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(feats)

        path_save = osp.join(dir_save, '{}.npy'.format(pid))
        np.save(path_save, cluster_assign)

        cnt_i += 1
        print('processed {}/{}.'.format(cnt_i, cnt))


# python3 deepattnmisl_cluster.py NLST 8
# 8 is the number of clusters you want to set
if __name__ == '__main__':
    dataset_name = sys.argv[1]
    N_cluster = int(sys.argv[2])
    data_dir = ROOT_DIR.format(dataset_name)
    dir_read = osp.join(data_dir, READ_DIR)
    dir_save = osp.join(data_dir, SAVE_DIR.format(N_cluster))
    if not osp.exists(dir_save):
        os.makedirs(dir_save)

    path_csv = path_csv.format(dataset_name, dataset_name.lower())
    main(dir_read, dir_save, path_csv, N_cluster)

