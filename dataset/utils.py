from .PatchWSI import WSIPatch


def prepare_dataset(patient_ids:list, cfg, **kws):
    """
    patient_ids: a list including all patient IDs.
    cfg: a dict where 'path_patch', 'path_label', and 'feat_format' are included.
    """
    path_patch = cfg['path_patch']
    path_label = cfg['path_label']
    mode = cfg['bcb_mode']
    feat_format = cfg['feat_format']
    time_format = cfg['time_format']
    time_bins = cfg['time_bins']
    if 'ratio_sampling' in kws:
        ratio_sampling = kws['ratio_sampling']
    else:
        ratio_sampling = None
    dataset = WSIPatch(
        patient_ids, path_patch, path_label, mode, 
        read_format=feat_format, time_format=time_format, time_bins=time_bins, ratio_sampling=ratio_sampling,
        cluster_path=cfg['path_cluster'], coord_path=cfg['path_coordx5'], graph_path=cfg['path_graph']
    )
    return dataset
