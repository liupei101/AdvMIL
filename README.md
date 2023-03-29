# AdvMIL: Adversarial Multiple Instance Learning for the Survival Analysis on Whole-Slide Images

arXiv Preprint: http://arxiv.org/abs/2212.06515

Model release: [Google Drive - AdvMIL-models](https://drive.google.com/drive/folders/1sSfUe537zWVIsNZ9t9nSS2GzwgmKG5ry?usp=sharing)

(on updating)

*TL;DR*: 
> This work proposes a novel adversarial MIL framework for the survival analysis on gagipixel Whole-Slide Images (WSIs). This framework directly estimates the distribution of time-to-event from WSIs by implicitly sampling from generator. It introduces adversarial time-to-event modeling into the MIL paradigm that is much necessary for WSI analysis, by constructing a MIL encoder and a region-level instance projection fusion network for generator and discriminator, respectively. We empirically demonstrate that AdvMIL has the following advantages or abilities: (1) combining it with existing MIL networks for predictive performance enhancement; (2) effectively utilizing unlabeled WSIs for semi-supervised learning; (3) the robustness to patch occlusion, image Gaussian blurring, and image HED color variation. 

## AdvMIL walkthrough 

Here we show **how to run AdvMIL** for WSI survival analysis. 

### Data preparation

*WSI preprocessing toolkit*: it is highly recommended to utilize an easy-to-use tool, [CLAM](https://github.com/mahmoodlab/CLAM), for WSI preprocessing, including dataset download, tissue segmentation, patching, and patch feature extraction. Please see a detailed documentation at https://github.com/mahmoodlab/CLAM. 

Next, we provide detailed steps to preprocess WSIs using `CLAM` (assuming you have already known its basic usage):
- patching at `level = 3`: go to CLAM directory and run `python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_level 3 --patch_size 256 --seg --patch --stitch`. This step will save the coordinates of segmented patches at `level = 3`. 
- patching at `level = 1`: go back to AdvMIL and run `python3 big_to_small_patching.py DIR_READ_COORDS DIR_TO_COORDS` in `./tools`. This step will compute and save the patch coordinates at `level = 1`. `DIR_READ_COORDS` should be the full path of the patch coordinates at `level = 3` from previous step. 
- Feature extracting: go to CLAM directory and run `CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs`. This step will compute all patch features and save them in `FEATURES_DIRECTORY`. Note that `DIR_TO_COORDS` should be the full path of the patch coordinates at `level = 1` from previous step. 

Now it is expected that you have the following file directories (taking `nlst` for example) in your computer.
- `/data/nlst/processed/feat-l1-RN50-B`: path to all patch features. 
- `/data/nlst/processed/tiles-l1-s256`: path to all segmented patch coordinates. 
- `/data/nlst/table/nlst_path_full.csv`: path to the csv table with `patient_id`, `pathology_id`, `t`, `e`. We have uploaded these files. Please see them in `./table`. 
- `/data/nlst/data_split/nlst-foldk.npz`: path to the file with data splitting details. We have uploaded these files. Please see them in `./data_split`. 

*Options*: if you want to a graph-based or cluster-based model, you should further prepare the followings:
- graph-based model: go to `./tools/` and run `python3 patchgcn_graph_s2.py nlst`. It will generate a new directory `/data/nlst/processed/wsigraph-l1-features` that stores patient-level graphs. 
- cluster-based model: go to `./tools/` and run `python3 deepattnmisl_cluster.py nlst 8`. It will generate a new directory `/data/nlst/processed/patch-l1-cluster8-ids` that stores cluster labels. 

### Network training

Now you should prepare a `YAML` file for configuring the setting of read/save path, network architecture, network training, etc. We have provided an example configuration (`./config/cfg_nlst.yaml`) for training nlst with AdvMIL, as well as the detailed descriptions regarding important configurations. 

Here we show and explain some important configurations so that you can successfully finish the configuration of network training. 
- `save_path`: the path for saving models, predictions, configurations, and evaluation results.
- `wandb_prj`: wandb project name, used to record all training logs. Please refer to [wandb](https://wandb.ai/) for more details.
- `bcb_mode`: the backbone of generator, one of `patch` (ESAT), `graph` (PatchGCN), `cluster` (DeepAttnMISL), and `abmil` (ABMIL).
- `disc_prj_iprd`: the way of fusion operation, one of `instance` (RLIP) and `bag` (regular fusion).
- `gen_noi_noise`: the setting of noise adding, one of `0-1`, `1-0`, and `1-1`.
- `semi_training`: whether running semi-supervised training with AdvMIL. All the related configurations are started with `ssl_`.
- `test`: whether in a test mode. Trained models will be loaded from `test_load_path` for testing the samples in `test_path`. All the related configurations are started with `test_`. 

When you finished the configuration above, you can run the following command for training, validation, and testing:
```bash
# load your config in config/cfg_nlst.yaml, and run AdvMIL
# there are three possible running modes after configuration:
# 1. common train/val/test pipeline 
# 2. only test using the models trained before (if set `test` to True)
# 3. semi-supervised train/val/test pipeline (if set `semi_training`` to True)
python3 main.py --config config/cfg_nlst.yaml --handler adv --multi_run
```

*Other options*:
- if you just want to test the model trained before, please change `test: False` to `test: True` in `config/cfg_nlst.yaml` before running. 
- if you want to run the semi-supervised training with AdvMIL, please change `semi_training: False` to `semi_training: True` in `config/cfg_nlst.yaml` before running. 

## Model release

The best models trained on WSIs, with an architecture of ESAT + AdvMIL, and their training logs are publicly-available at [Google Rrive - AdvMIL-models](https://drive.google.com/drive/folders/1sSfUe537zWVIsNZ9t9nSS2GzwgmKG5ry?usp=sharing). 

## Acknowledgment

- We thank CLAM's team [1] for contributing such an easy-to-use repo for WSI preprocessing,
- and NLST [2] and TCGA [3] for making WSI datasets publicly-available to facilitate cancer research,
- and DATE [4] for providing the demo to train basic DATE models on clinical tabular data,
- and all the authors of DeepAttnMISL [5], PatchGCN [6], and ESAT [7,8] for contributing their codes to the community.

## Reference

- Lu, M. Y.; Williamson, D. F.; Chen, T. Y.; Chen, R. J.; Bar- bieri, M.; and Mahmood, F. 2021. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6): 555–570.
- Team, N. L. S. T. R. 2011. The national lung screening trial: overview and study design. Radiology, 258(1): 243–53.
- Kandoth, C.; McLellan, M. D.; Vandin, F.; Ye, K.; Niu, B.; Lu, C.; Xie, M.; Zhang, Q.; McMichael, J. F.; Wycza- lkowski, M. A.; Leiserson, M. D. M.; Miller, C. A.; Welch, J. S.; Walter, M. J.; Wendl, M. C.; Ley, T. J.; Wilson, R. K.; Raphael, B. J.; and Ding, L. 2013. Mutational landscape and significance across 12 major cancer types. Nature, 502: 333 – 339.
- Chapfuwa, P.; Tao, C.; Li, C.; Page, C.; Goldstein, B.; Duke, L. C.; and Henao, R. 2018. Adversarial time-to-event mod- eling. In International Conference on Machine Learning, 735–744. PMLR. 
- Yao, J.; Zhu, X.; Jonnagaddala, J.; Hawkins, N.; and Huang, J. 2020. Whole slide images based cancer survival predic- tion using attention guided deep multiple instance learning networks. Medical Image Analysis, 65: 101789.
- Chen, R. J.; Lu, M. Y.; Shaban, M.; Chen, C.; Chen, T. Y.; Williamson, D. F.; and Mahmood, F. 2021. Whole Slide Im- ages are 2D Point Clouds: Context-Aware Survival Predic- tion using Patch-based Graph Convolutional Networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 339–349. Springer. 
- Shen, Y.; Liu, L.; Tang, Z.; Chen, Z.; Ma, G.; Dong, J.; Zhang, X.; Yang, L.; and Zheng, Q. 2022. Explainable Survival Analysis with Convolution-Involved Vision Transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, 2207--2215. AAAI Press. 
- Liu, P.; Fu, B.; Ye, F.; Yang, R.; Xu, B.; and Ji, L. 2022. Dual-Stream Transformer with Cross-Attention on Whole- Slide Image Pyramids for Cancer Prognosis. arXiv preprint arXiv:2206.05782.

## Citation

If this work helps your research, please consider citing our paper:
```
@article{liu2022advmil,
  title={AdvMIL: Adversarial Multiple Instance Learning for the Survival Analysis on Whole-Slide Images},
  author={Liu, Pei and Ji, Luping and Ye, Feng and Fu, Bo},
  journal={arXiv preprint arXiv:2212.06515},
  year={2022}
}
```
