# AdvMIL: Adversarial Multiple Instance Learning for the Survival Analysis on Whole-Slide Images

arXiv Preprint: http://arxiv.org/abs/2212.06515
Model release: [Google Rrive - AdvMIL-models](https://drive.google.com/drive/folders/1sSfUe537zWVIsNZ9t9nSS2GzwgmKG5ry?usp=sharing)

*TLDR*: 
> This work proposes a novel adversarial MIL framework for the survival analysis on gagipixel Whole-Slide Images (WSIs). This framework directly estimates the distribution of time-to-event from WSIs by implicitly sampling from generator. It introduces adversarial time-to-event modeling into the MIL paradigm that is much necessary for WSI analysis, by constructing a MIL encoder and a region-level instance projection fusion network for generator and discriminator, respectively. We empirically demonstrate that AdvMIL has the following advantages or abilities: (1) combining it with existing MIL networks for predictive performance enhancement; (2) effectively utilizing unlabeled WSIs for semi-supervised learning; (3) the robustness to patch occlusion, image gaussian blurring, and image HED color variation. 

## AdvMIL walkthrough 

Here we show **how to run AdvMIL** for WSI survival analysis. 

### Data preparation

It is highly recommended to utilize an easy-to-use tool, [CLAM](https://github.com/mahmoodlab/CLAM), for WSI preprocessing, including dataset download, tissue segmentation, patching, and patch feature extraction. Please see a detailed documentation at https://github.com/mahmoodlab/CLAM. 

With CLAM, it is expected that you have the following file directories (taking `nlst` for example) in your computer.
- `/data/nlst/processed/feat-x20-RN50-B`: path to all patch features. 
- `/data/nlst/processed/tiles-x20-s256`: path to all segmented patch coordinates. 

Option: if you want to a graph-based or cluster-based model, you should further prepare the followings:
- graph-based model: go to `./tools/` and run `python3 patchgcn_graph_s1.py nlst`  and `python3 patchgcn_graph_s2.py nlst`.
- cluster-based model: go to `./tools/` and run `python3 deepattnmisl_cluster.py nlst 8`.

### Network training

You should prepare a YAML file for configuring the setting of read/save path, network architecture, network training, etc. We have provided an example (`./config/cfg_nlst.yaml`), as well as detailed descriptions regarding important configurations. 

When you finished the configuration, you can run the following command for training, validation, and testing AdvMIL:
```bash
# train, val, and test
python3 main.py --config config/cfg_nlst.yaml --handler adv --multi_run
```

Other options:
- if you just want to test the model trained before, please open `config/cfg_nlst.yaml` and then change `test: False` to `test: True` before running.
- if you want to run the semi-supervised training with AdvMIL, please open `config/cfg_nlst.yaml` and then change `semi_training: False` to `semi_training: True` before running.

## Model release

The best models trained on WSIs, with an architecture of ESAT + AdvMIL, are publicly-available at [Google Rrive - AdvMIL-models](https://drive.google.com/drive/folders/1sSfUe537zWVIsNZ9t9nSS2GzwgmKG5ry?usp=sharing).

## Acknowledgment
- We thank CLAM's team [1] for contributing such an easy-to-use repo for WSI preprocessing,
- and NLST [2] and TCGA [3] for making WSI datasets publicly-available to facilitate cancer research,
- and DATE [4] for providing the demo to train basic DATE models on clinical tabular data,
- and all the authors of DeepAttnMISL [5], PatchGCN [6], and ESAT [7,8] for contributing their codes to the community.

## Reference
[1] Lu, M. Y.; Williamson, D. F.; Chen, T. Y.; Chen, R. J.; Bar- bieri, M.; and Mahmood, F. 2021. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6): 555–570.

[2] Team, N. L. S. T. R. 2011. The national lung screening trial: overview and study design. Radiology, 258(1): 243–53.

[3] Kandoth, C.; McLellan, M. D.; Vandin, F.; Ye, K.; Niu, B.; Lu, C.; Xie, M.; Zhang, Q.; McMichael, J. F.; Wycza- lkowski, M. A.; Leiserson, M. D. M.; Miller, C. A.; Welch, J. S.; Walter, M. J.; Wendl, M. C.; Ley, T. J.; Wilson, R. K.; Raphael, B. J.; and Ding, L. 2013. Mutational landscape and significance across 12 major cancer types. Nature, 502: 333 – 339.

[4] Chapfuwa, P.; Tao, C.; Li, C.; Page, C.; Goldstein, B.; Duke, L. C.; and Henao, R. 2018. Adversarial time-to-event mod- eling. In International Conference on Machine Learning, 735–744. PMLR. 

[5] Yao, J.; Zhu, X.; Jonnagaddala, J.; Hawkins, N.; and Huang, J. 2020. Whole slide images based cancer survival predic- tion using attention guided deep multiple instance learning networks. Medical Image Analysis, 65: 101789.

[6] Chen, R. J.; Lu, M. Y.; Shaban, M.; Chen, C.; Chen, T. Y.; Williamson, D. F.; and Mahmood, F. 2021. Whole Slide Im- ages are 2D Point Clouds: Context-Aware Survival Predic- tion using Patch-based Graph Convolutional Networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 339–349. Springer. 

[7] Shen, Y.; Liu, L.; Tang, Z.; Chen, Z.; Ma, G.; Dong, J.; Zhang, X.; Yang, L.; and Zheng, Q. 2022. Explainable Survival Analysis with Convolution-Involved Vision Transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, 2207--2215. AAAI Press. 

[8] Liu, P.; Fu, B.; Ye, F.; Yang, R.; Xu, B.; and Ji, L. 2022. Dual-Stream Transformer with Cross-Attention on Whole- Slide Image Pyramids for Cancer Prognosis. arXiv preprint arXiv:2206.05782.
