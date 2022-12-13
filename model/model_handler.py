import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from functools import partial
from types import SimpleNamespace
import wandb

from .GANSurv import Generator as Generator
from .GANSurv import PrjDiscriminator as Discriminator
from .model_utils import init_weights
from .backbone import load_backbone
from utils.func import *
from utils.io import read_datasplit_npz, read_maxt_from_table
from utils.io import save_prediction
from dataset.utils import prepare_dataset
from eval.utils import prepare_evaluator
from loss.utils import *
from optim import create_optimizer
from dataset.GraphBatchWSI import collate_MIL_graph

#######################################################################
#     Handling general model initialization, training, and test
# 
# This class handles three tasks:
# 1. general model training, validation, and test for AdvMIL;
# 2. k-fold semi-supervised training, validation, test for AdvMIL;
# 3. test using previous pre-trained model weights (to be completed).
#######################################################################

class MyHandler(object):
    def __init__(self, cfg):
        _check_configs(cfg)

        torch.cuda.set_device(cfg['cuda_id'])
        seed_everything(cfg['seed'])

        if not osp.exists(cfg['save_path']):
            os.makedirs(cfg['save_path'])
        run_name = cfg['save_path'].split('/')[-1]
        self.writer = wandb.init(project=cfg['wandb_prj'], name=run_name, dir=cfg['wandb_dir'], config=cfg, reinit=True)
        self.last_netD_ckpt_path = osp.join(cfg['save_path'], 'modelD-last.pth')
        self.best_netD_ckpt_path = osp.join(cfg['save_path'], 'modelD-best.pth')
        self.last_netG_ckpt_path = osp.join(cfg['save_path'], 'modelG-last.pth')
        self.best_netG_ckpt_path = osp.join(cfg['save_path'], 'modelG-best.pth')
        self.last_metrics_path   = osp.join(cfg['save_path'], 'metrics-last.txt')
        self.best_metrics_path   = osp.join(cfg['save_path'], 'metrics-best.txt')
        self.config_path    = osp.join(cfg['save_path'], 'print_config.txt')

        self.bcb = cfg['bcb_mode']
        # setup model
        if cfg['task'] in ['cont_gansurv', 'disc_gansurv']:
            # Generator
            backbone_dims = sparse_str(cfg['bcb_dims'])
            backbone = load_backbone(self.bcb, backbone_dims)
            dim_in, dim_out = sparse_str(cfg['gen_dims'])
            args_noise = SimpleNamespace(**sparse_key(cfg, prefixes='gen_noi'))
            args_noise.noise = sparse_str(args_noise.noise)
            self.netG = Generator(dim_in, dim_out, backbone, args_noise, 
                cfg['gen_norm'], cfg['gen_dropout'], cfg['gen_out_scale'])
            self.netG.apply(init_weights) # apply model initialization
            # Discriminator
            disc_x_args = SimpleNamespace(**sparse_key(cfg, prefixes='disc_netx'))
            disc_y_args = SimpleNamespace(**sparse_key(cfg, prefixes='disc_nety'))
            disc_y_args.hid_dims = sparse_str(disc_y_args.hid_dims)
            self.netD = Discriminator(disc_x_args, disc_y_args, prj_path=cfg['disc_prj_path'], inner_product=cfg['disc_prj_iprd'])
        else:
            raise ValueError(f"Expected cont_gansurv/disc_gansurv, but got {cfg['task']}")
        self.netG = self.netG.cuda()
        self.netD = self.netD.cuda()

        # loss
        self.real_fake_loss = partial(real_fake_loss, which=cfg['loss_netD'])
        if cfg['task'] == 'cont_gansurv':
            self.supervised_loss = partial(recon_loss, **sparse_key(cfg, prefixes='loss_recon'))
        elif cfg['task'] == 'disc_gansurv':
            self.supervised_loss = SurvMLE(**sparse_key(cfg, prefixes='loss_mle'))
        else:
            pass
        self.coef_ganloss = cfg['loss_gan_coef']
        self.loss_l1 = loss_reg_l1(cfg['loss_regl1_coef'])
        # optimizer
        cfg_optimizer = SimpleNamespace(opt=cfg['opt_netG'], weight_decay=cfg['opt_netG_weight_decay'], lr=cfg['opt_netG_lr'], 
            opt_eps=None, opt_betas=None, momentum=None)
        self.optimizerG = create_optimizer(cfg_optimizer, self.netG)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=cfg['opt_netD_lr'], betas=(0.9, 0.999), weight_decay=0.0)
        # LR scheduler for netG
        self.steplr = lr_scheduler.ReduceLROnPlateau(self.optimizerG, mode='min', factor=0.5, patience=10, verbose=True)

        # setup evaluator
        if cfg['time_format'] in ['origin', 'quantile']:
            end_time = read_maxt_from_table(cfg['path_label'])
        else:
            end_time = 1.0
        if cfg['task'] == 'cont_gansurv':
            self.evaluator = prepare_evaluator('continuous', 
                end_time=end_time, recon_loss=self.supervised_loss, rank_loss=None, 
                disc_loss=self.real_fake_loss,
            )
            self.metrics_list = ['c_index', 'loss_recon', 'loss_recon_org', 'loss_fake_netD', 'loss_fake_netG', 
                'avg_fake', 'event_t_rae', 'nonevent_t_rae', 'event_t_nre', 'nonevent_t_nre']
            # the printed loss is measured by alpha = 0.0, and the optimized loss is measured by the alpha one set previously
            self.ret_metrics = ['c_index', 'loss_recon_org']
        elif cfg['task'] == 'disc_gansurv':
            self.evaluator = prepare_evaluator('discrete', mle_loss=self.supervised_loss, disc_loss=self.real_fake_loss)
            self.metrics_list = ['c_index', 'loss_mle', 'loss_mle_org', 'loss_fake_netD', 'loss_fake_netG', 'avg_fake']
            # the printed loss is measured by alpha = 0.0, and the optimized loss is measured by the alpha one set previously
            self.ret_metrics = ['c_index', 'loss_mle_org']
            self.nbins = cfg['time_bins']
        else:
            pass

        self.patient_id = dict()
        self.task = cfg['task']
        self.cfg = cfg
        print_config(cfg, print_to_path=self.config_path)

    def exec(self):
        print('[exec] execute task {} using backbone-mode {}.'.format(self.task, self.bcb))
        
        # Prepare datasets 
        path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        self.patient_id.update({'label_visible': pids_train + pids_val + pids_test})
        print('[exec] read patient IDs from {}'.format(path_split))

        if self.bcb == 'graph':
            collate = collate_MIL_graph
        else:
            collate = default_collate

        # Prepare datasets 
        train_set  = prepare_dataset(pids_train, self.cfg, ratio_sampling=self.cfg['train_sampling'])
        self.patient_id.update({'train': train_set.pids})
        val_set    = prepare_dataset(pids_val, self.cfg)
        self.patient_id.update({'validation': val_set.pids})
        train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=True,  worker_init_fn=seed_worker, collate_fn=collate
        )
        val_loader = DataLoader(val_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False, worker_init_fn=seed_worker, collate_fn=collate
        )
        if pids_test is not None:
            test_set    = prepare_dataset(pids_test, self.cfg)
            self.patient_id.update({'test': test_set.pids})
            test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
                generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                shuffle=False, worker_init_fn=seed_worker, collate_fn=collate
            )
        else:
            test_set = None 
            test_loader = None

        # Train
        run_name = 'train'
        val_name = 'validation'
        val_loaders = {'validation': val_loader, 'test': test_loader}
        self._run_training(self.cfg['epochs'], train_loader, 'train', val_loaders=val_loaders, val_name=val_name, 
            measure_training_set=True, save_ckpt=True, early_stop=True, run_name=run_name)

        # Evals
        evals_loader = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
        metrics = self._eval_all(evals_loader, ckpt_type='best', run_name=run_name, if_print=True)
        return metrics

    def _run_training(self, epochs, train_loader, name_loader, val_loaders=None, val_name=None, 
        mode='wlabel', measure_training_set=True, save_ckpt=True, early_stop=False, run_name='train', **kws):
        """Traing model.

        Args:
            epochs (int): Epochs to run.
            train_loader ('DataLoader'): DatasetLoader of training set.
            name_loader (string): name of train_loader, used for infering patient IDs.
            val_loaders (dict): A dict like {'val': loader1, 'test': loader2}, which gives the datasets
                to evaluate at each epoch.
            val_name (string): The dataset used to perform early stopping and optimal model saving.
            mode (string): 'wlabel' indicates training with all labeled data, otherwise training with some unlabeled data.
            measure_training_set (bool): If measure training set at each epoch.
            save_ckpt (bool): If save models.
            early_stop (bool): If early stopping according to validation loss.
            run_name (string): Name of this training, which would be used as the prefixed name of ckpt files.
        """
        if mode == 'wlabel':
            if early_stop and self.cfg['es_patience'] is not None:
                self.early_stop = EarlyStopping(warmup=self.cfg['es_warmup'], patience=self.cfg['es_patience'], 
                    start_epoch=self.cfg['es_start_epoch'], verbose=self.cfg['es_verbose'])
            else:
                self.early_stop = None
        else:
            if early_stop and self.cfg['ssl_es_patience'] is not None:
                self.early_stop = EarlyStopping(warmup=self.cfg['ssl_es_warmup'], patience=self.cfg['ssl_es_patience'], 
                    start_epoch=self.cfg['ssl_es_start_epoch'], verbose=self.cfg['ssl_es_verbose'])
            else:
                self.early_stop = None

        if val_name is not None and self.early_stop is not None:
            assert val_name in val_loaders.keys(), "Not specify a dataloader to enable early stopping."
            print("[{} {}] {} epochs w early stopping on {}.".format(run_name, mode, epochs, val_name))
        else:
            print("[{} {}] {} epochs w/o early stopping.".format(run_name, mode, epochs))
        
        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch + 1
            if type(name_loader) == list: # used for kfold semi-supervised training
                cur_idx = epoch % len(name_loader)
                train_cltor = self._train_each_epoch(train_loader[cur_idx], name_loader[cur_idx], mode=mode)
                cur_name = name_loader[cur_idx]
            else:
                train_cltor = self._train_each_epoch(train_loader, name_loader, mode=mode)
                cur_name = name_loader

            if measure_training_set:
                self._eval_and_print(train_cltor, name=cur_name, at_epoch=epoch+1)

            # val/test
            val_metrics = None
            if val_loaders is not None:
                for k in val_loaders.keys():
                    if val_loaders[k] is None:
                        continue
                    val_cltor = self.test_model(self.netG, self.netD, self.bcb, val_loaders[k], times_test_sample=1)
                    met_ci, met_loss = self._eval_and_print(val_cltor, name=k, at_epoch=epoch+1)
                    if k == val_name:
                        val_metrics = met_ci if self.cfg['monitor_metrics'] == 'ci' else met_loss
            
            # early_stop using VAL_METRICS (**is measured using alpha=0.0**)
            if val_metrics is not None and self.early_stop is not None:
                self.steplr.step(val_metrics)
                self.early_stop(epoch, val_metrics)
                if self.early_stop.if_save_checkpoint():
                    self.save_model(epoch+1, ckpt_type='best', run_name=run_name)
                    print("[{} {}] best model saved at epoch {}".format(run_name, mode, epoch+1))
                if self.early_stop.if_stop():
                    break
        
        if save_ckpt:
            self.save_model(last_epoch, ckpt_type='last', run_name=run_name) # save models and optimizers
            print("[{} {}] last model saved at epoch {}".format(run_name, mode, last_epoch))

    def _train_each_epoch(self, train_loader, name_loader, mode='wlabel'):
        print("[train] train one epoch using train_loader={} under mode={}".format(name_loader, mode))
        bp_every_batch = self.cfg['bp_every_batch']
        num_update_gen = self.cfg['gen_updates']
        train_cltor = {'y': None, 'y_hat': None, 'f_fake': None}
        i_collector = []
        x_collector = []
        y_collector = []

        i_batch = 0
        for data_idx, data_x, data_y in train_loader:
            i_batch += 1
            # 1. forward propagation
            # it contains two parts: WSI and its extra data
            data_x = [dx.cuda() for dx in data_x] 
            data_y = data_y.cuda()
            i_collector.append(data_idx)
            x_collector.append(data_x)
            y_collector.append(data_y)

            if i_batch % bp_every_batch == 0:
                # get label mask
                label_visible_mask = self._get_label_visiable_mask(name_loader, i_collector)

                # update discriminator
                cur_preds, cur_fakes = self._update_disc(i_batch, x_collector, y_collector, mode, label_visible_mask)

                # update generator
                for _ in range(num_update_gen):
                    self._update_gen(i_batch, x_collector, y_collector, mode, label_visible_mask)

                # collect preditions
                train_cltor = agg_tensor(
                    train_cltor, 
                    {'y': torch.cat(y_collector, dim=0).detach().cpu(),
                     'y_hat': torch.cat(cur_preds, dim=0).detach().cpu(),
                     'f_fake': torch.cat(cur_fakes, dim=0).detach().cpu(),
                    }
                )

                # reset vars
                i_collector = []
                x_collector = []
                y_collector = []
                torch.cuda.empty_cache()

        return train_cltor

    def _update_disc(self, i_batch, xs, ys, mode='wlabel', label_visible_mask=None):
        """
        mode = 'wlabel': training with all label visible.
        mode = 'wolabel': training with some label invisible, i.e., real pairs won't be feed into netD.
        label_visible_mask (list): if mode = 'wolabel', we determine whether use a label according to its `label_visible_mask`.
        """
        self.netD.train()
        self.netG.eval()

        bp_collector = {'real': None, 'fake': None}
        pred_collector, fake_collector = [], []
        n_sample = len(xs)
        if label_visible_mask is None:
            label_visible_mask = [mode == 'wlabel' for _ in range(n_sample)]
        else:
            label_visible_mask = [mode == 'wlabel' or (mode == 'wolabel' and mask) for mask in label_visible_mask]

        for i in range(n_sample):
            data_x, data_x_ext, data_t, data_ind = xs[i][0], xs[i][1], ys[i][:, [0]], ys[i][:, [1]]

            if self.bcb == 'graph':
                data_x = data_x.unsqueeze(0)

            if self.task == 'cont_gansurv':
                ind_obs = (data_ind == 1).squeeze(-1) 
                # real pairs = (data_y, data_x) & e = 1
                if torch.sum(ind_obs) > 0 and label_visible_mask[i]:
                    f_real = self.netD(data_x[ind_obs, :], data_t[ind_obs, :])
                    f_real = f_real.view(-1)
                else:
                    f_real = None
            elif self.task == 'disc_gansurv':
                # real pairs = (data_vec_y, data_x)
                y, y_mask = get_label_mask(data_t, data_ind, self.nbins)
                f_real = self.netD(data_x, y * y_mask)
                f_real = f_real.view(-1)
            
            # fake pairs = (pred, data_x)
            if self.bcb == 'graph':
                pred = self.netG(data_x_ext, None) # data_x_ext -> GraphData if backbone=graph
            elif self.bcb == 'patch':
                pred = self.netG(data_x, None) # skip coords if backbone=patch
                # pred = self.netG(data_x, data_x_ext)
            else:
                pred = self.netG(data_x, data_x_ext)
            pred_collector.append(pred)

            pat_mask = torch.logical_or(data_ind == 1, data_ind == 0).squeeze(-1)
            # pat_mask = (data_ind == 1).squeeze() # if we only add the fake pairs with event to netD
            if torch.sum(pat_mask) > 0:
                masked_pred = pred * y_mask if self.task == 'disc_gansurv' else pred 
                f_fake = self.netD(data_x[pat_mask, :], masked_pred[pat_mask, :].detach()) 
                f_fake = f_fake.view(-1)
            else:
                f_fake = None
            
            bp_collector = collect_tensor(bp_collector, f_real, f_fake)
            fake_collector.append(f_fake.detach()) # used for evaluating training set

        # 2.1 zero gradients buffer
        self.optimizerD.zero_grad()

        # 2.2 calculate D loss for all real and fake pairs
        dis_loss = self.real_fake_loss(bp_collector['real'], bp_collector['fake'])
        print("[training one epoch] training batch {}, dis_loss: {:.6f}".format(i_batch, dis_loss.item()))
        wandb.log({
            'train_batch/netD/Loss_D': dis_loss.item(),
            'train_batch/netD/D_real': 0.0 if bp_collector['real'] is None else bp_collector['real'].mean().item(), 
            'train_batch/netD/D_fake': bp_collector['fake'].mean().item()
        })

        # 2.3 backwards gradients and update networks
        dis_loss.backward()
        self.optimizerD.step()

        return pred_collector, fake_collector

    def _update_gen(self, i_batch, xs, ys, mode='wlabel', label_visible_mask=None):
        """
        mode = 'wlabel': training with label visible.
        mode = 'wolabel': training with label invisible, i.e., SLoss won't be used to update netG.
        label_visible_mask (list): if mode = 'wolabel', we determine whether use a label according to its `label_visible_mask`.
        """
        self.netD.eval()
        self.netG.train()

        n_sample = len(xs)
        f_fake_collector, f_fake_list, preds = [], [], []
        if label_visible_mask is None:
            label_visible_mask = [mode == 'wlabel' for _ in range(n_sample)]
        else:
            label_visible_mask = [mode == 'wlabel' or (mode == 'wolabel' and mask) for mask in label_visible_mask]

        for i in range(n_sample):
            data_x, data_x_ext, data_t, data_ind = xs[i][0], xs[i][1], ys[i][:, [0]], ys[i][:, [1]]
            if self.task == 'disc_gansurv':
                y, y_mask = get_label_mask(data_t, data_ind, self.nbins)

            if self.bcb == 'graph':
                data_x = data_x.unsqueeze(0)
                pred = self.netG(data_x_ext, None) # data_x_ext -> GraphData if backbone=graph
            elif self.bcb == 'patch':
                pred = self.netG(data_x, None) # skip coords if backbone=patch
                # pred = self.netG(data_x, data_x_ext)
            else:
                pred = self.netG(data_x, data_x_ext) # generate pred given data_x
            preds.append(pred)

            pat_mask = torch.logical_or(data_ind == 1, data_ind == 0).squeeze(-1)
            # pat_mask = (ys[i][:, [1]] == 1).squeeze() # we only add the fake pairs with event to netD
            if torch.sum(pat_mask) > 0:
                masked_pred = pred * y_mask if self.task == 'disc_gansurv' else pred
                f_fake = self.netD(data_x[pat_mask, :], masked_pred[pat_mask, :])
                f_fake = f_fake.view(-1)
                f_fake_collector.append(f_fake)
                f_fake_list += list(f_fake.detach().cpu().numpy())
            else:
                f_fake = None

        # 3.1 zero gradients buffer
        self.optimizerG.zero_grad()

        # 3.2 loss
        gen_loss = fake_generator_loss(torch.cat(f_fake_collector, dim=0))
        if sum(label_visible_mask) > 0:
            # only select the example whose label is visible when computing supervised_loss
            t_preds = torch.cat([preds[i] for i in range(n_sample) if label_visible_mask[i]], dim=0)
            label_t = torch.cat([ys[i][:,[0]] for i in range(n_sample) if label_visible_mask[i]], dim=0)
            label_ind = torch.cat([ys[i][:,[1]] for i in range(n_sample) if label_visible_mask[i]], dim=0)
            t_reg_loss = self.supervised_loss(t_preds, label_t, label_ind)
        else:
            t_reg_loss = torch.Tensor([0.0]).to(gen_loss.device)
        if self.coef_ganloss == 0.0:
            gen_total_loss = t_reg_loss
        else:
            gen_total_loss = t_reg_loss + self.coef_ganloss * gen_loss
        gen_total_loss += self.loss_l1(self.netG.parameters())
        print("[training one epoch] training batch {}, gen_loss: {:.6f}, t_reg_loss: {:.6f}, \
            , gen_totol_loss: {:.6f}".\
            format(i_batch, gen_loss.item(), t_reg_loss.item(), gen_total_loss.item()))
        wandb.log({
            'train_batch/netG/Loss_G_fake': gen_loss.item(), 
            'train_batch/netG/Loss_G_time': t_reg_loss.item(),
            'train_batch/netG/Loss_G_total': gen_total_loss.item(),
            'train_batch/netG/D_fake_avg': sum(f_fake_list) / len(f_fake_list)
        })

        # 3.3 backwards gradients and update networks
        gen_total_loss.backward()
        self.optimizerG.step()

    def _eval_all(self, evals_loader, ckpt_type='best', run_name='train', if_print=True):
        if ckpt_type == 'best':
            ckpts = [add_prefix_to_filename(self.best_netG_ckpt_path, run_name), 
                     add_prefix_to_filename(self.best_netD_ckpt_path, run_name)]
            wandb_group = 'bestckpt/{}'.format(run_name)
            print_path = add_prefix_to_filename(self.best_metrics_path, run_name)
            csv_name = '{}_best'.format(run_name)
        elif ckpt_type == 'last':
            ckpts = [add_prefix_to_filename(self.last_netG_ckpt_path, run_name), 
                     add_prefix_to_filename(self.last_netD_ckpt_path, run_name)]
            wandb_group = 'lastckpt/{}'.format(run_name)
            print_path = add_prefix_to_filename(self.last_metrics_path, run_name)
            csv_name = '{}_last'.format(run_name)
        else:
            pass

        metrics = dict()
        for k, loader in evals_loader.items():
            if loader is None:
                continue
            cltor = self.test_model(self.netG, self.netD, self.bcb, loader, 
                times_test_sample=self.cfg['times_test_sample'], checkpoints=ckpts,
            )
            ci, loss = self._eval_and_print(cltor, name='{}/{}'.format(wandb_group, k))
            metrics[k] = [('cindex', ci), ('loss', loss)]

            cur_y = cltor['y']
            if 'avg_y_hat' in cltor:
                cur_y_hat = cltor['avg_y_hat']
            else:
                cur_y_hat = cltor['y_hat']

            if self.cfg['log_plot']:
                plt = plot_time_kde(cur_y, cur_y_hat)
                plt_img = wandb.Image(plt)
                plt_name = '{}/{}/chart'.format(wandb_group, k)
                wandb.log({plt_name: plt_img})

            if self.cfg['save_prediction']:
                path_save_pred = osp.join(self.cfg['save_path'], '{}_pred_{}.csv'.format(csv_name, k))
                patient_ids = self._get_patient_id(k, cltor['idx'])
                save_prediction(patient_ids, cur_y, cur_y_hat, path_save_pred)

        if if_print:
            print_metrics(metrics, print_to_path=print_path)

        return metrics

    def _eval_and_print(self, cltor, name='', ret_metrics=None, at_epoch=None):
        if ret_metrics is None:
            ret_metrics = self.ret_metrics

        eval_results = self.evaluator.compute(cltor, self.metrics_list)
        eval_results = rename_keys(eval_results, name, sep='/')

        print("[{}] At epoch {}:".format(name, at_epoch), end=' ')
        print(' '.join(['{}={:.6f},'.format(k, v) for k, v in eval_results.items()]))
        wandb.log(eval_results)

        return [eval_results[name+'/'+k] for k in ret_metrics]

    def _get_patient_id(self, k, idxs):
        if k not in self.patient_id:
            raise KeyError('Key {} not found in `patient_id`'.format(k))
        pids = self.patient_id[k]
        idxs = idxs.squeeze().tolist()
        return [pids[i] for i in idxs]

    def _get_label_visiable_mask(self, k, idxs):
        label_visible_pids = self.patient_id['label_visible']
        if isinstance(idxs, list):
            idxs = torch.cat(idxs, dim=0) 
        pids = self._get_patient_id(k, idxs)
        return [p in label_visible_pids for p in pids]

    @staticmethod
    def test_model(modelG, modelD, backbone, loader, times_test_sample=1, checkpoints=None):
        if checkpoints is not None:
            netG_ckpt = torch.load(checkpoints[0])
            netD_ckpt = torch.load(checkpoints[1])
            modelG.load_state_dict(netG_ckpt['model'])
            modelD.load_state_dict(netD_ckpt['model'])
        modelG.eval()
        modelD.eval()
        res = {'idx': None, 'y': None, 'y_hat': None, 'f_fake': None}
        with torch.no_grad():
            for idx, x, y in loader:
                x_data, x_ext = [_x.cuda() for _x in x]
                if backbone == 'graph':
                    x_data = x_data.unsqueeze(0)
                    y_hat = modelG(x_ext, None)
                elif backbone == 'patch':
                    y_hat = modelG(x_data, None) # skip coords if backbone=patch
                    # y_hat = modelG(x_data, x_ext)
                else:
                    y_hat = modelG(x_data, x_ext)
                f_fake = modelD(x_data, y_hat)
                res = agg_tensor(res, 
                    {'idx': idx.detach().cpu(), 'y': y.detach().cpu(), 
                     'y_hat': y_hat.detach().cpu(), 'f_fake': f_fake.detach().cpu()}
                )
                if times_test_sample > 1:
                    # run the model for [times_test_sample] times and use its median as final predition
                    # y_hat_list: [times_test_sample, B, out_dim]
                    y_hat_list = []
                    for i in range(times_test_sample):
                        if backbone == 'graph':
                            cur_y_hat = modelG(x_ext, None)
                        elif backbone == 'patch':
                            cur_y_hat = modelG(x_data, None) # skip coords if backbone=patch
                            # cur_y_hat = modelG(x_data, x_ext)
                        else:
                            cur_y_hat = modelG(x_data, x_ext)
                        y_hat_list.append(cur_y_hat)
                    y_hat_list = torch.stack(y_hat_list)
                    avg_y_hat, _ = torch.median(y_hat_list, dim=0)
                    res = agg_tensor(res, {'avg_y_hat': avg_y_hat.detach().cpu()})

        return res

    def _get_state_dict(self, epoch, model='G'):
        return {
            'epoch': epoch,
            'model': self.netG.state_dict() if model == 'G' else self.netD.state_dict(),
            'optimizer': self.optimizerG.state_dict() if model == 'G' else self.optimizerD.state_dict(),
        }

    def save_model(self, epoch, ckpt_type='best', run_name='train'):
        netG_ckpt_dict = self._get_state_dict(epoch, model='G')
        netD_ckpt_dict = self._get_state_dict(epoch, model='D')
        if ckpt_type == 'last':
            torch.save(netG_ckpt_dict, add_prefix_to_filename(self.last_netG_ckpt_path, prefix=run_name))
            torch.save(netD_ckpt_dict, add_prefix_to_filename(self.last_netD_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            torch.save(netG_ckpt_dict, add_prefix_to_filename(self.best_netG_ckpt_path, prefix=run_name))
            torch.save(netD_ckpt_dict, add_prefix_to_filename(self.best_netD_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))

    def resume_model(self, ckpt_type='best', run_name='train'):
        if ckpt_type == 'last':
            netG_ckpt = torch.load(add_prefix_to_filename(self.last_netG_ckpt_path, prefix=run_name))
            netD_ckpt = torch.load(add_prefix_to_filename(self.last_netD_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            netG_ckpt = torch.load(add_prefix_to_filename(self.best_netG_ckpt_path, prefix=run_name))
            netD_ckpt = torch.load(add_prefix_to_filename(self.best_netD_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))
        self.netG.load_state_dict(netG_ckpt['model']) 
        self.optimizerG.load_state_dict(netG_ckpt['optimizer']) 
        print('[model] resume the netG from {}_{} at epoch {}...'.format(ckpt_type, run_name, netG_ckpt['epoch']))
        self.netD.load_state_dict(netD_ckpt['model']) 
        self.optimizerD.load_state_dict(netD_ckpt['optimizer']) 
        print('[model] resume the netD from {}_{} at epoch {}...'.format(ckpt_type, run_name, netD_ckpt['epoch']))

    def exec_semi_sl(self):
        print('[exec_semi_sl] start experiments of SSL in task {}.'.format(self.task))
        assert self.cfg['semi_training']
        
        # Prepare datasets 
        path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        print('[exec_semi_sl] read patient IDs from {}'.format(path_split))

        # Split into the data w label and w/o label
        labeled_train_pids, unlabeled_train_pids = sampling_data(pids_train, self.cfg['ssl_num_labeled'])
        # used for label mask query
        self.patient_id.update({'label_visible': labeled_train_pids, 'label_invisible': unlabeled_train_pids})

        # Semi-Supervised Learning
        labeled_train_set = prepare_dataset(labeled_train_pids, self.cfg)
        unlabeled_train_set = prepare_dataset(unlabeled_train_pids, self.cfg)
        self.patient_id.update({'labeled_train': labeled_train_set.pids, 'unlabeled_train': unlabeled_train_set.pids})
        labeled_train_loader = DataLoader(labeled_train_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=True,  worker_init_fn=seed_worker
        )
        unlabeled_train_loader = DataLoader(unlabeled_train_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=True,  worker_init_fn=seed_worker
        )

        # val/test dataset
        val_set, test_set = prepare_dataset(pids_val, self.cfg), prepare_dataset(pids_test, self.cfg)
        self.patient_id.update({'validation': val_set.pids, 'test': test_set.pids})
        val_loader = DataLoader(val_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False,  worker_init_fn=seed_worker
        )
        test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False,  worker_init_fn=seed_worker
        )
        val_name = 'validation'
        val_loaders = {'validation': val_loader, 'test': test_loader}

        # First training phrase: labeled dataset for pretraining N epochs w/o early stopping.
        # Model evaluation using the pretraining model at last epoch.
        skip_first_phrase = True
        if not skip_first_phrase:
            run_name = 'pretrain'
            print('[exec_semi_sl] start the first phrase: {}'.format(run_name))
            self._run_training(self.cfg['epochs'], labeled_train_loader, 'labeled_train', val_loaders=val_loaders, val_name=val_name, 
                measure_training_set=True, save_ckpt=True, early_stop=False, run_name=run_name)
            evals_loader = {'labeled_train': labeled_train_loader, 'unlabeled_train': unlabeled_train_loader, 
                'validation': val_loader, 'test': test_loader}
            metrics = self._eval_all(evals_loader, ckpt_type='last', run_name=run_name, if_print=True)
        else:
            print("[exec_semi_sl] NOTE: you skipped the first phrase for supervised training.")

        # Second training phrase: labeled and unlabeled data training (mixed-ssl) / labeled data training (comparative experiments)
        mode = self.cfg['semi_training_mode'] # UD / LD / UD+LD / none
        if 'UD' in mode and 'LD' in mode:
            print('[exec_semi_sl] specify UD+LD, and start the second phrase ...')
            run_name = 'semitrain_LD_UD'
            # labeled and unlabeled data for kfold-training
            num_fold = self.cfg['ssl_kfold']
            kfold_pids = get_kfold_pids(unlabeled_train_pids, num_fold, keep_pids=labeled_train_pids, random_state=self.cfg['seed'])
            kfold_name_loaders, kfold_train_loaders = [], []
            for i, kth_pids in enumerate(kfold_pids):
                fold_name = 'fold{}_mixed_train'.format(i)
                kth_train_set = prepare_dataset(kth_pids, self.cfg)
                self.patient_id.update({fold_name: kth_train_set.pids})
                kth_train_loader = DataLoader(kth_train_set, batch_size=self.cfg['batch_size'], 
                    generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                    shuffle=True,  worker_init_fn=seed_worker
                )
                kfold_name_loaders.append(fold_name)
                kfold_train_loaders.append(kth_train_loader)
            self._run_training(self.cfg['ssl_epochs'], kfold_train_loaders, kfold_name_loaders, 
                mode='wolabel', val_loaders=val_loaders, val_name=val_name, measure_training_set=True, 
                save_ckpt=True, early_stop=True, run_name=run_name)
        elif 'LD' in mode:
            print('[exec_semi_sl] specify LD, and start the second phrase ...')
            run_name = 'semitrain_LD'
            self._run_training(self.cfg['ssl_epochs'], labeled_train_loader, 'labeled_train', 
                mode='wolabel', val_loaders=val_loaders, val_name=val_name, measure_training_set=True, 
                save_ckpt=True, early_stop=True, run_name=run_name)
        elif 'UD' in mode:
            print('[exec_semi_sl] specify UD, and start the second phrase ...')
            run_name = 'semitrain_UD'
            self._run_training(self.cfg['ssl_epochs'], unlabeled_train_loader, 'unlabeled_train', 
                mode='wolabel', val_loaders=val_loaders, val_name=val_name, measure_training_set=True, 
                save_ckpt=True, early_stop=True, run_name=run_name)
        else:
            print('[exec_semi_sl] not specify UD or LD, so skip the second phrase and \
                return metrics after the first training phrase ...')
            return metrics

        # evaluate using the model fitted on unlabeled dataset (best ckpt)
        evals_loader = {'labeled_train': labeled_train_loader, 'unlabeled_train': unlabeled_train_loader, 
            'validation': val_loader, 'test': test_loader}
        metrics = self._eval_all(evals_loader, ckpt_type='best', run_name=run_name, if_print=True)
        return metrics

def _check_configs(cfg):
    assert cfg['loss_netD'] in ['bce', 'hinge', 'wasserstein']
    assert cfg['loss_recon_norm'] in ['l1', 'l2']
    assert cfg['gen_noi_noise_dist'] in ['uniform', 'gaussian']
    assert cfg['gen_noi_hops'] + 1 == len(str(cfg['gen_noi_noise']).split('-'))
    assert cfg['disc_netx_in_dim'] == int(cfg['bcb_dims'].split('-')[0])
    assert cfg['disc_nety_in_dim'] == int(cfg['gen_dims'].split('-')[-1])
    assert cfg['disc_netx_out_dim'] == int(cfg['disc_nety_hid_dims'].split('-')[-1])
    assert cfg['ssl_resume_ckpt'] in ['last', 'best']
    noise_existing = sum(sparse_str(cfg['gen_noi_noise'])) > 0
    if noise_existing:
        assert cfg['times_test_sample'] > 1
    else:
        assert cfg['times_test_sample'] == 1
    mode = cfg['semi_training_mode']
    if 'UD' in mode and 'LD' in mode:
        cfg['ssl_es_warmup'] = cfg['ssl_kfold'] # force to be same with ssl_kfold
    else:
        cfg['ssl_es_warmup'] = 0
    if cfg['task'] == 'cont_gansurv':
        assert cfg['time_format'] in ['origin', 'ratio']
        assert cfg['gen_dims'][-2:] == '-1'
        assert (cfg['gen_out_scale'] == 'sigmoid' and cfg['time_format'] == 'ratio') or \
               (cfg['gen_out_scale'] != 'sigmoid' and cfg['time_format'] == 'origin')
        assert (cfg['time_format'] == 'ratio'  and cfg['loss_recon_gamma'] == 0) or \
               (cfg['time_format'] == 'origin' and cfg['loss_recon_gamma'] >= 1)
    elif cfg['task'] == 'disc_gansurv':
        assert cfg['time_format'] == 'quantile'
        assert cfg['gen_out_scale'] == 'sigmoid'
        assert cfg['disc_nety_in_dim'] == cfg['time_bins']
        assert cfg['log_plot'] == False
    else:
        pass
