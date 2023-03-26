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

from .BaseSurv import SurvNet
from .model_utils import init_weights, general_init_weight
from .backbone import load_backbone
from utils.func import *
from utils.io import read_datasplit_npz, read_maxt_from_table
from utils.io import save_prediction
from dataset.utils import prepare_dataset
from eval.utils import prepare_evaluator
from loss.utils import *
from optim import create_optimizer
from dataset.GraphBatchWSI import collate_MIL_graph


class BaselineHandler(object):
    """Handling baseline model initialization, training, and test.
    Baseline models use their original losses for comparisons.
    1) Patch-based Model (ESAT): Cox PLE
    2) Graph-based Model (PatchGCN): NLL
    3) Cluster-based Model (DeepAttnMISL): Cox PLE
    """
    def __init__(self, cfg):
        assert cfg['task'] in ['surv_cox', 'surv_nll', 'surv_reg']
        assert cfg['bcb_mode'] in ['patch', 'cluster', 'graph', 'abmil']

        torch.cuda.set_device(cfg['cuda_id'])
        seed_everything(cfg['seed'])

        if cfg['test']:
            cfg['test_save_path'] = cfg['test_save_path'].format(cfg['test_mask_ratio'], cfg['data_split_seed'])
            cfg['test_load_path'] = cfg['test_load_path'].format(cfg['data_split_seed'])
            if not osp.exists(cfg['test_save_path']):
                os.makedirs(cfg['test_save_path'])
            run_name = cfg['test_save_path'].split('/')[-1]
            self.writer = wandb.init(project=cfg['test_wandb_prj'], name=run_name, dir=cfg['wandb_dir'], config=cfg, reinit=True)
            self.last_net_ckpt_path = osp.join(cfg['test_load_path'], 'model-last.pth')
            self.best_net_ckpt_path = osp.join(cfg['test_load_path'], 'model-best.pth')
            self.last_metrics_path   = osp.join(cfg['test_save_path'], 'metrics-last.txt')
            self.best_metrics_path   = osp.join(cfg['test_save_path'], 'metrics-best.txt')
            self.config_path    = osp.join(cfg['test_save_path'], 'print_config.txt')
        else:
            if not osp.exists(cfg['save_path']):
                os.makedirs(cfg['save_path'])
            run_name = cfg['save_path'].split('/')[-1]
            self.writer = wandb.init(project=cfg['wandb_prj'], name=run_name, dir=cfg['wandb_dir'], config=cfg, reinit=True)
            self.last_net_ckpt_path = osp.join(cfg['save_path'], 'model-last.pth')
            self.best_net_ckpt_path = osp.join(cfg['save_path'], 'model-best.pth')
            self.last_metrics_path  = osp.join(cfg['save_path'], 'metrics-last.txt')
            self.best_metrics_path  = osp.join(cfg['save_path'], 'metrics-best.txt')
            self.config_path    = osp.join(cfg['save_path'], 'print_config.txt')

        self.bcb = cfg['bcb_mode'] 
        self.task = cfg['task'] 
        # infer bcb_mode, out_scale, and time_format according to task
        if self.task == 'surv_nll':
            out_scale = 'sigmoid'
            cfg['time_format'] = 'quantile'
        elif self.task == 'surv_reg':
            out_scale = 'sigmoid' # when output ratio
            cfg['time_format'] = 'ratio'
        elif self.task == 'surv_cox':
            out_scale = 'none'
            cfg['time_format'] = 'origin'
        else:
            pass

        # setup baseline model
        backbone_dims = sparse_str(cfg['bcb_dims']) # cfg backbone
        backbone = load_backbone(self.bcb, backbone_dims)
        dim_in, dim_out = sparse_str(cfg['pdh_dims']) # cfg prediction head
        self.net = SurvNet(dim_in, dim_out, backbone, hops=cfg['mlp_hops'], norm=cfg['mlp_norm'],
            dropout=cfg['mlp_dropout'], out_scale=out_scale)
        if self.task == 'surv_reg' or self.task == 'surv_nll': # indicate out_scale = Sigmoid
            self.net.apply(init_weights) # apply model initialization (keep the same as `model_handler`)
        else:
            self.net.apply(general_init_weight)
        self.net = self.net.cuda()
        
        # loss
        if self.task == 'surv_nll':
            self.supervised_loss = SurvMLE(**sparse_key(cfg, prefixes='loss_mle'))
        elif self.task == 'surv_cox':
            self.supervised_loss = SurvPLE()
        elif self.task == 'surv_reg':
            if self.bcb == 'patch':
                # following ESAT's implementation 
                # https://github.com/notbadforme/ESAT/blob/main/esat/trainforesat.py#L111
                self.supervised_loss = partial(MSE_loss, include_censored=cfg['loss_use_censored'])
            else:
                self.supervised_loss = partial(recon_loss, **sparse_key(cfg, prefixes='loss_recon'))
        else:
            pass
        self.loss_l1 = loss_reg_l1(cfg['loss_regl1_coef'])
        # optimizer
        cfg_optimizer = SimpleNamespace(opt=cfg['opt_net'], weight_decay=cfg['opt_net_weight_decay'], lr=cfg['opt_net_lr'], 
            opt_eps=None, opt_betas=None, momentum=None)
        self.optimizer = create_optimizer(cfg_optimizer, self.net)
        # LR scheduler for net
        self.steplr = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # setup evaluator
        if cfg['time_format'] in ['origin', 'quantile']:
            end_time = read_maxt_from_table(cfg['path_label'])
        else:
            end_time = 1.0
        if cfg['task'] == 'surv_reg':
            self.evaluator = prepare_evaluator('continuous', 
                end_time=end_time, recon_loss=self.supervised_loss,
            )
            self.metrics_list = ['c_index', 'loss_recon', 'mae', 'event_t_rae', 'nonevent_t_rae', 'event_t_nre', 'nonevent_t_nre']
            # the printed loss is measured by alpha = 0.0, and the optimized loss is measured by the alpha one set previously
            self.ret_metrics = ['c_index', 'loss_recon']
        elif cfg['task'] == 'surv_nll':
            self.evaluator = prepare_evaluator('discrete', mle_loss=self.supervised_loss)
            self.metrics_list = ['c_index', 'loss_mle', 'loss_mle_org']
            # the printed loss is measured by alpha = 0.0, and the optimized loss is measured by the alpha one set previously
            self.ret_metrics = ['c_index', 'loss_mle_org']
            self.nbins = cfg['time_bins']
        elif cfg['task'] == 'surv_cox':
            self.evaluator = prepare_evaluator('prohazard', ple_loss=self.supervised_loss)
            self.metrics_list = ['c_index', 'loss_ple']
            self.ret_metrics = ['c_index', 'loss_ple']
        else:
            pass

        self.patient_id = dict()
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
    
    def exec_test(self):
        print('[exec] execute test {} using backbone-mode {}.'.format(self.task, self.bcb))
        print('[exec] start testing {}.'.format(self.cfg['test_path']))
        mode_name = 'test_mode'
        
        # Prepare datasets 
        path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        if self.cfg['test_path'] == 'train':
            pids = pids_train
        elif self.cfg['test_path'] == 'val':
            pids = pids_val
        elif self.cfg['test_path'] == 'test':
            pids = pids_test
        else:
            pass
        print('[exec] test patient IDs from {}'.format(self.cfg['test_path']))

        if self.bcb == 'graph':
            collate = collate_MIL_graph
        else:
            collate = default_collate

        # Prepare datasets 
        test_set = prepare_dataset(pids, self.cfg, mask_ratio=self.cfg['test_mask_ratio'])
        self.patient_id.update({'exec-test': test_set.pids})
        test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False, worker_init_fn=seed_worker, collate_fn=collate
        )

        # Evals
        evals_loader = {'exec-test': test_loader}
        metrics = self._eval_all(evals_loader, ckpt_type='best', if_print=True, test_mode=True, test_mode_name=mode_name)
        return metrics

    def _run_training(self, epochs, train_loader, name_loader, val_loaders=None, val_name=None, 
        measure_training_set=True, save_ckpt=True, early_stop=False, run_name='train', **kws):
        """Traing model.

        Args:
            epochs (int): Epochs to run.
            train_loader ('DataLoader'): DatasetLoader of training set.
            name_loader (string): name of train_loader, used for infering patient IDs.
            val_loaders (dict): A dict like {'val': loader1, 'test': loader2}, which gives the datasets
                to evaluate at each epoch.
            val_name (string): The dataset used to perform early stopping and optimal model saving.
            measure_training_set (bool): If measure training set at each epoch.
            save_ckpt (bool): If save models.
            early_stop (bool): If early stopping according to validation loss.
            run_name (string): Name of this training, which would be used as the prefixed name of ckpt files.
        """
        if early_stop and self.cfg['es_patience'] is not None:
            self.early_stop = EarlyStopping(warmup=self.cfg['es_warmup'], patience=self.cfg['es_patience'], 
                start_epoch=self.cfg['es_start_epoch'], verbose=self.cfg['es_verbose'])
        else:
            self.early_stop = None

        if val_name is not None and self.early_stop is not None:
            assert val_name in val_loaders.keys(), "Not specify a dataloader to enable early stopping."
            print("[{}] {} epochs w early stopping on {}.".format(run_name, epochs, val_name))
        else:
            print("[{}] {} epochs w/o early stopping.".format(run_name, epochs))
        
        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch + 1
            train_cltor = self._train_each_epoch(train_loader, name_loader)
            cur_name = name_loader

            if measure_training_set:
                self._eval_and_print(train_cltor, name=cur_name, at_epoch=epoch+1)

            # val/test
            val_metrics = None
            if val_loaders is not None:
                for k in val_loaders.keys():
                    if val_loaders[k] is None:
                        continue
                    val_cltor = self.test_model(self.net, self.bcb, val_loaders[k], times_test_sample=1)
                    met_ci, met_loss = self._eval_and_print(val_cltor, name=k, at_epoch=epoch+1)
                    if k == val_name:
                        val_metrics = met_ci if self.cfg['monitor_metrics'] == 'ci' else met_loss
            
            # early_stop using VAL_METRICS (**is measured using alpha=0.0**)
            if val_metrics is not None and self.early_stop is not None:
                self.steplr.step(val_metrics)
                self.early_stop(epoch, val_metrics)
                if self.early_stop.if_save_checkpoint():
                    self.save_model(epoch+1, ckpt_type='best', run_name=run_name)
                    print("[{}] best model saved at epoch {}".format(run_name, epoch+1))
                if self.early_stop.if_stop():
                    break
        
        if save_ckpt:
            self.save_model(last_epoch, ckpt_type='last', run_name=run_name) # save models and optimizers
            print("[{}] last model saved at epoch {}".format(run_name, last_epoch))

    def _train_each_epoch(self, train_loader, name_loader):
        print("[train] train one epoch using train_loader={}".format(name_loader))
        bp_every_batch = self.cfg['bp_every_batch']
        train_cltor = {'y': None, 'y_hat': None}
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
                # update network (return preds, a detached one in CPU)
                cur_preds = self._update_network(i_batch, x_collector, y_collector)

                # collect preditions
                train_cltor = agg_tensor(
                    train_cltor, 
                    {'y': torch.cat(y_collector, dim=0).detach().cpu(),
                     'y_hat': cur_preds.detach().cpu(),
                    }
                )

                # reset vars
                i_collector = []
                x_collector = []
                y_collector = []
                torch.cuda.empty_cache()

        return train_cltor

    def _update_network(self, i_batch, xs, ys):
        self.net.train()

        n_sample = len(xs)
        f_fake_collector, f_fake_list, preds = [], [], []

        for i in range(n_sample):
            data_x, data_x_ext, data_t, data_ind = xs[i][0], xs[i][1], ys[i][:, [0]], ys[i][:, [1]]

            if self.bcb == 'graph':
                data_x = data_x.unsqueeze(0)
                pred = self.net(data_x_ext, None) # data_x_ext -> GraphData if backbone=graph
            elif self.bcb == 'patch':
                pred = self.net(data_x, None) # skip coords if backbone=patch
                # pred = self.netG(data_x, data_x_ext)
            else:
                pred = self.net(data_x, data_x_ext) # generate pred given data_x
            # print("{}-th sample: {}".format(i, pred))
            preds.append(pred)

        # 3.1 zero gradients buffer
        self.optimizer.zero_grad()

        # 3.2 loss
        cur_preds = torch.cat(preds, dim=0)
        label_t = torch.cat([ys[i][:,[0]] for i in range(n_sample)], dim=0)
        label_ind = torch.cat([ys[i][:,[1]] for i in range(n_sample)], dim=0)
        net_loss = self.supervised_loss(cur_preds, label_t, label_ind) 
        total_loss = net_loss + self.loss_l1(self.net.parameters())
        print("[training one epoch] training batch {}, network_loss: {:.6f}, totol_loss: {:.6f}".\
            format(i_batch, net_loss.item(), total_loss.item()))
        wandb.log({
            'train_batch/net/loss_supervision': net_loss.item(), 
            'train_batch/net/loss_total': total_loss.item(),
        })

        # 3.3 backwards gradients and update networks
        total_loss.backward()
        self.optimizer.step()

        return cur_preds

    def _eval_all(self, evals_loader, ckpt_type='best', run_name='train', if_print=True, 
        test_mode=False, test_mode_name='test_mode'):
        """
        test_mode=True only if run self.exec_test(), indicating a test mode.
        """
        if test_mode:
            print('[warning] you are in test mode now.')
            ckpt_run_name = 'train'
            wandb_group_name = test_mode_name
            metrics_path_name = test_mode_name
            csv_prefix_name = test_mode_name
            sampling_times = self.cfg['test_sampling_times']
        else:
            ckpt_run_name = run_name
            wandb_group_name = run_name
            metrics_path_name = run_name
            csv_prefix_name = run_name
            sampling_times = 1
        
        if ckpt_type == 'best':
            ckpt = add_prefix_to_filename(self.best_net_ckpt_path, ckpt_run_name)
            wandb_group = 'bestckpt/{}'.format(wandb_group_name)
            print_path = add_prefix_to_filename(self.best_metrics_path, metrics_path_name)
            csv_name = '{}_best'.format(csv_prefix_name)
        elif ckpt_type == 'last':
            ckpt = add_prefix_to_filename(self.last_net_ckpt_path, ckpt_run_name)
            wandb_group = 'lastckpt/{}'.format(wandb_group_name)
            print_path = add_prefix_to_filename(self.last_metrics_path, metrics_path_name)
            csv_name = '{}_last'.format(csv_prefix_name)
        else:
            pass

        metrics = dict()
        for k, loader in evals_loader.items():
            if loader is None:
                continue
            cltor = self.test_model(self.net, self.bcb, loader, 
                times_test_sample=sampling_times, checkpoint=ckpt,
            )
            ci, loss = self._eval_and_print(cltor, name='{}/{}'.format(wandb_group, k))
            metrics[k] = [('cindex', ci), ('loss', loss)]

            cur_y = cltor['y']
            if 'avg_y_hat' in cltor:
                cur_y_hat = cltor['avg_y_hat']
            else:
                cur_y_hat = cltor['y_hat']
            if 'dist_y_hat' in cltor:
                dist_y_hat = cltor['dist_y_hat']
            else:
                dist_y_hat = None

            if self.cfg['save_prediction']:
                dir_save_pred = self.cfg['save_path'] if not test_mode else self.cfg['test_save_path']
                path_save_pred = osp.join(dir_save_pred, '{}_pred_{}.csv'.format(csv_name, k))
                patient_ids = self._get_patient_id(k, cltor['idx'])
                save_prediction(patient_ids, cur_y, cur_y_hat, dist_y_hat, path_save_pred)

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

    @staticmethod
    def test_model(model, backbone, loader, times_test_sample=1, checkpoint=None):
        if checkpoint is not None:
            net_ckpt = torch.load(checkpoint)
            model.load_state_dict(net_ckpt['model'])
        model.eval()
        res = {'idx': None, 'y': None, 'y_hat': None}
        with torch.no_grad():
            for idx, x, y in loader:
                x_data, x_ext = [_x.cuda() for _x in x]
                if backbone == 'graph':
                    x_data = x_data.unsqueeze(0)
                    y_hat = model(x_ext, None)
                elif backbone == 'patch':
                    y_hat = model(x_data, None) # skip coords if backbone=patch
                    # y_hat = model(x_data, x_ext)
                else:
                    y_hat = model(x_data, x_ext)
                res = agg_tensor(res, 
                    {'idx': idx.detach().cpu(), 'y': y.detach().cpu(), 
                     'y_hat': y_hat.detach().cpu()}
                )
                if times_test_sample > 1:
                    # run the model for [times_test_sample] times and use its median as final predition
                    # y_hat_list: [times_test_sample, B, out_dim]
                    y_hat_list = []
                    for i in range(times_test_sample):
                        if backbone == 'graph':
                            cur_y_hat = model(x_ext, None)
                        elif backbone == 'patch':
                            cur_y_hat = model(x_data, None) # skip coords if backbone=patch
                            # cur_y_hat = modelG(x_data, x_ext)
                        else:
                            cur_y_hat = model(x_data, x_ext)
                        y_hat_list.append(cur_y_hat)
                    y_hat_list = torch.stack(y_hat_list)
                    res = agg_tensor(res, {'dist_y_hat': y_hat_list.transpose(0,1).detach().cpu()}) # [B, times_test, out_dim]
                    avg_y_hat, _ = torch.median(y_hat_list, dim=0)
                    # avg_y_hat = torch.mean(y_hat_list, dim=0)
                    res = agg_tensor(res, {'avg_y_hat': avg_y_hat.detach().cpu()})

        return res

    def _get_state_dict(self, epoch):
        return {
            'epoch': epoch,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def save_model(self, epoch, ckpt_type='best', run_name='train'):
        net_ckpt_dict = self._get_state_dict(epoch)
        if ckpt_type == 'last':
            torch.save(net_ckpt_dict, add_prefix_to_filename(self.last_net_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            torch.save(net_ckpt_dict, add_prefix_to_filename(self.best_net_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))

    def resume_model(self, ckpt_type='best', run_name='train'):
        if ckpt_type == 'last':
            net_ckpt = torch.load(add_prefix_to_filename(self.last_net_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            net_ckpt = torch.load(add_prefix_to_filename(self.best_net_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))
        self.net.load_state_dict(net_ckpt['model']) 
        self.optimizer.load_state_dict(net_ckpt['optimizer']) 
        print('[model] resume the netG from {}_{} at epoch {}...'.format(ckpt_type, run_name, net_ckpt['epoch']))
