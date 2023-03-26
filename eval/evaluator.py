import torch
import torch.nn.functional as F

from loss.utils import *
from .cindex import concordance_index

#######################################################################
# Evaluator for measuring model performance (i.e., C-Index and MAE)
#######################################################################

class ContSurv_Evaluator(object):
    """Performance evaluator for continuous survival model"""
    def __init__(self, **kws):
        super(ContSurv_Evaluator, self).__init__()
        self.kws = kws
        self.end_time = kws['end_time']
        self.valid_functions = {
            'c_index': self._c_index,
            'loss_rank': self._rank_loss,
            'loss_recon': self._recon_loss,
            'loss_recon_org': self._recon_loss_org,
            'loss_fake_netD': self._loss_fake_dis,
            'loss_fake_netG': self._loss_fake_gen,
            'avg_fake': self._avg_fake,
            'event_t_rae': self._evt_t_rae,
            'nonevent_t_rae': self._noevt_t_rae,
            'event_t_nre': self._evt_t_nre,
            'nonevent_t_nre': self._noevt_t_nre,
            'mae': self._mae,
        }
        self.valid_metrics = ['c_index', 'loss_rank', 'loss_recon', 'loss_recon_org', 'loss_fake_netD', 'loss_fake_netG',
            'avg_fake', 'event_t_rae', 'nonevent_t_rae', 'event_t_nre', 'nonevent_t_nre', 'mae']

    def _check_metrics(self, metrics):
        for m in metrics:
            assert m in self.valid_metrics

    def _pre_compute(self, data):
        self.y = data['y']
        self.t = data['y'][:, 0]
        self.e = data['y'][:, 1]
        if 'f_fake' in data:
            self.f_fake = data['f_fake'].squeeze()
        else:
            self.f_fake = None
        # only used for computing CI
        if 'avg_y_hat' in data:
            self.y_hat = data['avg_y_hat'].squeeze()
            self.avg_y_hat = data['avg_y_hat'].squeeze()
        else:
            self.y_hat = data['y_hat'].squeeze()
            self.avg_y_hat = data['y_hat'].squeeze()

    def _c_index(self):
        y_true = self.y.numpy()
        y_pred = self.avg_y_hat.unsqueeze(-1).numpy()
        return concordance_index(y_true, y_pred)

    def _rank_loss(self):
        if 'rank_loss' not in self.kws:
            return 0
        elif self.kws['rank_loss'] is None:
            return 0
        else:
            return self.kws['rank_loss'](self.y_hat, self.t, self.e).item()

    def _recon_loss(self):
        if 'recon_loss' not in self.kws:
            return 0
        elif self.kws['recon_loss'] is None:
            return 0
        else:
            return self.kws['recon_loss'](self.y_hat, self.t, self.e).item()

    def _recon_loss_org(self):
        if 'recon_loss' not in self.kws:
            return 0
        elif self.kws['recon_loss'] is None:
            return 0
        else:
            return self.kws['recon_loss'](self.y_hat, self.t, self.e, cur_alpha=0.0).item()

    def _mae(self):
        return recon_loss(self.y_hat, self.t, self.e, cur_alpha=0.0).item()

    def _loss_fake_dis(self):
        if 'disc_loss' not in self.kws:
            return 0
        elif self.kws['disc_loss'] is None:
            return 0
        else:
            return self.kws['disc_loss'](None, self.f_fake).item()

    def _loss_fake_gen(self):
        return fake_generator_loss(self.f_fake).item()

    def _avg_fake(self):
        return torch.mean(self.f_fake).item()

    def _evt_t_rae(self):
        """Ones with event, RAE = relative absolute error"""
        idcs = self.e == 1
        diff = self.t[idcs] - self.y_hat[idcs]
        return torch.mean(torch.abs(diff) / self.end_time).item()

    def _noevt_t_rae(self):
        """Ones without event, RAE = relative absolute error"""
        idcs = self.e == 0
        diff = self.t[idcs] - self.y_hat[idcs]
        return torch.mean(F.relu(diff) / self.end_time).item()

    def _evt_t_nre(self):
        """Ones with event, NRE = normlized relative error"""
        idcs = self.e == 1
        diff = self.y_hat[idcs] - self.t[idcs]
        return torch.mean(diff / self.end_time).item()

    def _noevt_t_nre(self):
        """Ones without event, NRE = normlized relative error"""
        idcs = self.e == 0
        diff = self.y_hat[idcs] - self.t[idcs]
        return torch.mean(-F.relu(-diff) / self.end_time).item()

    def compute(self, data, metrics):
        self._check_metrics(metrics)
        self._pre_compute(data)
        res_metrics = dict()
        for m in metrics:
            res_metrics[m] = self.valid_functions[m]()
        return res_metrics


class DiscSurv_Evaluator(object):
    """docstring for DiscSurv_Evaluator.
    cltor looks like {'y': *, 'y_hat': *, 'f_fake': *, 'avg_y_hat': *}
    y_hat: [discrete_time, event indicator]
    """
    def __init__(self, **kws):
        self.kws = kws
        self.valid_functions = {
            'c_index': self._c_index,
            'loss_mle': self._loss_mle,
            'loss_mle_org': self._loss_mle_org,
            'loss_fake_netD': self._loss_fake_dis,
            'loss_fake_netG': self._loss_fake_gen,
            'avg_fake': self._avg_fake,
        }
        self.valid_metrics = ['c_index', 'loss_mle', 'loss_mle_org', 'loss_fake_netD', 
            'loss_fake_netG', 'avg_fake']

    def _check_metrics(self, metrics):
        for m in metrics:
            assert m in self.valid_metrics

    def _pre_compute(self, data):
        self.y = data['y']
        self.t = data['y'][:, 0]
        self.e = data['y'][:, 1]
        self.c = 1.0 - data['y'][:, 1]
        if 'f_fake' in data:
            self.f_fake = data['f_fake'].squeeze()
        else:
            self.f_fake = None
        # only used for computing CI
        if 'avg_y_hat' in data:
            self.y_hat = data['avg_y_hat']
            self.avg_y_hat = data['avg_y_hat']
        else:
            self.y_hat = data['y_hat']
            self.avg_y_hat = data['y_hat']
        
        self.survival = torch.cumprod(1.0 - self.avg_y_hat, dim=1)
        self.risk = torch.sum(self.survival, dim=1)

    def _c_index(self):
        y_true = self.y.numpy()
        y_pred = self.avg_y_hat.numpy()
        return concordance_index(y_true, y_pred)

    def _loss_mle(self):
        assert 'mle_loss' in self.kws
        loss = self.kws['mle_loss'](self.y_hat, self.t, self.e)
        return loss.item()

    def _loss_mle_org(self):
        assert 'mle_loss' in self.kws
        loss = self.kws['mle_loss'](self.y_hat, self.t, self.e, cur_alpha=0.0)
        return loss.item()

    def _loss_fake_dis(self):
        if 'disc_loss' not in self.kws:
            return 0
        elif self.kws['disc_loss'] is None:
            return 0
        else:
            return self.kws['disc_loss'](None, self.f_fake).item()

    def _loss_fake_gen(self):
        return fake_generator_loss(self.f_fake).item()

    def _avg_fake(self):
        return torch.mean(self.f_fake).item()

    def compute(self, data, metrics):
        self._check_metrics(metrics)
        self._pre_compute(data)
        res_metrics = dict()
        for m in metrics:
            res_metrics[m] = self.valid_functions[m]()
        return res_metrics


class CoxSurv_Evaluator(object):
    """Performance evaluator for Cox-based survival model"""
    def __init__(self, **kws):
        super(CoxSurv_Evaluator, self).__init__()
        self.kws = kws
        self.valid_functions = {
            'c_index': self._c_index,
            'loss_ple': self._ple_loss,
        }
        self.valid_metrics = ['c_index', 'loss_ple']

    def _check_metrics(self, metrics):
        for m in metrics:
            assert m in self.valid_metrics

    def _pre_compute(self, data):
        self.y = data['y']
        self.t = data['y'][:, 0]
        self.e = data['y'][:, 1]
        # only used for computing CI
        if 'avg_y_hat' in data:
            self.y_hat = data['avg_y_hat'].squeeze()
            self.avg_y_hat = data['avg_y_hat'].squeeze()
        else:
            self.y_hat = data['y_hat'].squeeze()
            self.avg_y_hat = data['y_hat'].squeeze()

    def _c_index(self):
        y_true = self.y.numpy()
        y_pred = self.avg_y_hat.unsqueeze(-1).numpy()
        return concordance_index(y_true, y_pred)

    def _ple_loss(self):
        if 'ple_loss' not in self.kws:
            return 0
        elif self.kws['ple_loss'] is None:
            return 0
        else:
            return self.kws['ple_loss'](self.y_hat, self.t, self.e).item()

    def compute(self, data, metrics):
        self._check_metrics(metrics)
        self._pre_compute(data)
        res_metrics = dict()
        for m in metrics:
            res_metrics[m] = self.valid_functions[m]()
        return res_metrics
