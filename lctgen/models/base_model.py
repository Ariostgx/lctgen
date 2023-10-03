import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR, SequentialLR, LambdaLR
# from lion_pytorch import Lion

import numpy as np

import wandb
import pytorch_lightning as pl

from lctgen.core.registry import registry
from .loss import loss_funcs

class BaseModel(pl.LightningModule):
    def __init__(self, config, metrics):
      super().__init__()
      self.config = config
      self.metrics = metrics
      self.class_mean_imu_abs = None

      for mode in ['train', 'val', 'test']:
          if mode in self.metrics:
              for metric_name in self.metrics[mode].keys():
                  setattr(self, 'metric_{}_{}'.format(mode, metric_name), self.metrics[mode][metric_name])
      self._config_loss_functions()
      self.lr = self.config.TRAIN.LR

    def _config_parameters(self):
      self.model_params = []
      self.mul_model_params = []

      mul_model = self.config.TRAIN.LR_MUL_MODEL
      mul_lr_factor = self.config.TRAIN.LR_MUL_FACTOR

      for model in self.models:
        model_name = str(type(model))
        if mul_model and mul_model in model_name:
           self.mul_model_params += [p for p in model.parameters()]
        else:
           self.model_params += [p for p in model.parameters()]
      
      self.params = [{'params': self.model_params, 'lr': self.lr}]
      if len(self.mul_model_params) > 0:
        self.params += [{'params': self.mul_model_params, 'lr': self.lr * mul_lr_factor}]

    def _config_loss_functions(self):
      self.loss_func = loss_funcs[self.config.LOSS.TYPE]

    def _compute_loss(self, model_output):
      loss = self.loss_func(model_output, self.config)
      return loss

    def forward(self, batch):
      raise NotImplementedError
    
    def _batch_forward(self, batch, mode, batch_idx):
      model_output = self.forward(batch, mode)

      loss = self._compute_loss(model_output)

      if type(loss) is not dict:
        loss = {'full_loss': loss}

      if mode == 'train':
          on_step = True
          on_epoch = False
      else:
          on_step = False
          on_epoch = True

      sync_dist = True if len(self.config.GPU) > 1 else False
      
      for loss_name, value in loss.items():
        self.log('{}/{}'.format(mode, loss_name), value, on_step=on_step,
                    on_epoch=on_epoch, sync_dist=sync_dist)
      
      if mode in ['val', 'test']:
        for metric_name in self.metrics[mode].keys():
            try:
                self.metrics[mode][metric_name](model_output)
                metric_value = self.metrics[mode][metric_name].compute()
                if type(metric_value) is dict:
                    for subname, subvalue in metric_value.items():
                        self.log('{}/metric-{}-{}'.format(mode, metric_name, subname), subvalue, on_epoch=on_epoch, on_step=on_step, sync_dist=sync_dist)
                else:
                    self.log('{}/metric-{}'.format(mode, metric_name), self.metrics[mode][metric_name], on_epoch=on_epoch, on_step=on_step, sync_dist=sync_dist)
            except Exception as e:
                logging.warning('Error when computing metric {}: {}'.format(metric_name, e))
                continue

      return {'loss': loss['full_loss'], 'model_output': model_output}
    
    def training_step(self, batch, batch_idx):
        return self._batch_forward(batch, 'train', batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._batch_forward(batch, 'val', batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self._batch_forward(batch, 'test', batch_idx)

    def _configure_schedulers(self, optimizer, scheduler_config):
      scheduler_name = scheduler_config.TYPE
      
      if scheduler_name == 'StepLR':
          scheduler = StepLR(
              optimizer, step_size=scheduler_config.STEP, gamma=scheduler_config.GAMMA)
          
          print('using StepLR scheduler')
      
      elif scheduler_name == 'MultiStepLR':
          scheduler = MultiStepLR(
              optimizer, milestones=scheduler_config.MILESTONES, gamma=scheduler_config.GAMMA)
          
          print('using multistep lr scheduler')

      elif scheduler_name == 'PatienceMultiplicativeLR':
          patience_epoch = scheduler_config.STEP

          def patience_gamma(epoch):
              if epoch >= patience_epoch:
                  return scheduler_config.GAMMA ** (epoch - patience_epoch + 1)
              else:
                  return 1.0

          scheduler = LambdaLR(optimizer, lr_lambda=patience_gamma)

          print('using patience scheduler: GAMMA = {}, start_step = {}'.format(scheduler_config.GAMMA, patience_epoch))

      elif scheduler_name == 'CosineAnneling':
          eta_min = scheduler_config.ETA_MIN
          T_max = self.config.MAX_EPOCHES

          scheduler = CosineAnnealingLR(
              optimizer, T_max=T_max, eta_min=eta_min)
          
          print('using CosineAnnealingLR: eta_min={}, T_max={}'.format(eta_min, T_max))

      else:
          raise ValueError(
              'Invalid scheduler name: {}'.format(scheduler_name))
      
      return scheduler

    def configure_optimizers(self):
      optimizer_name = self.config.TRAIN.OPTIMIZER
      lr = self.lr

      scheduler_config = self.config.TRAIN.SCHEDULER

      if len(self.params) == 0:
          return

      if optimizer_name == 'Adam':
          optimizer = torch.optim.Adam(self.params, lr=lr, betas=(0.9, 0.999), weight_decay=self.config.TRAIN.WEIGHT_DECAY, amsgrad=True, eps=1e-09)
          print('using Adam optimizer: lr={}, weight_decay={}'.format(lr, self.config.TRAIN.WEIGHT_DECAY))
      elif optimizer_name == 'AdamW':
          optimizer = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.999), weight_decay=self.config.TRAIN.WEIGHT_DECAY, amsgrad=True, eps=1e-09)
          print('using AdamW optimizer: lr={}, weight_decay={}'.format(lr, self.config.TRAIN.WEIGHT_DECAY))
      elif optimizer_name == 'SGD':
          optimizer = torch.optim.SGD(self.params, lr=lr, momentum=self.config.TRAIN.MOMENTUM, weight_decay=self.config.TRAIN.WEIGHT_DECAY, nesterov=self.config.TRAIN.NESTEROV)
          print('using SGD optimizer: lr={}, momentum={}, weight_decay={}, nesterov={}'.format(lr, self.config.TRAIN.MOMENTUM, self.config.TRAIN.WEIGHT_DECAY, self.config.TRAIN.NESTEROV))
      else:
          raise ValueError(
              'Invalid optimizer name: {}'.format(optimizer_name))

      scheduler = self._configure_schedulers(optimizer, scheduler_config)

      return {'optimizer': optimizer, 'lr_scheduler': scheduler}