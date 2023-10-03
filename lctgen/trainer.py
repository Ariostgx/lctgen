import torch

import os
import pathlib

from datetime import datetime
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

def current_time_str():
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H:%M:%S")

    return date_time

from .datasets.utils import fc_collate_fn
from lctgen.core.registry import registry

class BaseTrainer():
    def __init__(self, config, optional_callbacks = []):
        self.config = config
        self.profier = 'simple'
        avalible_gpus = [i for i in range(torch.cuda.device_count())]

        # use all avalible GPUs
        if self.config.GPU is None or len(self.config.GPU) > torch.cuda.device_count():
            self.config['GPU'] = avalible_gpus
        
        # device config
        device_cfg = {}
        if len(self.config.GPU) > 1:
            device_cfg['strategy'] = 'ddp_find_unused_parameters_false'
            device_cfg['devices'] = self.config.GPU
            device_cfg['accelerator'] = 'gpu'
        elif len(self.config.GPU) == 1:
            device_cfg['devices'] ='auto'
            device_cfg['accelerator'] ='gpu'
        else:
            device_cfg['devices'] ='auto'
            device_cfg['accelerator'] ='cpu'
        self.device_cfg = device_cfg
        
        print('GPU: ', self.config.GPU)

        seed_everything(self.config.SEED, workers=True)
        self._config_save_dir()
        self._config_logger()
        self._config_data()
        self._config_metric()
        self._set_checkpoint_monitor()
        self._set_lightning_model()
        self._config_trainer(optional_callbacks)

    def _set_checkpoint_monitor(self):
        self.checkpoint_monitor = 'val/full_loss'
        self.checkpoint_monitor_mode = 'min'

    def _set_lightning_model(self):
        model_cls = registry.get_model(self.config.MODEL.TYPE)
        if self.config.LOAD_CHECKPOINT_MODEL:
            self.lightning_model = model_cls.load_from_checkpoint(self.config.LOAD_CHECKPOINT_PATH, config=self.config, metrics=self.metrics, strict=False)
            print('load full model from checkpoint: ', self.config.LOAD_CHECKPOINT_PATH)
        else:
            self.lightning_model = model_cls(self.config, self.metrics)

    def _config_data(self):
        task_config = self.config
        
        dataset_configs = {'train':self.config.TRAIN, 'val': self.config.VAL, 'test': self.config.TEST}
        dataset_type = task_config.DATASET.TYPE

        collate_fn = fc_collate_fn

        self.data_loaders = {}
        for mode, config in dataset_configs.items():
            dataset = registry.get_dataset(dataset_type)(task_config, config.SPLIT)
            
            # batch_size should be the effective batch size regardless of the number of GPUs
            batch_size = config.BATCH_SIZE
            if len(self.config.GPU) > 1:
                batch_size = int(batch_size / len(self.config.GPU))
            if self.config.DEBUG:
                self.data_loaders[mode] = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory = False,
                drop_last=config.DROP_LAST, num_workers=1, collate_fn=collate_fn)
            else:
                self.data_loaders[mode] = DataLoader(dataset, batch_size=batch_size, shuffle=config.SHUFFLE, pin_memory = True,
                drop_last=config.DROP_LAST, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)

    def _config_metric(self, model_sources=['model']):
        task_config = self.config
        metric_types = task_config.METRIC.TYPE

        self.metrics = {}
        for mode in ['train', 'val', 'test']:
            self.metrics[mode] = {}

            for metric_type in metric_types:
              self.metrics[mode][metric_type] = registry.get_metric(metric_type)(task_config)

    def _config_save_dir(self):
        if self.config.EXPERIMENT_NAME:
            self.exp_name = self.config.EXPERIMENT_NAME
        else:
            self.exp_name = current_time_str()

        # create save dirs
        if self.config.SAVE_DIR:
            self.save_dir = os.path.join(self.config.SAVE_DIR , self.config.EXPERIMENT_DIR, str(self.exp_name))
        else:
            self.save_dir = os.path.join(self.config.ROOT_DIR, self.config.EXPERIMENT_DIR, str(self.exp_name))

        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        print(self.save_dir)

        # save config file
        config_dir = os.path.join(self.save_dir, 'config.yaml')
        with open(config_dir, 'w') as file:
            self.config.dump(stream=file)

    def _config_logger(self):
        if self.config.LOGGER == 'wandb':
            exp_name = os.path.join(self.config.EXPERIMENT_DIR, self.config.EXPERIMENT_NAME)
            wandb_proj_name = self.config.WANDB_PROJ
            self.logger = WandbLogger(project=wandb_proj_name, name=exp_name, save_dir=self.save_dir)
        elif self.config.LOGGER == 'tsboard':
            exp_name = current_time_str()
            self.logger = TensorBoardLogger(self.save_dir, name=exp_name)

    def _config_trainer(self, optional_callbacks):
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor] + optional_callbacks
        
        if self.config.SAVE_CHECKPOINT:
            checkpoint_callback = ModelCheckpoint(
                monitor=self.checkpoint_monitor,
                mode=self.checkpoint_monitor_mode,
                auto_insert_metric_name=True,
                save_last=True,
                save_top_k=1
            )
            callbacks.append(checkpoint_callback)
            enable_checkpointing = True
        else:
            enable_checkpointing = False

        self.trainer = Trainer(max_epochs=self.config.MAX_EPOCHES,
                            log_every_n_steps=self.config.LOG_INTERVAL_STEPS,
                            check_val_every_n_epoch=self.config.VAL_INTERVAL,
                            limit_val_batches = self.config.LIMIT_VAL_BATCHES,
                            default_root_dir=self.save_dir,
                            logger=self.logger,
                            callbacks=callbacks,
                            profiler=self.profier,
                            enable_checkpointing=enable_checkpointing,
                            **self.device_cfg)

    def _config_test_trainer(self):
        test_trainer = Trainer(log_every_n_steps=self.config.LOG_INTERVAL_STEPS,
                    default_root_dir=self.save_dir,
                    logger=self.logger,
                    profiler=self.profier,
                    **self.device_cfg)
        
        return test_trainer

    def train(self):
        if self.config.LOAD_CHECKPOINT_TRAINER and self.config.LOAD_CHECKPOINT_PATH is not None:
            ckpt_path = self.config.LOAD_CHECKPOINT_PATH
            print('loading training state from checkpoint: {}'.format(ckpt_path))
        else:
            ckpt_path = None
        self.trainer.fit(self.lightning_model, self.data_loaders['train'], self.data_loaders['val'], ckpt_path=ckpt_path)
        
        try:
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
            print('perform evaluation with best checkpoint on a single GPU')
            self.trainer.test(self.lightning_model, self.data_loaders['test'], ckpt_path=ckpt_path)
        except Exception as e:
            print('test best model failed: {}'.format(e))
            print('test last model instead')
            self.trainer.test(self.lightning_model, self.data_loaders['test'])

        return self.lightning_model.metrics

    def eval(self):
        if self.config.LOAD_CHECKPOINT_TRAINER and self.config.LOAD_CHECKPOINT_PATH is not None:
            ckpt_path = self.config.LOAD_CHECKPOINT_PATH
            print('loading training state from checkpoint: {}'.format(ckpt_path))
        else:
            ckpt_path = None
        self.trainer.test(model=self.lightning_model, dataloaders=self.data_loaders['test'], ckpt_path=ckpt_path)

        return self.lightning_model.metrics