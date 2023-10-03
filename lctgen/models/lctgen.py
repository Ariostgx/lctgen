import torch
import wandb

import numpy as np

from lctgen.core.registry import registry

from .base_model import BaseModel
from .detr_model import DETRAgentQuery
from .utils import visualize_input, visualize_input_seq
from .matcher import HungarianMatcher
from .detr_loss import SetCriterion
from .posprocess import PostProcess

@registry.register_model(name='lctgen')
class LCTGen(BaseModel):
  def __init__(self, config, metrics):
    super().__init__(config, metrics)

    self.metrics = metrics
    self._config_models()
    self._config_parameters()
    cls_num = config.MODEL.SCENE.INIT_CFG.DECODER.LANE_NUM
    
    loss_cfg = config.LOSS.DETR
    matcher = HungarianMatcher(cost_class=loss_cfg.MATCH_COST.CLASS)
    self.criterion = SetCriterion(num_classes=cls_num, matcher=matcher, weight_dict=loss_cfg.WEIGHT, eos_coef=loss_cfg.EOS_COEF, losses=loss_cfg.LOSSES, use_center_mask=loss_cfg.USE_CENTER_MASK, cfg=config)

    self.with_attribute = 'attributes' in self.config.LOSS.DETR.LOSSES
    self.pred_ego = self.config.MODEL.PREDICT_EGO
    self.pred_motion = self.config.MODEL.MOTION.ENABLE

    self.process = PostProcess(config)
  
  def _config_models(self):
    self.models = []
    self.trafficgen_model = DETRAgentQuery(self.config)
    self.trafficgen_model.train()
    self.models.append(self.trafficgen_model)

  def _format_target_for_detr(self, batch):
    b = batch['agent'].shape[0]
    targets = []

    for i in range(b):
      target = {}

      # decide whether to predict the ego vehicle
      start_idx = 0 if self.pred_ego else 1

      agent_mask = batch['agent_mask'][i]

      # gt vector label for each actor in the scenario
      target['labels'] = batch['agent_vec_index'][i, agent_mask][start_idx:]
      target['pos'] = batch['gt_long_lat'][i, target['labels'].long()].float()
      target['speed'] = batch['gt_speed'][i, target['labels'].long()].float()
      target['bbox'] = batch['gt_bbox'][i, target['labels'].long()].float()
      target['vel_heading'] = batch['gt_vel_heading'][i, target['labels'].long()].float()

      if self.config.MODEL.USE_REL_HEADING:
        target['heading'] = batch['gt_heading'][i, target['labels'].long()].float()
      else:
        target['heading'] = batch['gt_agent_heading'][i, agent_mask].float()

      # add attribute for motion prediction
      if self.pred_motion:
        motion_mask = batch['all_agent_mask'][i][1:, agent_mask]
        motion_mask = motion_mask[:, start_idx:]
        motion_mask = torch.swapaxes(motion_mask, 0, 1).unsqueeze(-1).repeat(1, 1, 2)

        target['motion_mask'] = motion_mask
        
        motion = batch['traj'][i][1:, start_idx:]
        target['motion'] = torch.swapaxes(motion, 0, 1).float()

        if 'future_heading' in batch:
          future_heading = batch['future_heading'][i][1:, start_idx:]
          target['future_heading'] = torch.swapaxes(future_heading, 0, 1).float()
          future_vel = batch['future_vel'][i][1:, start_idx:]
          target['future_vel'] = torch.swapaxes(future_vel, 0, 1).float()
      
      # sort target according to distance
      distances = torch.norm(batch['agent'][i][agent_mask][start_idx:, :2], dim=1)
      sort_idx = torch.argsort(distances)
      for attr in target:
        target[attr] = target[attr][sort_idx]
      target['distance'] = distances[sort_idx]

      targets.append(target)

    batch['targets'] = targets
    return batch
  
  def _compute_loss(self, model_output):
    loss = self.criterion(model_output, model_output['data'])
    return loss

  def _visualize(self, batch, output, mode, z_mode, batch_idx):    
    ae_output = output['{}_scene_output'.format(z_mode)]

    if len(ae_output[0]['agent']) == 0:
      return
    
    if not self.pred_motion:
      input_vis = visualize_input(batch)
      decode_vis = visualize_input(batch, agents=ae_output[0]['agent'])
    else:
      input_vis = visualize_input_seq(batch)
      decode_vis = visualize_input_seq(batch, agents=ae_output[0]['agent'], traj=ae_output[0]['traj'])
    
    # img = np.concatenate([input_vis, decode_vis], axis=1)
    output_name = 'ae' if z_mode == 'input' else z_mode

    caption = batch['text'][0]
    caption = str(caption)
    self._log_image([input_vis, decode_vis], mode, '{}_input_output'.format(output_name), caption, batch_idx)

  def _log_image(self, images, mode, name, caption="", batch_idx=0):
    log_name = 'visualize/{}_{}'.format(mode, name)
    if self.config.LOGGER == 'wandb':
      img = np.concatenate(images, axis=1)
      wandb_logger = self.logger.experiment
      wandb_logger.log({log_name: [wandb.Image(img, caption=caption)]})
    elif self.config.LOGGER == 'tsboard':
      img = np.concatenate(images, axis=1)
      log_step = self.trainer.global_step + batch_idx
      self.logger.experiment.add_image(log_name, img, global_step=log_step, dataformats='HWC')

  def _batch_forward(self, batch, mode, batch_idx):
    result = super()._batch_forward(batch, mode, batch_idx)
    vis_interval = self.config.VIS_INTERVAL

    if self.global_rank == 0 and batch_idx % vis_interval == 0 and mode in ['val', 'test']:
      self._visualize(batch, result['model_output'], mode, 'text', batch_idx)

    return result
  
  def forward(self, batch, mode):
    result = {}
    result['text_decode_output'] = self.trafficgen_model(batch)
    
    batch = self._format_target_for_detr(batch)
    result['data'] = batch

    if mode in ['val', 'test']:
      result['text_scene_output'] = self.process(result['text_decode_output'], batch, with_attribute=self.with_attribute, pred_ego=self.pred_ego, pred_motion=self.pred_motion)

    return result