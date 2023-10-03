"""
Modules to postprocess detr output of the model to actor format of Waymo Open dataset
"""
import torch
from torch import nn
import numpy as np

from trafficgen.utils.utils import get_agent_pos_from_vec, rotate, cal_rel_dir
from trafficgen.utils.visual_init import get_heatmap
from trafficgen.utils.data_process.agent_process import WaymoAgent

class PostProcess(nn.Module):
    def __init__(self, cfg) -> None:
      self.cfg = cfg
      self.use_background = cfg.LOSS.DETR.PRED_BACKGROUND
      self.remove_collision = cfg.MODEL.SCENE.REMOVE_COLLISION
      self.use_attr_gmm = cfg.MODEL.SCENE.INIT_CFG.DECODER.ATTR_GMM_ENABLE
      self.use_rel_heading = cfg.MODEL.USE_REL_HEADING
      super().__init__()

    def _convert_motion_pred_to_traj(self, output, motion, motion_prob, heading):
      trajs = []
      rel_trajs = []
      for idx in range(len(output['agent'])):
        start = output['agent'][idx].position
        if self.cfg.MODEL.MOTION.PRED_MODE == 'mlp':
          a_motion = motion[idx]
        elif self.cfg.MODEL.MOTION.PRED_MODE in ['mlp_gmm', 'mtf']:
          m_idx = np.argmax(motion_prob[idx])
          a_motion = motion[idx][m_idx]
        traj = a_motion
        rel_traj = np.concatenate([np.zeros((1, 2)), traj], axis=0)
        if self.cfg.DATASET.TRAJ_TYPE == 'xy_theta_relative':
          traj = rotate(traj[..., 0], traj[..., 1], heading[idx])
        traj = traj + start
        traj = np.concatenate([start, traj], axis=0)
        traj = traj[:, np.newaxis, :]
        trajs.append(traj)
        rel_trajs.append(rel_traj[:, np.newaxis, :])
      trajs = np.concatenate(trajs, axis=1)
      rel_trajs = np.concatenate(rel_trajs, axis=1)
      return trajs, rel_trajs
  
    def _convert_future_heading_pred(self, output, future_heading, future_vel, motion_prob, heading):
      headings = []
      vels = []
      for idx in range(len(output['agent'])):
        m_idx = np.argmax(motion_prob[idx])

        a_heading = future_heading[idx][m_idx]
        a_vel = future_vel[idx][m_idx]

        a_heading = cal_rel_dir(a_heading, -heading[idx])[:, np.newaxis, :]
        a_vel = rotate(a_vel[..., 0], a_vel[..., 1], heading[idx])[:, np.newaxis, :]

        headings.append(a_heading)
        vels.append(a_vel)
      
      headings = np.concatenate(headings, axis=1)
      vels = np.concatenate(vels, axis=1)

      return headings, vels

    def _gmm_sample(self, dists, indices):
      '''
      dists (list): list of distributions
      '''
      samples = [dists[idx].sample().cpu() for idx in indices]
      return torch.stack(samples)

    @torch.no_grad()
    def forward(self, preds, data, num_limit=None, with_attribute=False, pred_ego=False, pred_motion=False):
      '''
      preds (dict): preds of the model
        pred_logits: the classification logits (including no-object) for all queries.
          Shape= [batch_size, num_queries, num_classes] without sigmoid
      data (dict): data of the batch
      '''
      pred_logits = preds['pred_logits'].cpu()
      bz, nq, nc = pred_logits.shape
      bg_class_ind = nc - 1

      if num_limit is not None:
        bz = min(bz, num_limit)

      outputs = []
      for i in range(bz):
        output = {}
        center = data['center'][i].cpu()
        center_mask = data['center_mask'][i].cpu().numpy()
        agent_mask = data['agent_mask'][i].cpu().numpy()
        output['center'] = center
        output['file'] = data['file'][i]
        output['center_id'] = data['center_id'][i]

        # get the predicted class for each query
        # remove non_mask category before argmax
        if self.use_background:
          query_logits = pred_logits[i, :, :-1].clone()
          query_logits[:, ~center_mask] = -float('inf')
          query_logits = torch.cat([query_logits, pred_logits[i, :, -1:]], dim=-1)
          query_num = nq
          pred_indices = query_logits.argmax(-1)
        else:
          query_logits = pred_logits[i].clone()
          query_logits[:, ~center_mask] = -float('inf')
          query_num = np.sum(agent_mask)
          pred_indices = query_logits.argmax(-1)[agent_mask]

        pred_scores = query_logits.softmax(-1)[torch.arange(query_num), pred_indices]

        if self.use_background:
          bg_mask = pred_indices == bg_class_ind
          query_idx = torch.arange(query_num)[~bg_mask]
          vec_idx = pred_indices[~bg_mask]
          all_vec_scores = pred_scores[~bg_mask]
        
          # remove query idx with the same vec idx according to prediction score
          unique_vec_idx = torch.unique(vec_idx)
          unique_query_idx = []
          for vec in unique_vec_idx:
            vec_queries = query_idx[vec_idx == vec]
            vec_scores = all_vec_scores[vec_idx == vec]
            unique_query_idx.append(vec_queries[vec_scores.argmax()].item())
          query_idx = torch.tensor(unique_query_idx)
          vec_idx = unique_vec_idx
        
        else:
          query_idx = torch.arange(query_num)
          vec_idx = pred_indices
          all_vec_scores = pred_scores

        vec = center[vec_idx]
        agent_num = vec.shape[0]

        if agent_num == 0:
          output['agent'] = []
          output['agent_mask'] = []
          output['agent_vec_index'] = []
          output['probs'] = []
          output['pred_logits'] = []
          output['traj'] = []
          output['rel_traj'] = []

          outputs.append(output)
          continue
        
        if not with_attribute:
          pos = torch.zeros((agent_num, 2), dtype=torch.float32)
          speed = torch.zeros((agent_num), dtype=torch.float32)
          vel_heading = torch.zeros((agent_num), dtype=torch.float32)
          heading = torch.ones((agent_num), dtype=torch.float32) * (torch.pi / 2)
          bbox = torch.ones((agent_num, 2), dtype=torch.float32)
          bbox[:, 0] = 1.8
          bbox[:, 1] = 4.5
        else:
          speed = torch.clip(preds['pred_speed'][i,query_idx].cpu(), min=0.0).squeeze(-1)
          vel_heading = preds['pred_vel_heading'][i,query_idx].cpu().squeeze(-1)
          if not self.use_attr_gmm:
            pos = preds['pred_pos'][i,query_idx].cpu()
            heading = preds['pred_heading'][i,query_idx].cpu().squeeze(-1)
            bbox = torch.clip(preds['pred_bbox'][i,query_idx].cpu(), min=0.0)
          else:
            pos = preds['pred_pos'].sample()[i, query_idx].cpu()
            heading = torch.clip(preds['pred_heading'].sample()[i, query_idx].cpu(), min=-np.pi / 2, max=np.pi / 2)
            bbox = torch.clip(preds['pred_bbox'].sample()[i, query_idx].cpu(), min=0.1)

        all_agents = get_agent_pos_from_vec(vec, pos, speed, vel_heading, heading, bbox, self.use_rel_heading)
        agent_list = all_agents.get_list()

        if self.remove_collision:
          shapes = []
          non_collide = [0]
          poly = agent_list[0].get_polygon()[0]
          shapes.append(poly)
          for agent_i in range(1, len(agent_list)):
            intersect = False
            poly = agent_list[agent_i].get_polygon()[0]
            for shape in shapes:
              if poly.intersects(shape):
                  intersect = True
                  break
            if not intersect:
              non_collide.append(agent_i) 
              shapes.append(poly)
          agent_list = [agent_list[j] for j in non_collide]
          query_idx = torch.stack([query_idx[j] for j in non_collide])
          vec_idx = torch.stack([vec_idx[j] for j in non_collide])

        # get prob for each existing query
        probs = []
        for q_idx in query_idx:
          if self.use_background:
            prob = pred_logits[i, q_idx, :-1][center_mask].softmax(-1)
          else:
            prob = pred_logits[i, q_idx][center_mask].softmax(-1)
          probs.append(prob)

        output['agent'] = agent_list
        output['agent_vec_index'] = vec_idx.long()
        
        output['agent_mask'] = torch.ones(len(output['agent']), dtype=torch.bool)
        output['probs'] = probs
        
        if self.use_background:
          output['pred_logits'] = pred_logits[i, :, :-1].clone()
        else:
          output['pred_logits'] = pred_logits[i, :].clone()

        if pred_motion:
          motion = preds['pred_motion'][i, query_idx].cpu().numpy()
          if 'motion_prob' in preds:
            motion_prob = preds['motion_prob']
            if len(motion_prob.shape) == 2:
              motion_prob = motion_prob[None, ...]
            motion_prob = motion_prob[i, query_idx].cpu().numpy()
          else:
            motion_prob = None
          heading = np.concatenate([agent.heading for agent in agent_list])
          output['traj'], output['rel_traj'] = self._convert_motion_pred_to_traj(output, motion, motion_prob, heading)

          # TODO: add postprocessing for future heading and velocity
          if 'pred_future_heading' in preds:
            future_heading = preds['pred_future_heading'][i, query_idx].cpu().numpy()
            future_vel = preds['pred_future_vel'][i, query_idx].cpu().numpy()
            output['future_heading'], output['future_vel'] = self._convert_future_heading_pred(output, future_heading, future_vel, motion_prob, heading)
        
        # add gt ego vehicle to the scene if not pred ego
        if not pred_ego:
          output['agent'] = [WaymoAgent(data['agent'][i][:1].cpu().numpy())] + output['agent']
          output['agent_vec_index'] = torch.cat([data['agent_vec_index'][i][:1].cpu().long(), output['agent_vec_index']], dim=0)
          output['agent_mask'] = torch.cat([torch.ones(1, dtype=torch.bool), output['agent_mask']], dim=0)
          if pred_motion:
            output['traj'] = np.concatenate([data['traj'][i][:, [0]].cpu().numpy(), output['traj']], axis=1)
        
        outputs.append(output)
      return outputs