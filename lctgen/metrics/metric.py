import torch
import torch.nn as nn

# from ax import Metric
import numpy as np
from scipy.optimize import linear_sum_assignment

from torchmetrics import Accuracy, MeanMetric, Metric
from lctgen.core.registry import registry
from lctgen.datasets.description import descriptions
from trafficgen.utils.data_process.agent_process import WaymoAgent

from lctgen.models.utils import transform_traj_output_to_waymo_agent


cosine_sim = nn.CosineSimilarity(dim=2, eps=1e-6)

def normalize_angle(angle):
    if isinstance(angle, torch.Tensor):
        while not torch.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not torch.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2
        return angle

    else:
        while not np.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not np.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2

        return angle

class MMD(MeanMetric):
  def __init__(
      self,
      kernel_mul=2.0,
      kernel_num=5,
  ):
      super().__init__()
      self.kernel_num = kernel_num
      self.kernel_mul = kernel_mul
      self.fix_sigma = None

  def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
      n_samples = int(source.size()[0]) + int(target.size()[0])
      total = torch.cat([source, target], dim=0)

      total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
      total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
      L2_distance = ((total0 - total1)**2).sum(2)
      if fix_sigma:
          bandwidth = fix_sigma
      else:
          bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
      bandwidth /= kernel_mul**(kernel_num // 2)
      bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
      kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
      return sum(kernel_val)

  def update(self, source, target):
      batch_size = int(source.size()[0])
      kernels = self.guassian_kernel(
          source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
      )
      XX = kernels[:batch_size, :batch_size]
      YY = kernels[batch_size:, batch_size:]
      XY = kernels[:batch_size, batch_size:]
      YX = kernels[batch_size:, :batch_size]
      mmd_score = XX.mean() + YY.mean() - XY.mean() - YX.mean()

      super().update(mmd_score)

@registry.register_metric(name='MMD')
class MMD_All(Metric):
    def __init__(self, config):
      super().__init__()
      self.mmd_cfg = config.METRIC.MMD
      self.mmd_metrics = {}
      
      for attr in self.mmd_cfg.ATTR:
        self.mmd_metrics[attr] = MMD(self.mmd_cfg.KERNEL_MUL, self.mmd_cfg.KERNEL_NUM)
    
    def _get_model_output_scene(self, input):
      return input['text_scene_output']

    def update(self, input):
      model_output_scene = self._get_model_output_scene(input)
      B = len(model_output_scene)

      for i in range(B):
        agents = {}
        gt_agent_feats = input['data']['agent'][i][input['data']['agent_mask'][i]]

        if len(model_output_scene[i]['agent']) == 0:
           continue
        
        # remove ego agent from the list
        agents['source'] = [WaymoAgent(agent.unsqueeze(0).cpu().numpy()) for agent in gt_agent_feats]
        agents['target'] = model_output_scene[i]['agent']

        if len(agents['source']) == 0 or len(agents['target']) == 0:
          continue
        
        feats = {}
        for data_type in ['source', 'target']:
          feats[data_type] = {
          'heading': torch.tensor(normalize_angle(np.concatenate([x.heading for x in agents[data_type]], axis=0))),
          'size': torch.tensor(np.concatenate([x.length_width for x in agents[data_type]], axis=0)),
          'speed': torch.tensor(np.concatenate([x.velocity for x in agents[data_type]], axis=0)),
          'position': torch.tensor(np.concatenate([x.position for x in agents[data_type]], axis=0)),
          }
        
        for attr in self.mmd_metrics:
          self.mmd_metrics[attr].update(feats['source'][attr], feats['target'][attr])
    
    def compute(self):
      results = {}
      for attr in self.mmd_metrics:
        self.mmd_metrics[attr].to(self.device)
        results[attr] = self.mmd_metrics[attr].compute()
      return results

    def reset(self):
      for attr in self.mmd_metrics:
        self.mmd_metrics[attr].reset()


@registry.register_metric(name='traj_match')
class TrajMatch(Metric):
  def __init__(self, cfg):
    super().__init__(full_state_update=True)
    self.traj_metrics = {}
    self.traj_metrics['ade'] = MeanMetric()
    self.traj_metrics['fde'] = MeanMetric()
    self.traj_metrics['scr'] = MeanMetric()
  
  def _position_match(self, real_agents, sim_agents):
    real_positions = np.array([agent.position[0] for agent in real_agents])
    sim_positions = np.array([agent.position[0] for agent in sim_agents])

    # Hungarian algorithm according to agent position
    assignment_matrix = np.zeros((len(real_positions), len(sim_positions)))
    for i in range(len(real_positions)):
        assignment_matrix[i] = np.linalg.norm(real_positions[i] - sim_positions, axis=1)
    
    real_indices, sim_indices = linear_sum_assignment(assignment_matrix)

    return real_indices, sim_indices

  def _compute_scr(self, output_scene):
    IOU_THRESHOLD = 0.1

    if 'pred_waymo_agents_T' in output_scene:
      agents_T = output_scene['pred_waymo_agents_T']
    else:
      agents_T = transform_traj_output_to_waymo_agent(output_scene, fps=10)
      
    collide_idx = np.zeros(len(agents_T[0]), dtype=bool)

    for t in range(len(agents_T)):
      agents_t = agents_T[t]
      polygons = [agent.get_polygon()[0] for agent in agents_t]
      for idx, poly in enumerate(polygons):
        for ano_idx, another_poly in enumerate(polygons):
          if idx == ano_idx:
            continue
          union_area = poly.union(another_poly).area
          inter_area = poly.intersection(another_poly).area
          union_area = union_area if union_area > 0 else 1e-5
          iou = inter_area / union_area
          if iou > IOU_THRESHOLD:
            collide_idx[idx] = True
            collide_idx[ano_idx] = True

    return np.mean(collide_idx)

  def update(self, output):
    model_output_scene = output['text_scene_output']
    B = len(model_output_scene)
    data = output['data']

    for i in range(B):
      agents = {}
      gt_agent_feats = data['agent'][i][data['agent_mask'][i]]
      
      if len(model_output_scene[i]['agent']) == 0:
         continue
      
      agents['real'] = [WaymoAgent(agent.unsqueeze(0).cpu().numpy()) for agent in gt_agent_feats]
      agents['sim'] = model_output_scene[i]['agent']

      real_indices, sim_indices = self._position_match(agents['real'], agents['sim'])
      MSE = nn.MSELoss(reduction='none')

      motion_mask = data['all_agent_mask'][i][:, data['agent_mask'][i]]

      for real_idx, sim_idx in zip(real_indices, sim_indices):
        # remove the first timestep
        real_traj = data['traj'][i][:, data['agent_mask'][i]][:, real_idx][1:].cpu()
        sim_traj = torch.tensor(model_output_scene[i]['rel_traj'][:, sim_idx][1:]).cpu()

        real_mask = motion_mask[:, real_idx][1:].cpu()

        if not real_mask.any():
          continue
        
        last_true_idx = torch.where(real_mask)[0][-1]

        ade_all = MSE(real_traj, sim_traj).sum(dim=-1).sqrt()
        ade = ade_all[real_mask].mean()
        fde = MSE(real_traj[last_true_idx], sim_traj[last_true_idx]).sum(dim=-1).sqrt()

        self.traj_metrics['ade'].update(ade)
        self.traj_metrics['fde'].update(fde)
      
      scr = self._compute_scr(model_output_scene[i])
      self.traj_metrics['scr'].update(scr)
  
  def compute(self):
    results = {}
    for attr in self.traj_metrics:
      self.traj_metrics[attr].to(self.device)
      results[attr] = self.traj_metrics[attr].compute()
    return results

  def reset(self):
    for attr in self.traj_metrics:
      self.traj_metrics[attr].reset()
