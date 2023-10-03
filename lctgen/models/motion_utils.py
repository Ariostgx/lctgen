import torch
import math
from trafficgen.utils.utils import get_agent_pos_from_vec, rotate

def extract_query_centers(preds, data, bev_range=50.0):
  pred_logits = preds['pred_logits']
  b = pred_logits.shape[0]
  all_positions = []
  all_headings = []
  for i in range(b):
    center = data['center'][i]
    center_mask = data['center_mask'][i]
    if pred_logits.shape[-1] > center_mask.shape[-1]:
      query_logits = pred_logits[i, :, :-1].clone()
    else:
      query_logits = pred_logits[i, :].clone()
    query_logits[:, ~center_mask] = -float('inf')
    pred_indices = query_logits.argmax(-1)

    pos = torch.clip(preds['pred_pos'][i].cpu(), min=-0.5, max=0.5)
    speed = torch.clip(preds['pred_speed'][i].cpu(), min=0).squeeze(-1)
    vel_heading = torch.clip(preds['pred_vel_heading'][i].cpu(), min=-torch.pi/2,max=torch.pi/2).squeeze(-1)
    heading = torch.clip(preds['pred_heading'][i].cpu(), min=-torch.pi/2,max=torch.pi/2).squeeze(-1)
    bbox = torch.clip(preds['pred_bbox'][i].cpu(), min=0.1)

    vec = center[pred_indices].cpu()
    all_agents = get_agent_pos_from_vec(vec, pos, speed, vel_heading, heading, bbox)

    all_positions.append(torch.tensor(all_agents.position))
    all_headings.append(torch.tensor(all_agents.heading))

  all_positions = torch.stack(all_positions, dim=0)
  all_positions = all_positions / bev_range
  all_positions = all_positions.to(center.device)

  all_headings = torch.stack(all_headings, dim=0)
  all_headings = all_headings.to(center.device)

  return all_positions, all_headings

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
  """
  Copied from https://github.com/OpenDriveLab/UniAD/blob/main/projects/mmdet3d_plugin/models/utils/functional.py
  Convert 2D position into positional embeddings.

  Args:
      pos (torch.Tensor): Input 2D position tensor.
      num_pos_feats (int, optional): Number of positional features. Default is 128.
      temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

  Returns:
      torch.Tensor: Positional embeddings tensor.
  """
  scale = 2 * math.pi
  pos = pos * scale
  dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
  dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
  pos_x = pos[..., 0, None] / dim_t
  pos_y = pos[..., 1, None] / dim_t
  pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
  pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
  posemb = torch.cat((pos_y, pos_x), dim=-1)
  return posemb

def anchor_coordinate_transform(anchors, bbox_centers, headings, with_translation_transform=True, with_rotation_transform=True):
  batch_size = len(bbox_centers)
  batched_anchors = []
  for i in range(batch_size):
    N = bbox_centers[i].shape[0]
    P = anchors.shape[0]
    transformed_anchors = anchors[None].clone()
    transformed_anchors = transformed_anchors.repeat(N, 1, 1, 1)

    if with_rotation_transform:
      for p in range(P):
        transformed_anchors[:,p,:,:2] = rotate(transformed_anchors[:,p,:,0], transformed_anchors[:,p,:,1], headings[i])
    
    if with_translation_transform:
      transformed_anchors += bbox_centers[i][:,None,None,:]

    batched_anchors.append(transformed_anchors)
  
  return torch.stack(batched_anchors)