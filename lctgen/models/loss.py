import torch
import torch.nn as nn

loss_ce = nn.CrossEntropyLoss()
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

def null_loss_func(input, *args):
  return torch.tensor(0.0)

def obtain_norm_feat(input):
  scene_feat = input['input_feature']
  text_feat = input['text_feature']
  scene_feat_norm = scene_feat / scene_feat.norm(dim=-1, keepdim=True)
  text_feat_norm = text_feat / text_feat.norm(dim=-1, keepdim=True)

  return scene_feat_norm, text_feat_norm

def clip_loss_func(input, cfg):
  scene_feat_norm, text_feat_norm = obtain_norm_feat(input)

  logit_scale = input['logit_scale'] * cfg.LOSS.CLIP.LOGIT_SCALE
  logits_per_scene = logit_scale * scene_feat_norm @ text_feat_norm.t()
  logits_per_text = logits_per_scene.t()

  batch_size = scene_feat_norm.shape[0]
  ground_truth = torch.arange(batch_size, dtype=torch.long, device=scene_feat_norm.device)

  ce_from_motion_loss = loss_ce(logits_per_scene, ground_truth)
  ce_from_d_loss = loss_ce(logits_per_text, ground_truth)
  clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

  clip_accuracy = (logits_per_scene.argmax(dim=-1) == ground_truth).float().mean() * 100

  return clip_mixed_loss, clip_accuracy

def cosine_loss_func(input, *args):
  scene_feat_norm, text_feat_norm = obtain_norm_feat(input)

  cos = cosine_sim(scene_feat_norm, text_feat_norm)
  cosine_loss = (1 - cos).mean()
  
  return cosine_loss

def heatmap_loss_func(input, cfg):
  BCE = torch.nn.BCELoss(reduction='none')
  MSE = torch.nn.MSELoss(reduction='none')

  data = input['data']
  
  # remove ego vehicle from the gt distribution
  ego_idx = data['agent_vec_index'][:, 0].long()
  B = ego_idx.shape[0]
  data['gt_distribution'][torch.arange(B), ego_idx] = 0

  heatmap_loss = 0
  
  prob_sources = ['input', 'text']
  result = {}

  for prob_source in prob_sources:
    pred_dists = input[prob_source+'_decode_prob']
    prob_loss = BCE(pred_dists['prob'], data['gt_distribution'])

    line_mask = data['center_mask']
    prob_loss = torch.sum(prob_loss * line_mask) / max(torch.sum(line_mask), 1)

    gt_mask = data['gt_distribution']
    gt_sum = torch.clip(torch.sum(gt_mask, dim=1).unsqueeze(-1), min=1)
    # pos_loss = MSE(pred_dists['pos'],data['gt_long_lat']).mean(-1)
    # pos_loss = (torch.sum(pos_loss * gt_mask, dim=1) / gt_sum).mean()

    # pos_loss = -pred_dists['pos'].log_prob(data['gt_long_lat'])
    # pos_loss = (torch.sum(pos_loss * gt_mask, dim=1) / gt_sum).mean()
    result[prob_source+'_pos_loss'] = 0

    if prob_source == 'input' or cfg.LOSS.AE.TEXT_AE:
      heatmap_loss += prob_loss

    result[prob_source+'_prob_loss'] = prob_loss
  
  result['heatmap'] = heatmap_loss

  return result

def mix_loss_func(input, cfg):
  clip_loss = clip_loss_func(input, cfg) 
  heatmap_losses = heatmap_loss_func(input, cfg)
  heatmap_loss = heatmap_losses['heatmap']
  cos_loss = cosine_loss_func(input)
  mse_loss = mse_loss_func(input)
  
  mix_loss = clip_loss * cfg.LOSS.WEIGHTS[0] 
  mix_loss += heatmap_loss * cfg.LOSS.WEIGHTS[1]
  
  if cfg.LOSS.WEIGHTS[2] > 0:
    mix_loss += cos_loss * cfg.LOSS.WEIGHTS[2]
  
  if len(cfg.LOSS.WEIGHTS) > 3 and cfg.LOSS.WEIGHTS[3] > 0:
    mix_loss += mse_loss * cfg.LOSS.WEIGHTS[3]

  result = {'full_loss': mix_loss, 'clip_loss': clip_loss, 'cos_loss': cos_loss, 'mse_loss': mse_loss}
  result.update(heatmap_losses)

  return result

def alignment_loss_func(input, cfg):
  clip_loss, clip_accuracy = clip_loss_func(input, cfg)
  cos_loss = cosine_loss_func(input)
  mse_loss = mse_loss_func(input)
  
  mix_loss = clip_loss * cfg.LOSS.DETR.ALIGNMENT.WEIGHT.CLIP
  mix_loss += cos_loss * cfg.LOSS.DETR.ALIGNMENT.WEIGHT.COSINE
  mix_loss += mse_loss * cfg.LOSS.DETR.ALIGNMENT.WEIGHT.MSE

  result = {'alignment_loss': mix_loss, 'clip_loss': clip_loss, 'cos_loss': cos_loss, 'mse_loss': mse_loss, 'clip_accuracy': clip_accuracy}

  return result


def mse_loss_func(input):
  MSE = torch.nn.MSELoss()
  scene_feat = input['input_feature']
  text_feat = input['text_feature']

  mse_loss = MSE(scene_feat, text_feat)

  return mse_loss

loss_funcs = {'null': null_loss_func, 'clip': clip_loss_func, 'cosine': cosine_loss_func, 'heatmap': heatmap_loss_func, 'mix': mix_loss_func, 'alignment': alignment_loss_func}