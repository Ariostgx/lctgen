import torch
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from lctgen.core.registry import registry
from lctgen.datasets.utils import fc_collate_fn
from lctgen.models.utils import  visualize_input_seq
from trafficgen.utils.typedef import *

def map_dict_to_vec(map_data):
  DIST_INTERVAL = 5

  map_vector = np.zeros(6)
  map_vector[0] = map_data['same_direction_lane_cnt']
  map_vector[1] = map_data['opposite_direction_lane_cnt']
  map_vector[2] = map_data['vertical_up_lane_cnt']
  map_vector[3] = map_data['vertical_down_lane_cnt']
  map_vector[4] = map_data['dist_to_intersection'] // DIST_INTERVAL
  map_vector[5] = 1 + len(map_data['same_right_dir_lanes'])

  return map_vector

def map_vec_distance(query, map_vec):
  weight = np.array([1, 1, 1, 1, 1, 1])

  if query[2] + query[3] == 0:
    weight[4] = 0

  result = np.abs(np.array(query)-map_vec)
  result = result * weight
  return np.sum(result, axis=1)

def load_map_data(map_id, data_root):
  map_path = os.path.join(data_root, map_id + '.json')
  with open(map_path, 'r') as f:
    map_data = json.load(f)

  return map_data

def map_retrival(target_vec, map_vecs):
  map_dist = map_vec_distance(target_vec, map_vecs)
  map_dist_idx = np.argsort(map_dist)
  return map_dist_idx

def load_all_map_vectors(map_file):
  map_data = np.load(map_file, allow_pickle=True).item()
  map_vectors = map_data['vectors']
  map_ids = map_data['ids']

  data_list = []
  for map_id in map_ids:
    data_list.append('_'.join(map_id.split('_')[:2]) + '.pkl' + ' ' + map_id.split('_')[-1])

  return map_vectors, data_list

def get_map_data_batch(map_id, cfg):
  dataset_type = cfg.DATASET.TYPE
  cfg['DATASET']['CACHE'] = False
  dataset = registry.get_dataset(dataset_type)(cfg, 'train')
  dataset.data_list = [map_id]
  
  collate_fn = fc_collate_fn
  loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory = False,
                  drop_last=False, num_workers=1, collate_fn=collate_fn)
  
  for idx, batch in enumerate(loader):
    if idx == 1:
      break

  return batch

def load_inference_model(cfg):
  model_cls = registry.get_model(cfg.MODEL.TYPE)
  lightning_model = model_cls.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)

  return lightning_model

def vis_decode(batch, ae_output):
  img = visualize_input_seq(batch, agents=ae_output[0]['agent'], traj=ae_output[0]['traj'])
  return Image.fromarray(img)

def output_formating_cot(result):
  lines = result.split('\n')
  agent_vectors = []
  vector_idx = [idx for idx, line in enumerate(lines) if 'Actor Vector' in line]
  if len(vector_idx) == 0:
    return [], []
  vector_idx = vector_idx[0]

  for line in lines[vector_idx+1:]:
    if 'V' in line or 'Map' in line:
      if 'Vector' in line:
        continue

      data_line = line.split(':')[-1].strip()
      data_vec = eval(data_line)
      
      if 'Map' in line:
        map_vector = data_vec
      else:
        agent_vectors.append(data_vec)

  print('Agent vectors:', agent_vectors)
  print('Map vector:', map_vector)
  
  return agent_vectors, map_vector

def transform_dist_base(agent_vector, cfg):
  text_distance_base = 5
  distance_base = cfg.DISTANCE_BASE
  ratio = text_distance_base / distance_base

  if ratio == 1:
    return agent_vector

  for idx in range(len(agent_vector)):
    agent_vector[idx][1] = int(agent_vector[idx][1] * ratio)
  
  print('Transformed agent vector:', agent_vector)
  
  return agent_vector
