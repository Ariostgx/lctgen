import torch
import numpy as np
from trafficgen.utils.data_process.agent_process import WaymoAgent
from trafficgen.utils.visual_init import draw
from torch.utils.data.dataloader import default_collate

# try:
#   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# except:
#   tokenizer = None

def vis_init(data, agent_num=None):
  center = data["center"]
  rest = data["rest"]
  bound = data["bound"]
  agent = data["agent"]
  agents = [WaymoAgent(agent[i:i+1]) for i in range(agent.shape[0])]

  if agent_num:
    agents = agents[:agent_num]

  return draw(center, agents, other=rest, edge=bound, save_np=True, showup=False)

def get_degree_accel(traj, init_speed=0):
  last_speed = init_speed
  last_deg = 0
  
  accels = []
  degrees = []
  speeds = []

  step = traj.shape[0]
  for i in range(step-1):
    shift = traj[i+1] - traj[i]
    degree = np.rad2deg(np.arctan2(shift[1], shift[0]))
    degrees.append(degree-last_deg)
    last_deg = degree

    speed = np.linalg.norm(shift)
    accels.append(speed-last_speed)
    last_speed = speed
    speeds.append(speed)
  
  return degrees, accels, speeds

def fc_collate_fn(batch):
  result_batch = {}
  
  for key in batch[0].keys():
    if 'other' in key or 'center_info' in key:
      result_batch[key] = [item[key] for item in batch]
    else:
      result_batch[key] = default_collate([item[key] for item in batch])
  
  return result_batch