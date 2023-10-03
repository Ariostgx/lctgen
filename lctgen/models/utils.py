import torch
import copy
import numpy as np

import multiprocessing
from functools import partial
import tqdm
from PIL import Image

from trafficgen.utils.visual_init import get_heatmap, draw, draw_seq, draw_seq_map
from trafficgen.utils.utils import get_agent_coord_from_vec
from trafficgen.utils.data_process.agent_process import WaymoAgent

def visualize_decoder(data, decode_probs):
  # visualize heatmap of the first element in the batch

  center = data['center'][0].cpu().numpy()
  rest = data['rest'][0].cpu().numpy()
  bound = data['bound'][0].cpu().numpy()

  center_mask = data['center_mask'][0].cpu().numpy()

  prob = decode_probs['prob'][0].cpu().numpy()
  pos = torch.clip(decode_probs['pos'].sample(), min=-0.5, max=0.5)[0].cpu().numpy()
  #  debug: remove position prediction from heatmap visualization
  pos = pos * 0

  coords = get_agent_coord_from_vec(center, pos).numpy()

  heatmap = get_heatmap(coords[:, 0][center_mask], coords[:, 1][center_mask], prob[center_mask], 20)

  return draw(center, [], other=rest, edge=bound, save_np=True, showup=False, heat_map=heatmap)

def visualize_input(data, agents = None):
  center = data["center"][0].cpu().numpy()
  rest = data["rest"][0].cpu().numpy()
  bound = data["bound"][0].cpu().numpy()
  agent_mask = data["agent_mask"][0].cpu().numpy()
  
  if agents is None:
    agent = data["agent"][0].cpu().numpy()
    agents = [WaymoAgent(agent[i:i+1]) for i in range(agent.shape[0]) if agent_mask[i]]

  return draw(center, agents, other=rest, edge=bound, save_np=True, showup=False)

def visualize_map(data):
  center = data["center"][0].cpu().numpy()
  rest = data["rest"][0].cpu().numpy()
  bound = data["bound"][0].cpu().numpy()

  return draw_seq_map(center, other=rest, edge=bound, save_np=True)


def visualize_input_seq(data, agents = None, traj=None, sort_agent=True, clip_size=True):
  MIN_LENGTH = 4.0
  MIN_WIDTH = 1.5

  center = data["center"][0].cpu().numpy()
  rest = data["rest"][0].cpu().numpy()
  bound = data["bound"][0].cpu().numpy()
  agent_mask = data["agent_mask"][0].cpu().numpy()
  
  if agents is None:
    agent = data["agent"][0].cpu().numpy()
    agents = [WaymoAgent(agent[i:i+1]) for i in range(agent.shape[0]) if agent_mask[i]]
  
  if traj is None:
    traj = data['gt_pos'][0][:, agent_mask].cpu().numpy()

  if sort_agent:
    agent_dists = [np.linalg.norm(agent.position) for agent in agents]
    agent_idx = np.argsort(agent_dists)
    agents = [agents[i] for i in agent_idx]
    traj = traj[:, agent_idx]
  
  if clip_size:
    for i in range(len(agents)):
      agents[i].length_width = np.clip(agents[i].length_width, [MIN_LENGTH, MIN_WIDTH], [10.0, 5.0])

  return draw_seq(center, agents, traj=traj, other=rest, edge=bound, save_np=True)

def visualize_query_heatmaps(data, output):
  MAX_QUERY_NUM = 12
  ROWS = 3
  COLS = MAX_QUERY_NUM // ROWS

  center = data['center'][0].cpu().numpy()
  center_mask = data['center_mask'][0].cpu().numpy()
  rest = data['rest'][0].cpu().numpy()
  bound = data['bound'][0].cpu().numpy()
  probs = output['probs']

  if len(probs) == 1:
    MAX_QUERY_NUM = 1
    ROWS = 1
    COLS = 1
    probs = output['pred_logits'].sigmoid().cpu().numpy()[:, center_mask]
    smooth = 20
    size = 1000
    figure_size = (10, 10)
  else:
    smooth = 10
    size = 500
    figure_size = (5, 5)

  max_num = min(MAX_QUERY_NUM, len(probs))
  figures = []
  
  for i in range(max_num):
    heatmap = get_heatmap(center[:, 0][center_mask], center[:, 1][center_mask], probs[i], smooth, size)
    figures.append(draw(center, [], other=rest, edge=bound, save_np=True, showup=False, heat_map=heatmap, figsize=figure_size, draw_traf_state=False))

  if max_num < MAX_QUERY_NUM:
    for _ in range(MAX_QUERY_NUM - max_num):
      figures.append(np.zeros_like(figures[0]))
  
  figure_rows = []
  for row in range(ROWS):
    figure_rows.append(np.concatenate(figures[row*COLS:(row+1)*COLS], axis=1))
  figure = np.concatenate(figure_rows, axis=0)

  return figure

def transform_traj_output_to_waymo_agent(output, fps=10):
  STOP_SPEED = 1.0
  init_agent = output['agent']
  traj = output['traj']
  pred_future_heading = 'future_heading' in output
  pred_future_vel = 'future_vel' in output
  
  T = len(traj)

  pred_agents = [init_agent]

  if not pred_future_heading:
    traj_headings = []
    traj_speeds = []
    for i in range(len(traj)-1):
      start = traj[i]
      end = traj[i+1]
      traj_headings.append(np.arctan2(end[:, 1]-start[:, 1], end[:, 0]-start[:, 0]))
      traj_speeds.append(np.linalg.norm(end-start, axis=1)*fps)

    init_speeds = np.array([np.linalg.norm(np.linalg.norm(agent.velocity[0])) for agent in init_agent])
    traj_speeds = [init_speeds] + traj_speeds
    traj_headings.append(traj_headings[-1])

  for t in range(T-1):
    agents_t = copy.deepcopy(init_agent)
    for aidx, agent in enumerate(agents_t):
      agent.position = np.array([traj[t+1, aidx]])

      if not pred_future_heading:
        if traj_speeds[t][aidx] < STOP_SPEED:
          agent.heading = init_agent[aidx].heading
        else:
          agent.heading = np.array([traj_headings[t+1][aidx]])
      else:
        agent.heading = output['future_heading'][t][aidx]
      
      if pred_future_vel:
        agent.velocity = output['future_vel'][t][aidx]
    
    pred_agents.append(agents_t)
  
  return pred_agents

def draw_frame(t, output_scene, pred_agents, data):
  frame = Image.fromarray(draw_seq(output_scene['center'], pred_agents[t], traj=output_scene['traj'], other=data['rest'][0], edge=data['bound'][0],save_np=True))
  return frame

def visualize_output_seq(data, output, fps=10, pool_num=16):
  pred_agents = transform_traj_output_to_waymo_agent(output, fps=fps)
  T = len(pred_agents)

  image_list = []
  with multiprocessing.Pool(pool_num) as pool:
    draw_frame_partial = partial(draw_frame, output_scene=output, pred_agents=pred_agents, data=data)
    for frame in tqdm.tqdm(pool.imap(draw_frame_partial, range(T)), total=T):
      image_list.append(frame)

  return image_list