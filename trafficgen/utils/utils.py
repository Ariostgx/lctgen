import copy
import datetime
import pickle
import time

import numpy as np
import torch
from shapely.geometry import Polygon
from torch import Tensor

from trafficgen.utils.data_process.agent_process import WaymoAgent
from trafficgen.utils.typedef import AgentType
from trafficgen.utils.typedef import RoadLineType, RoadEdgeType


def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        ret = fn(*args, **kwargs)
        return ret, time.clock() - start

    return _wrapper

def MDdata_to_initdata(MDdata):
    ret = {}
    tracks = MDdata['tracks']

    ret['context_num']=1
    all_agent= np.zeros([128,7])
    agent_mask = np.zeros(128)

    sdc = tracks[MDdata['sdc_index']]['state']
    all_agent[0,:2] = sdc[0,:2]
    all_agent[0,2:4] =sdc[0,7:9]
    all_agent[0,4] = sdc[0,6]
    all_agent[0,5:7] = sdc[0,3:5]

    cnt=1
    for id, track in tracks.items():
        if id == MDdata['sdc_index']:continue
        if not track['type'] == AgentType.VEHICLE: continue
        if track['state'][0,-1]==0:continue
        state = track['state']
        all_agent[cnt, :2] = state[0, :2]
        all_agent[cnt, 2:4] = state[0, 7:9]
        all_agent[cnt, 4] = state[0, 6]
        all_agent[cnt, 5:7] = state[0, 3:5]
        cnt+=1

    all_agent = all_agent[:32]
    agent_num = min(32,cnt)
    agent_mask[:agent_num]=1
    agent_mask=agent_mask.astype(bool)

    lanes = []
    for k, lane in input['map'].items():
        a_lane = np.zeros([20, 4])
        tp = 0
        try:
            lane_type = lane['type']
        except:
            lane_type = lane['sign']
            poly_line = lane['polygon']
            if lane_type == 'cross_walk':
                tp = 18
            elif lane_type == 'speed_bump':
                tp = 19

        if lane_type == 'center_lane':
            poly_line = lane['polyline']
            tp = 1

        elif lane_type == RoadEdgeType.BOUNDARY or lane_type == RoadEdgeType.MEDIAN:
            tp = 15 if lane_type == RoadEdgeType.BOUNDARY else 16
            poly_line = lane['polyline']
        elif 'polyline' in lane:
            tp = 7
            poly_line = lane['polyline']
        if tp == 0:
            continue

        a_lane[:, 2] = tp
        a_lane[:, :2] = poly_line

        lanes.append(a_lane)
    lanes = np.stack(lanes)


    return

def get_polygon(center, yaw, L, W):
    l, w = L / 2, W / 2
    yaw += torch.pi / 2
    theta = torch.atan(w / l)
    s1 = torch.sqrt(l ** 2 + w ** 2)
    x1 = abs(torch.cos(theta + yaw) * s1)
    y1 = abs(torch.sin(theta + yaw) * s1)
    x2 = abs(torch.cos(theta - yaw) * s1)
    y2 = abs(torch.sin(theta - yaw) * s1)

    p1 = [center[0] + x1, center[1] + y1]
    p2 = [center[0] + x2, center[1] - y2]
    p3 = [center[0] - x1, center[1] - y1]
    p4 = [center[0] - x2, center[1] + y2]
    return Polygon([p1, p3, p2, p4])

def get_agent_coord_from_vec(vec, long_lat):
    vec = torch.tensor(vec)
    x1, y1, x2, y2 = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

    vec_len = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    vec_dir = torch.atan2(y2 - y1, x2 - x1)

    long_pos = vec_len * long_lat[..., 0]
    lat_pos = vec_len * long_lat[..., 1]

    coord = rotate(lat_pos, long_pos, -np.pi / 2 + vec_dir)

    coord[:, 0] += x_center
    coord[:, 1] += y_center

    return coord


def get_agent_pos_from_vec(vec, long_lat, speed, vel_heading, heading, bbox, use_rel_heading=True):
    x1, y1, x2, y2 = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

    vec_len = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    vec_dir = torch.atan2(y2 - y1, x2 - x1)

    long_pos = vec_len * long_lat[..., 0]
    lat_pos = vec_len * long_lat[..., 1]

    coord = rotate(lat_pos, long_pos, -np.pi / 2 + vec_dir)

    coord[:, 0] += x_center
    coord[:, 1] += y_center

    if use_rel_heading:
        agent_dir = vec_dir + heading
    else:
        agent_dir = heading
        
    v_dir = vel_heading + agent_dir

    vel = torch.stack([torch.cos(v_dir) * speed, torch.sin(v_dir) * speed], axis=-1)
    agent_num, _ = vel.shape

    type = Tensor([[1]]).repeat(agent_num, 1).to(coord.device)
    agent = torch.cat([coord, vel, agent_dir.unsqueeze(1), bbox, type], dim=-1).detach().cpu().numpy()

    vec_based_rep = torch.cat([long_lat, speed.unsqueeze(-1), vel_heading.unsqueeze(-1), heading.unsqueeze(-1), vec],
                              dim=-1).detach().cpu().numpy()

    agent = WaymoAgent(agent, vec_based_rep)

    return agent


def process_lane(lane, max_vec, lane_range, offset=-40):
    # dist = lane[..., 0]**2+lane[..., 1]**2
    # idx = np.argsort(dist)
    # lane = lane[idx]

    vec_dim = 6

    lane_point_mask = (abs(lane[..., 0] + offset) < lane_range) * (abs(lane[..., 1]) < lane_range)

    lane_id = np.unique(lane[..., -2]).astype(int)

    vec_list = []
    vec_mask_list = []
    vec_id_list = []
    b_s, _, lane_dim = lane.shape

    for id in lane_id:
        id_set = lane[..., -2] == id
        points = lane[id_set].reshape(b_s, -1, lane_dim)
        masks = lane_point_mask[id_set].reshape(b_s, -1)

        vec_ids = np.ones([b_s, points.shape[1] - 1, 1]) * id
        vector = np.zeros([b_s, points.shape[1] - 1, vec_dim])
        vector[..., 0:2] = points[:, :-1, :2]
        vector[..., 2:4] = points[:, 1:, :2]
        # id
        # vector[..., 4] = points[:,1:, 3]
        # type
        vector[..., 4] = points[:, 1:, 2]
        # traffic light
        vector[..., 5] = points[:, 1:, 4]
        vec_mask = masks[:, :-1] * masks[:, 1:]
        vector[vec_mask == 0] = 0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)
        vec_id_list.append(vec_ids)

    vector = np.concatenate(vec_list, axis=1) if vec_list else np.zeros([b_s, 0, vec_dim])
    vector_mask = np.concatenate(vec_mask_list, axis=1) if vec_mask_list else np.zeros([b_s, 0], dtype=bool)
    vec_id = np.concatenate(vec_id_list, axis=1) if vec_id_list else np.zeros([b_s, 0, 1])

    all_vec = np.zeros([b_s, max_vec, vec_dim])
    all_mask = np.zeros([b_s, max_vec])
    all_id = np.zeros([b_s, max_vec, 1])

    for t in range(b_s):
        mask_t = vector_mask[t]
        vector_t = vector[t][mask_t]
        vec_id_t = vec_id[t][mask_t]

        dist = vector_t[..., 0] ** 2 + vector_t[..., 1] ** 2
        idx = np.argsort(dist)
        vector_t = vector_t[idx]
        mask_t = np.ones(vector_t.shape[0])
        vec_id_t = vec_id_t[idx]

        vector_t = vector_t[:max_vec]
        mask_t = mask_t[:max_vec]
        vec_id_t = vec_id_t[:max_vec]

        vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
        mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
        vec_id_t = np.pad(vec_id_t, ([0, max_vec - vec_id_t.shape[0]], [0, 0]))

        all_vec[t] = vector_t
        all_mask[t] = mask_t
        all_id[t] = vec_id_t

    return all_vec, all_mask.astype(bool), all_id.astype(int)


def process_map(lane, traf=None, center_num=384, edge_num=128, lane_range=60, offest=-40, rest_num=192):
    lane_with_traf = np.zeros([*lane.shape[:-1], 5])
    lane_with_traf[..., :4] = lane

    lane_id = lane[..., -1]
    b_s = lane_id.shape[0]

    # print(traf)
    if traf is not None:
        for i in range(b_s):
            traf_t = traf[i]
            lane_id_t = lane_id[i]
            # print(traf_t)
            for a_traf in traf_t:
                # print(a_traf)
                control_lane_id = a_traf[0]
                state = a_traf[-2]
                lane_idx = np.where(lane_id_t == control_lane_id)
                lane_with_traf[i, lane_idx, -1] = state
        lane = lane_with_traf

    # lane = np.delete(lane_with_traf,-2,axis=-1)
    lane_type = lane[0, :, 2]
    center_1 = lane_type == 1
    center_2 = lane_type == 2
    center_3 = lane_type == 3
    center_ind = center_1 + center_2 + center_3

    boundary_1 = lane_type == 15
    boundary_2 = lane_type == 16
    bound_ind = boundary_1 + boundary_2

    cross_walk = lane_type == 18
    speed_bump = lane_type == 19
    cross_ind = cross_walk + speed_bump

    rest = ~(center_ind + bound_ind + cross_walk + speed_bump + cross_ind)

    cent, cent_mask, cent_id = process_lane(lane[:, center_ind], center_num, lane_range, offest)
    bound, bound_mask, _ = process_lane(lane[:, bound_ind], edge_num, lane_range, offest)
    cross, cross_mask, _ = process_lane(lane[:, cross_ind], 32, lane_range, offest)
    rest, rest_mask, _ = process_lane(lane[:, rest], rest_num, lane_range, offest)

    return cent, cent_mask, cent_id, bound, bound_mask, cross, cross_mask, rest, rest_mask


def get_time_str():
    return datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")


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


def cal_rel_dir(dir1, dir2):
    dist = dir1 - dir2

    while not np.all(dist >= 0):
        dist[dist < 0] += np.pi * 2
    while not np.all(dist < np.pi * 2):
        dist[dist >= np.pi * 2] -= np.pi * 2

    dist[dist > np.pi] -= np.pi * 2
    return dist


def rotate(x, y, angle):
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


def from_list_to_batch(inp_list):
    keys = inp_list[0].keys()

    batch = {}
    for key in keys:
        one_item = [item[key] for item in inp_list]
        batch[key] = Tensor(np.stack(one_item))

    return batch


def transform_to_agent(agent_i, agent, lane):
    all_ = copy.deepcopy(agent)

    center = copy.deepcopy(agent_i[:2])
    center_yaw = copy.deepcopy(agent_i[4])

    all_[..., :2] -= center
    coord = rotate(all_[..., 0], all_[..., 1], -center_yaw)
    vel = rotate(all_[..., 2], all_[..., 3], -center_yaw)

    all_[..., :2] = coord
    all_[..., 2:4] = vel
    all_[..., 4] = all_[..., 4] - center_yaw
    # then recover lane's position
    lane = copy.deepcopy(lane)
    lane[..., :2] -= center
    output_coords = rotate(lane[..., 0], lane[..., 1], -center_yaw)
    if isinstance(lane, Tensor):
        output_coords = Tensor(output_coords)
    lane[..., :2] = output_coords

    return all_, lane


def get_type_class(line_type):
    if line_type in range(1, 4):
        return 'center_lane'
    elif line_type == 6:
        return RoadLineType.BROKEN_SINGLE_WHITE
    elif line_type == 7:
        return RoadLineType.SOLID_SINGLE_WHITE
    elif line_type == 8:
        return RoadLineType.SOLID_DOUBLE_WHITE
    elif line_type == 9:
        return RoadLineType.BROKEN_SINGLE_YELLOW
    elif line_type == 10:
        return RoadLineType.BROKEN_DOUBLE_YELLOW
    elif line_type == 11:
        return RoadLineType.SOLID_SINGLE_YELLOW
    elif line_type == 12:
        return RoadLineType.SOLID_DOUBLE_YELLOW
    elif line_type == 13:
        return RoadLineType.PASSING_DOUBLE_YELLOW
    elif line_type == 15:
        return RoadEdgeType.BOUNDARY
    elif line_type == 16:
        return RoadEdgeType.MEDIAN
    else:
        return 'other'

def transform_to_metadrive_data(pred_i, other):
    output_temp = {}
    output_temp['id'] = 'fake'
    output_temp['ts'] = [x / 10 for x in range(190)]
    output_temp['dynamic_map_states'] = [{}]
    output_temp['sdc_index'] = 0
    cnt = 0

    center_info = other['center_info']
    output = copy.deepcopy(output_temp)
    output['tracks'] = {}
    output['map'] = {}
    # extract agents
    agent = pred_i

    for i in range(agent.shape[1]):
        track = {}
        agent_i = agent[:, i]
        track['type'] = AgentType.VEHICLE
        state = np.zeros([agent_i.shape[0], 10])
        state[:, :2] = agent_i[:, :2]
        state[:, 3] = 5.286
        state[:, 4] = 2.332
        state[:, 7:9] = agent_i[:, 2:4]
        state[:, -1] = 1
        state[:, 6] = agent_i[:, 4]  # + np.pi / 2
        track['state'] = state
        output['tracks'][i] = track

    # extract maps
    lane = other['unsampled_lane']
    lane_id = np.unique(lane[..., -1]).astype(int)
    for id in lane_id:

        a_lane = {}
        id_set = lane[..., -1] == id
        points = lane[id_set]
        polyline = np.zeros([points.shape[0], 3])
        line_type = points[0, -2]
        polyline[:, :2] = points[:, :2]
        a_lane['type'] = get_type_class(line_type)
        a_lane['polyline'] = polyline
        if id in center_info.keys():
            a_lane.update(center_info[id])
        output['map'][id] = a_lane

    return output

def save_as_metadrive_data(pred_i, other, save_path):
    output = transform_to_metadrive_data(pred_i, other)
    
    with open(save_path, 'wb') as f:
        pickle.dump(output, f)
