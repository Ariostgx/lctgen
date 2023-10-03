import copy
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from trafficgen.utils.data_process.agent_process import WaymoAgent
from trafficgen.utils.utils import process_map, rotate, cal_rel_dir
from lctgen.core.registry import registry

from .description import descriptions

@registry.register_dataset(name='WaymoOpenMotion')
class WaymoOpenMotionDataset(Dataset):
    def __init__(self, cfg, mode="", prefix=""):
        
        model_cfg = copy.deepcopy(cfg.MODEL)
        data_cfg = copy.deepcopy(cfg.DATASET)
        
        self.data_list_file = os.path.join(data_cfg.DATA_LIST.ROOT, data_cfg.DATA_LIST[mode.upper()])
        self.data_path = data_cfg.DATA_PATH
        self.RANGE = data_cfg.RANGE
        self.MAX_AGENT_NUM = data_cfg.MAX_AGENT_NUM
        self.THRES = data_cfg.THRES
        self.mode = mode

        with open(self.data_list_file, 'r') as f:
            self.data_list = f.readlines()

        self.data_len = None
        self.data_loaded = {}
        self.text_lib = set()
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.prefix = prefix

        self.use_cache = data_cfg.CACHE
        self.enable_out_range_traj = data_cfg.ENABLE_OUT_RANGE_TRAJ

        if self.use_cache:
            self._load_cache_data()

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):

        # try:
        if self.use_cache:
            data = self._cached_data[index]
            data, _ = self._add_text_attr(data)
        else:
            data = self._get_item_helper(index)
        return data

    def _load_cache_data(self):
        print('Loading cached data...')
        data_name = 'train' if self.mode == 'train' else 'val'

        if self.enable_out_range_traj:
            file_template = 'cached_{}_data_out_range_mask.npy'
        else:
            file_template = 'cached_{}_data.npy'

        cache_data_path = os.path.join(self.data_path, 'cache', file_template.format(data_name))
        self._cached_data = np.load(cache_data_path, allow_pickle=True).item()
        print('Loading cached data finished.')

        assert len(self._cached_data) == len(self.data_list)

    def _add_text_attr(self, data):
        txt_result = self._get_text(data)
        data['text'] = txt_result['text']
        data['token'] = txt_result['token']
        data['text_index'] = txt_result['index']

        return data, txt_result

    def _get_text(self, data):
        txt_cfg = self.data_cfg.TEXT
        description = descriptions[txt_cfg.TYPE](data, txt_cfg)
        result = {}

        text = description.get_category_text(txt_cfg.CLASS)
        token = text
        index = []
        
        result['text'] = text
        result['token'] = token
        result['index'] = index

        return result

    def _get_item_helper(self, index):
        file = self.data_list[index].strip()

        if len(file.split(' ')) > 1:
            file, num = file.split(' ')
            index = int(num)
        else:
            index = -1

        data_file_path = os.path.join(self.data_path, file).strip()
        with open(data_file_path, 'rb') as f:
            datas = pickle.load(f)
        data = self._process(datas, index)
        
        data['file'] = file
        data['index'] = index
        
        data, txt_result = self._add_text_attr(data)
        return data

    def _process_agent(self, agent, sort_agent):

        ego = agent[:, 0]

        # transform every frame into ego coordinate in the first frame
        ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
        ego_heading = ego[[0], [4]]

        agent[..., :2] -= ego_pos
        agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
        agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
        agent[..., 4] -= ego_heading

        agent_mask = agent[..., -1]
        agent_type_mask = agent[..., -2]
        agent_range_mask = (abs(agent[..., 0]) < self.RANGE) * (abs(agent[..., 1]) < self.RANGE)
        
        if self.enable_out_range_traj:
            mask = agent_mask * agent_type_mask
            # use agent range mask only for the first frame
            # allow agent to be out of range in the future frames
            mask[0, :] *= agent_range_mask[0, :]
        else:
            mask = agent_mask * agent_type_mask * agent_range_mask

        return agent, mask.astype(bool)

    def _get_vec_based_rep(self, case_info):

        thres = self.THRES
        max_agent_num = self.MAX_AGENT_NUM
        # _process future agent

        agent = case_info['agent']
        vectors = case_info["center"]

        agent_mask = case_info['agent_mask']

        vec_x = ((vectors[..., 0] + vectors[..., 2]) / 2)
        vec_y = ((vectors[..., 1] + vectors[..., 3]) / 2)

        agent_x = agent[..., 0]
        agent_y = agent[..., 1]

        b, vec_num = vec_y.shape
        _, agent_num = agent_x.shape

        vec_x = np.repeat(vec_x[:, np.newaxis], axis=1, repeats=agent_num)
        vec_y = np.repeat(vec_y[:, np.newaxis], axis=1, repeats=agent_num)

        agent_x = np.repeat(agent_x[:, :, np.newaxis], axis=-1, repeats=vec_num)
        agent_y = np.repeat(agent_y[:, :, np.newaxis], axis=-1, repeats=vec_num)

        dist = np.sqrt((vec_x - agent_x) ** 2 + (vec_y - agent_y) ** 2)

        cent_mask = np.repeat(case_info['center_mask'][:, np.newaxis], axis=1, repeats=agent_num)
        dist[cent_mask == 0] = 10e5
        vec_index = np.argmin(dist, -1)
        min_dist_to_lane = np.min(dist, -1)
        min_dist_mask = min_dist_to_lane < thres

        selected_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], axis=1)

        vx, vy = agent[..., 2], agent[..., 3]
        v_value = np.sqrt(vx ** 2 + vy ** 2)
        low_vel = v_value < 0.1

        dir_v = np.arctan2(vy, vx)
        x1, y1, x2, y2 = selected_vec[..., 0], selected_vec[..., 1], selected_vec[..., 2], selected_vec[..., 3]
        dir = np.arctan2(y2 - y1, x2 - x1)
        agent_dir = agent[..., 4]

        v_relative_dir = cal_rel_dir(dir_v, agent_dir)
        relative_dir = cal_rel_dir(agent_dir, dir)

        v_relative_dir[low_vel] = 0

        v_dir_mask = abs(v_relative_dir) < np.pi / 6
        dir_mask = abs(relative_dir) < np.pi / 4

        agent_x = agent[..., 0]
        agent_y = agent[..., 1]
        vec_x = (x1 + x2) / 2
        vec_y = (y1 + y2) / 2

        cent_to_agent_x = agent_x - vec_x
        cent_to_agent_y = agent_y - vec_y

        coord = rotate(cent_to_agent_x, cent_to_agent_y, np.pi / 2 - dir)

        vec_len = np.clip(np.sqrt(np.square(y2 - y1) + np.square(x1 - x2)), a_min=4.5, a_max=5.5)

        lat_perc = np.clip(coord[..., 0], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
        long_perc = np.clip(coord[..., 1], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
        
        if self.enable_out_range_traj:
            # ignore other masks for future agents (to support out-of-range agent prediction)
            total_mask = agent_mask
            # for the first frame, use all masks to filter out off-road agents
            total_mask[0, :] = (min_dist_mask * agent_mask * v_dir_mask * dir_mask)[0, :]
        else:
            total_mask = agent_mask * min_dist_mask * v_dir_mask * dir_mask

        total_mask[:, 0] = 1
        total_mask = total_mask.astype(bool)

        b_s, agent_num, agent_dim = agent.shape
        agent_ = np.zeros([b_s, max_agent_num, agent_dim])
        agent_mask_ = np.zeros([b_s, max_agent_num]).astype(bool)

        the_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], 1)
        # 0: vec_index
        # 1-2 long and lat percent
        # 3-5 velocity and direction
        # 6-9 lane vector
        # 10-11 lane type and traff state
        info = np.concatenate(
            [vec_index[..., np.newaxis], long_perc[..., np.newaxis], lat_perc[..., np.newaxis],
            v_value[..., np.newaxis], v_relative_dir[..., np.newaxis], relative_dir[..., np.newaxis], the_vec], -1)

        info_ = np.zeros([b_s, max_agent_num, info.shape[-1]])

        start_mask = total_mask[0]
        for i in range(agent.shape[0]):
            agent_i = agent[i][start_mask]
            info_i = info[i][start_mask]

            step_mask = total_mask[i]
            valid_mask = step_mask[start_mask]

            agent_i = agent_i[:max_agent_num]
            info_i = info_i[:max_agent_num]

            valid_num = agent_i.shape[0]
            agent_i = np.pad(agent_i, [[0, max_agent_num - agent_i.shape[0]], [0, 0]])
            info_i = np.pad(info_i, [[0, max_agent_num - info_i.shape[0]], [0, 0]])

            agent_[i] = agent_i
            info_[i] = info_i
            agent_mask_[i, :valid_num] = valid_mask[:valid_num]

        case_info['vec_based_rep'] = info_[..., 1:]
        case_info['agent_vec_index'] = info_[..., 0].astype(int)
        case_info['agent_mask'] = agent_mask_
        case_info["agent"] = agent_

        return case_info

    def _get_traj(self, case_info):
        traj = case_info['gt_pos']
        
        if self.data_cfg.TRAJ_TYPE == 'xy_relative':
            traj = traj - traj[[0], :]
        elif self.data_cfg.TRAJ_TYPE == 'xy':
            traj = traj
        elif self.data_cfg.TRAJ_TYPE == 'xy_theta_relative':
            # rotate traj of each actor to the direction of the vehicle
            traj = traj - traj[[0], :]
            init_heading = case_info['gt_agent_heading'][0]
            traj = rotate(traj[..., 0], traj[..., 1], -init_heading)
        
        return traj

    def _get_future_heading_vel(self, case_info):
        # get the future heading and velocity of each agent
        # use the relative direction and velocity to the initial direction

        future_heading = copy.deepcopy(case_info['gt_agent_heading'])
        future_vel = copy.deepcopy(case_info['agent'][..., 2:4])

        future_heading = cal_rel_dir(future_heading, case_info['gt_agent_heading'][0])
        future_vel = rotate(future_vel[..., 0], future_vel[..., 1], -case_info['gt_agent_heading'][0])

        return future_heading, future_vel

    def _get_gt(self, case_info):
        # 0: vec_index
        # 1-2 long and lat percent
        # 3-5 speed, angle between velocity and car heading, angle between car heading and lane vector
        # 6-9 lane vector
        # 10-11 lane type and traff state
        center_num = case_info['center'].shape[1]
        lane_inp, agent_vec_index, vec_based_rep, bbox, pos = \
            case_info['lane_inp'][:, :center_num], case_info['agent_vec_index'], case_info['vec_based_rep'], case_info['agent'][..., 5:7], case_info['agent'][..., :2]
        b, lane_num, _ = lane_inp.shape
        gt_distribution = np.zeros([b, lane_num])
        gt_vec_based_coord = np.zeros([b, lane_num, 5])
        gt_bbox = np.zeros([b, lane_num, 2])
        for i in range(b):
            mask = case_info['agent_mask'][i]
            index = agent_vec_index[i][mask].astype(int)
            gt_distribution[i][index] = 1
            gt_vec_based_coord[i, index] = vec_based_rep[i, mask, :5]
            gt_bbox[i, index] = bbox[i, mask]
        case_info['gt_vec_index'] = agent_vec_index
        case_info['gt_pos'] = pos
        case_info['gt_bbox'] = gt_bbox
        case_info['gt_distribution'] = gt_distribution
        case_info['gt_long_lat'] = gt_vec_based_coord[..., :2]
        case_info['gt_speed'] = gt_vec_based_coord[..., 2]
        case_info['gt_vel_heading'] = gt_vec_based_coord[..., 3]
        case_info['gt_heading'] = gt_vec_based_coord[..., 4]
        case_info['gt_agent_heading'] = case_info['agent'][..., 4]
        case_info['traj'] = self._get_traj(case_info)
        case_info['future_heading'], case_info['future_vel'] = self._get_future_heading_vel(case_info)

        return case_info

    def _process(self, data, index):
        case_info = {}
        other = {}

        agent = copy.deepcopy(data['all_agent'])
        other['traf'] = copy.deepcopy(data['traffic_light'])
        max_time_step = self.data_cfg.MAX_TIME_STEP
        gap = self.data_cfg.TIME_SAMPLE_GAP
        
        if index == -1:
            data['all_agent'] = data['all_agent'][0:max_time_step:gap]
            data['traffic_light'] = data['traffic_light'][0:max_time_step:gap]
        else:
            index = min(index, len(data['all_agent'])-1)
            data['all_agent'] = data['all_agent'][index:index+self.data_cfg.MAX_TIME_STEP:gap]
            data['traffic_light'] = data['traffic_light'][index:index+self.data_cfg.MAX_TIME_STEP:gap]

        data['lane'], other['unsampled_lane'] = self._transform_coordinate_map(data)
        other['lane'] = data['lane']

        # transform agent coordinate
        ego = agent[:, 0]
        ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
        ego_heading = ego[[0], [4]]
        agent[..., :2] -= ego_pos
        agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
        agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
        agent[..., 4] -= ego_heading
        agent_mask = agent[..., -1]
        agent_type_mask = agent[..., -2]
        agent_range_mask = (abs(agent[..., 0]) < self.RANGE) * (abs(agent[..., 1]) < self.RANGE)
        mask = agent_mask * agent_type_mask * agent_range_mask

        agent = WaymoAgent(agent)
        other['gt_agent'] = agent.get_inp(act=True)
        other['gt_agent_mask'] = mask
        other['center_info'] = data['center_info']

        # _process agent and lane data
        case_info["agent"], case_info["agent_mask"] = self._process_agent(data['all_agent'], False)
        case_info['center'], case_info['center_mask'], case_info['center_id'], case_info['bound'], case_info['bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
            data['lane'], data['traffic_light'], lane_range=self.RANGE, offest=0)

        # get vector-based representatiomn
        case_info = self._get_vec_based_rep(case_info)

        agent = WaymoAgent(case_info['agent'], case_info['vec_based_rep'])

        case_info['agent_feat'] = agent.get_inp()
        
        case_info = self._process_map_inp(case_info)
        case_info = self._get_gt(case_info)

        case_num = case_info['agent'].shape[0]
        case_list = []
        for i in range(case_num):
            dic = {}
            for k, v in case_info.items():
                dic[k] = v[i]
            case_list.append(dic)
        
        future_attrs = ['gt_pos', 'gt_vec_index', 'traj', 'future_heading', 'future_vel']
        for attr in future_attrs:
            case_list[0][attr] = case_info[attr]
        case_list[0]['all_agent_mask'] = case_info['agent_mask']

        # include other info for use in MetaDrive
        if self.data_cfg.INCLUDE_LANE_INFO:
            case_list[0]['lane'] = other['lane']
            case_list[0]['traf'] = data['traffic_light']
            case_list[0]['other'] = other
        
        return case_list[0]

    def _process_map_inp(self, case_info):
        center = copy.deepcopy(case_info['center'])
        center[..., :4] /= self.RANGE
        edge = copy.deepcopy(case_info['bound'])
        edge[..., :4] /= self.RANGE
        cross = copy.deepcopy(case_info['cross'])
        cross[..., :4] /= self.RANGE
        rest = copy.deepcopy(case_info['rest'])
        rest[..., :4] /= self.RANGE

        case_info['lane_inp'] = np.concatenate([center, edge, cross, rest], axis=1)
        case_info['lane_mask'] = np.concatenate(
            [case_info['center_mask'], case_info['bound_mask'], case_info['cross_mask'], case_info['rest_mask']],
            axis=1)
        return case_info

    def _transform_coordinate_map(self, data):
        """
        Every frame is different
        """
        timestep = data['all_agent'].shape[0]

        ego = data['all_agent'][:, 0]
        pos = ego[:, [0, 1]][:, np.newaxis]

        lane = data['lane'][np.newaxis]
        lane = np.repeat(lane, timestep, axis=0)
        lane[..., :2] -= pos

        x = lane[..., 0]
        y = lane[..., 1]
        ego_heading = ego[:, [4]]
        lane[..., :2] = rotate(x, y, -ego_heading)

        unsampled_lane = data['unsampled_lane'][np.newaxis]
        unsampled_lane = np.repeat(unsampled_lane, timestep, axis=0)
        unsampled_lane[..., :2] -= pos

        x = unsampled_lane[..., 0]
        y = unsampled_lane[..., 1]
        ego_heading = ego[:, [4]]
        unsampled_lane[..., :2] = rotate(x, y, -ego_heading)
        return lane, unsampled_lane[0]
