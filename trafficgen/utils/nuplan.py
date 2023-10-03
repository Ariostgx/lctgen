import numpy as np

from trafficgen.scripts.trans20 import down_sampling
from trafficgen.utils.utils import process_map, rotate, cal_rel_dir, process_lane


def extract_nuplan_map(np_data):
  SAMPLE_NUM = 10

  polylines = []
  for key, value in np_data['map_features'].items():
    polyline = value['polyline']
    full_polyline = np.zeros((len(polyline), 4))
    full_polyline[:, :2] = polyline
    
    type_name = value['type']
    if type_name == 'LANE_SURFACE_STREET':
      type_id = 1.0
    elif type_name == 'ROAD_LINE_BROKEN_SINGLE_WHITE':
      type_id = 8.0
    else:
      type_id = 15.0
    
    full_polyline[:, -2] = type_id
    full_polyline[:, -1] = len(polylines)

    full_polyline = down_sampling(full_polyline, type_id, SAMPLE_NUM)
    full_polyline = np.stack(full_polyline)

    polylines.append(full_polyline)

  polylines = np.concatenate(polylines, axis=0)

  return polylines

def convert_nuplan_data(np_data):
  map_input = extract_nuplan_map(np_data)