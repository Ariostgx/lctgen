from enum import Enum

class Action(Enum):
  stop = 0
  turn_left = 1
  left_lane_change = 2
  decelerate = 3
  keep_speed = 4
  accelerate = 5
  right_lane_change = 6
  turn_right = 7
  straight = 8