import os
import tqdm
import shutil
import sys
import subprocess
import tqdm

# PATH_A = 'YOUR_PATH_TO_WAYMO_OPEN_TRAINING_20S_DATA'
# PATH_B = 'YOUR_PATH_TO_PROCESSED_DATA'

if __name__ == '__main__':
  PATH_A = sys.argv[1]
  PATH_B = sys.argv[2]

  # make 10 foloders under PATH
  for i in range(10):
      os.makedirs('{}/{}'.format(PATH_A, i))

  # for each folder, move 1/10 of the data to it (data name: training_20s.tfrecord-xxxxx-of-01000)
  for i in tqdm.tqdm(range(10)):
    for j in tqdm.tqdm(range(100)):
      shutil.move('{}/training_20s.tfrecord-{:05d}-of-01000'.format(PATH_A, j+100*i), '{}/{}/'.format(PATH_A, i))


  source_root = PATH_A + '/{}'
  target_dir = PATH_B

  # for each folder, run trans20.py to process the data and save it to PATH_B
  for x in range(10):
    source_dir = source_root.format(x)
    command = "nohup python trafficgen/utils/trans20.py {} {} {} > {}.log 2>&1 &".format(source_dir, target_dir, x, x)
    print(command)
    subprocess.call(command, shell=True)