import os
import pickle

import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm


def csv_to_pickle(dir_path, target_path):
    df = pd.read_csv(os.path.join(dir_path, 'run.csv'))

    obs, next_obs, state, next_state, action = [], [], [], [], []
    for i in tqdm(range(len(df))):
        _obs = np.asarray(
            Image.open(os.path.join(dir_path, df['img_before'][i].split('/')[1])).crop((165, 165, 465, 415)).convert(
                'L'))
        obs.append(np.expand_dims(_obs, 0))
        next_obs.append(np.expand_dims(
            np.asarray(
                Image.open(os.path.join(dir_path, df['img_after'][i].split('/')[1])).crop((165, 165, 465, 415)).convert(
                    'L')), 0))
        state.append(df[['old_x', 'old_y', 'old_z']].values[i])
        next_state.append(df[['new_x', 'new_y', 'new_z']].values[i])
        action.append(df[['new_x', 'new_y', 'new_z']].values[i] - df[['old_x', 'old_y', 'old_z']].values[i])

    pickle.dump({'X': np.array(obs).astype('float32'),
                 'ast': np.array(state).astype('float32'),
                 'A': np.array(action).astype('float32'),
                 'est': np.zeros_like(state).astype('float32')},
                open(target_path, 'wb'))


if __name__ == '__main__':
    csv_to_pickle("robot_data", os.path.join('robot_data.p'))
