# -*- coding: utf-8 -*-
"""epoch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xv4FtC4FPcj8G3Jef708MjjTKGJ-uKLE
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time
import pickle
import random
from envs import OfflineEnv
from recommender import DRRAgent

import os
import torch
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')
STATE_SIZE = 4
MAX_EPISODE_NUM = 5

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

    print('Data loading...')

    doc_info = pd.read_csv('./data1/doc_info.csv')

    ipc_to_description = {ipc:doc_info[doc_info['transformed_ipc']==ipc]['ipc_description'].tolist()[0] for ipc in doc_info['transformed_ipc'].unique()}
    #ratings_df = ratings_df.applymap(int)
    with open('./data1/users_dict.pkl', 'rb') as fr:
        users_dict = pickle.load(fr)
    with open('./data1/users_history_lens.pkl', 'rb') as fr:
        users_history_lens = pickle.load(fr)



    pickle_file_path = './data1/ipc_to_embedding.pkl'

# 피클 파일에서 모델 로드
    with open(pickle_file_path, 'rb') as file:
    # 피클 파일에서 모델 로드
        ipc_to_embedding = pickle.load(file)

# 모델을 CPU로 이동
    #ipc_to_embedding.to(torch.device('cpu'))


    print("Data loading complete!")
    print("Data preprocessing...")

    users_num = max(users_dict.keys()) + 1
    items_num = doc_info['transformed_ipc'].nunique()

    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num


    shuffled_keys = list(users_dict.keys())
    random.shuffle(shuffled_keys)

# 랜덤하게 선택된 키 값을 사용하여 train_users_dict 생성
    train_users_dict = {k: users_dict[k] for k in shuffled_keys[:train_users_num]}

    #train_users_dict = {k:users_dict.get(k) for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:shuffled_keys[:train_users_num]]

    print('DONE!')
    time.sleep(2)

    env = OfflineEnv(train_users_dict, train_users_history_lens, ipc_to_description, STATE_SIZE)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE, ipc_to_description, ipc_to_embedding, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(train_users_num, load_model=True)