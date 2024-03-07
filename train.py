#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time
import pickle
import sys
from tqdm import tqdm
from envs import OfflineEnv
from recommender import DRRAgent

import os
import torch
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')
STATE_SIZE = 4
MAX_EPISODE_NUM = 5
NUM_EPOCHS = 1 


# os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

    print('Data loading...')

    doc_info = pd.read_csv('./data1/doc_info.csv')
    
    paper_to_description = {title:doc_info[doc_info['paper']==title]['Keyword'].tolist()[0] for title in doc_info['paper'].unique()}
    #ratings_df = ratings_df.applymap(int)
    with open('./data1/users_dict2.pkl', 'rb') as fr:
        users_dict = pickle.load(fr)
    with open('./data1/users_history_lens.pkl', 'rb') as fr:
        users_history_lens = pickle.load(fr)
    
    
    
    pickle_file_path = './data1/paper_to_embedding.pkl'

# 피클 파일에서 모델 로드
    with open(pickle_file_path, 'rb') as file:
    # 피클 파일에서 모델 로드
        paper_to_embedding = pickle.load(file)

# 모델을 CPU로 이동
    #ipc_to_embedding.to(torch.device('cpu'))

    users_num = max(users_dict.keys()) + 1
    items_num = doc_info['paper'].nunique()
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k:users_dict.get(k) for k in range(1, train_users_num+1)}
    sliced_users_dict = {key: train_users_dict[key] for key in list(train_users_dict.keys())[1868:]}

    train_users_history_lens = users_history_lens[:train_users_num]
    #sliced_users_history_lens = {key: train_users_history_lens[key] for key in train_users_history_lens.keys()[1868:]}
    sliced_users_history_lens=train_users_history_lens[1868:]
    train_loss = []
    train_pre=[]
    final_list=[]
    #test_loss=100

    print("Data loading complete!")
    print("Data preprocessing...")  
    print('DONE!')
    
    for epoch in range(NUM_EPOCHS):
      print(f"Training Epoch {epoch + 1}/{NUM_EPOCHS}")                                                
    # Training setting
      if epoch == 0:
        time.sleep(2)
        env = OfflineEnv(train_users_dict, train_users_history_lens, paper_to_description, STATE_SIZE, paper_to_embedding)
        recommender = DRRAgent(env, users_num, items_num, STATE_SIZE, paper_to_description, paper_to_embedding, use_wandb=False)
        recommender.actor.build_networks()
        recommender.critic.build_networks()
        recommender.train(train_users_num, load_model=False)
        #train_pre.append(train_precision)
        #train_loss.append(train_loss_data)
        #final_list.append((train_pre,train_loss))
    # 필요한 경우 모델 가중치를 불러와서 학습을 계속 진행합니다
      elif (epoch>0) and (epoch < NUM_EPOCHS - 1):
          #test_loss=loss_list_value
          recommender.train(train_users_num, load_model=True)
          #loss_list=[]
          #loss_list.append(loss)
      #loss_list_value=np.mean(loss_list)          
      #print(f"Epoch {epoch + 1} completed. Model trained and weights saved successfully.")

    #sys.exit(final_list)






