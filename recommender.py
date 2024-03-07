
from replay_buffer import PriorityExperienceReplay
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_math_ops import Exp
#import tensorflow_probability as tfp
from actor import Actor
from critic import Critic
from replay_memory import ReplayMemory
from state_representation import DRRAveStateRepresentation
from datetime import datetime
from torch._C import Use
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import wandb
import random
import torch
#from transformers import BertModel, BertTokenizer
import pandas as pd



class DRRAgent:
    
    def __init__(self, env, users_num, items_num, state_size, ipc_to_description, ipc_to_embedding, is_test=False, use_wandb=False):
        
        self.env = env
        self.ipc_to_description = ipc_to_description
        self.users_num = users_num
        self.items_num = items_num
        self.state_size = state_size
        self.embedding_dim = 768
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.001
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.9
        self.tau = 0.001
        self.ipc_to_embedding=ipc_to_embedding
        self.replay_memory_size = 1000000
        self.batch_size = 32
        self.ite_id=ipc_to_embedding.keys()
        #self.top_k=20
        
        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)

        self.save_model_weight_dir = f"./save_model/model_weight_dict2/"
        if not os.path.exists(self.save_model_weight_dir):
            os.makedirs(os.path.join(self.save_model_weight_dir, 'imagess'))

        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 768,)),np.zeros((1,state_size, 768))])

        # PER
        self.buffer = PriorityExperienceReplay(self.replay_memory_size, self.embedding_dim)
        self.epsilon_for_priority = 1e-6

        # ε-탐욕 탐색 하이퍼파라미터 ε-greedy exploration hyperparameter
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1)/500000
        self.std = 1.5

        self.is_test = is_test

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="drr", 
            entity="diominor",
            config={'users_num':users_num,
            'items_num' : items_num,
            'state_size' : state_size,
            'embedding_dim' : self.embedding_dim,
            'actor_hidden_dim' : self.actor_hidden_dim,
            'actor_learning_rate' : self.actor_learning_rate,
            'critic_hidden_dim' : self.critic_hidden_dim,
            'critic_learning_rate' : self.critic_learning_rate,
            'discount_factor' : self.discount_factor,
            'tau' : self.tau,
            'replay_memory_size' : self.replay_memory_size,
            'batch_size' : self.batch_size,
            'std_for_exploration': self.std})

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        for i in range(q_values.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i])*(self.discount_factor * q_values[i])
        return y_t

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        
        all_user_id, all_items_ids, all_done = self.env.reset_all()
        if items_ids == None:
            ### recommend set 생성
            
            
            ### 기존의 추천셋(state)에 있는 것들은 제외하고 진행
            items_ids=[item for item in all_items_ids if item not in recommended_items]
            items_ebs = self.bert(items_ids)
        
            items_ebs = tf.convert_to_tensor(items_ebs)
            items_ebs = tf.cast(items_ebs, tf.float32)
        else:
            #print(items_ids)
            items_ids=[item for item in all_items_ids if item not in recommended_items]
            plus_ids=random.sample([item for item in list(self.ipc_to_embedding.keys()) if item not in recommended_items],10)
            use_ids=items_ids+plus_ids
            #items_ids=list(self.ipc_to_embedding.keys())
            #print(items_ids)
            #items_ids=list(items_ids)
            item_list=[]
            for id in use_ids:
                item_list.append(self.ipc_to_embedding[id])
            items_tensor = torch.cat(item_list, dim=0)
            items_tensor_cpu = items_tensor.cpu()

            #이제 items_tensor_cpu를 NumPy 배열로 변환할 수 있습니다.
            items_numpy = items_tensor_cpu.numpy()
            items_ebs=tf.convert_to_tensor(items_numpy)
            items_ebs=tf.cast(items_ebs,tf.float32)
        #print(len(items_ebs))
        
        action = tf.transpose(action, perm=(1,0))
        action = tf.cast(action, tf.float32)
        
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1,0)))[0][-top_k:]
            #item_indice = np.argsort(torch.transpose(torch.matmul(items_ebs, torch.tensor(action)), perm=(1,0)))[0][-top_k:]
            #print(items_ids)
            #print(item_indice)
            result=[use_ids[i] for i in item_indice]
            #result_score=np.mean(np.sort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1,0)))[::-1][top_k])
            
            return result
        else:
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action))
            #recommend_score = np.max(tf.keras.backend.dot(items_ebs,action))
            #all_score= np.sum(items_ebs)+np.sum(action)
            #row_sums = tf.reduce_sum(tf.keras.backend.dot(items_ebs, action), axis=0, keepdims=True)
            #percentile_value = tfp.stats.percentile(tf.keras.backend.dot(items_ebs, action), q=100 * recommend_score / recommend_score)
            #percentile_value = recommend_score / tf.reduce_sum(tf.keras.backend.dot(items_ebs,action)) * 100
            
            return items_ids[item_idx]
    

    def bert(self,items_ids):
        items_array = []

        for id in items_ids:
            try:
                items_array.append(self.ipc_to_embedding[id])
            
            except KeyError:
                print('error')
                continue
      
        items_tensor = torch.cat(items_array, dim=0)
        items_tensor_cpu = items_tensor.cpu()
      
        return items_tensor_cpu  
    
    def train(self, max_episode_num, top_k=False, load_model=False):
        # 타겟 네트워크들 초기화
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            self.load_model('./save_model/model_weight_dict2/actor_fixed.h5',
                                './save_model/model_weight_dict2/critic_fixed.h5')
            print('Completely load weights!')

        episodic_precision_history = []
        episodic_loss_history=[]
        save_loss=[]

        for episode in tqdm(range(max_episode_num)):
            if episode == 1868:
              #print('1868 error')
              continue
            # episodic reward 리셋
            else:
              episode_reward = 0
              correct_count = 0
              steps = 0
              q_loss = 0
              mean_action = 0
              # Environment 리셋
              user_id, items_ids, done = self.env.reset()
              #print(f'user_id : {user_id}, rated_items_length:{len(self.env.user_items)}')
              print(items_ids)
              while not done:
                  
                  # Observe current state & Find action
                  ## Embedding 해주기
                  user_eb = np.ones((1, 768,))
                  items_eb = self.bert(items_ids)
                  ## SRM으로 state 출력
                  state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
                  #print('state shape : ', state.shape)


                  ## Action(ranking score) 출력
                  action = self.actor.network(state)


                  ## ε-greedy exploration
                  if self.epsilon > np.random.uniform() and not self.is_test:
                      self.epsilon -= self.epsilon_decay
                      action += np.random.normal(0,self.std,size=action.shape)

                  ## Item 추천
                  recommended_item = self.recommend_item(action, self.env.recommended_items, top_k=top_k)
                  
                  # Calculate reward & observe new state (in env)
                  ## Step
                  #print('추천 된 문서 : ',recommended_item)
                  
                  
                  next_items_ids, reward, done, _ = self.env.step(recommended_item,top_k=top_k)
                  if top_k:
                      reward = np.sum(reward)

                  
                  next_items_ids=next_items_ids[:self.state_size]
                  next_items_eb = self.bert(next_items_ids)
              
                  next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)])

                  # buffer에 저장
                  self.buffer.append(state, action, np.sum(reward), next_state, done)
                  
                  if self.buffer.crt_idx > 1 or self.buffer.is_full:
                      # Sample a minibatch
                      batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = self.buffer.sample(self.batch_size)

                      # Set TD targets
                      target_next_action= self.actor.target_network(batch_next_states)
                      qs = self.critic.network([target_next_action, batch_next_states])
                      target_qs = self.critic.target_network([target_next_action, batch_next_states])
                      min_qs = tf.raw_ops.Min(input=tf.concat([target_qs, qs], axis=1), axis=1, keep_dims=True) # Double Q method
                      td_targets = self.calculate_td_target(batch_rewards, min_qs, batch_dones)
          
                      # Update priority
                      for (p, i) in zip(td_targets, index_batch):
                          self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

                      # print(weight_batch.shape)
                      # print(td_targets.shape)
                      # raise Exception
                      # Update critic network
                      q_loss += self.critic.train([batch_actions, batch_states], td_targets, weight_batch)
                      
                      # Update actor network
                      s_grads = self.critic.dq_da([batch_actions, batch_states])
                      self.actor.train(batch_states, s_grads)
                      self.actor.update_target_network()
                      self.critic.update_target_network()

                  items_ids = next_items_ids
                  episode_reward += reward
                  mean_action += np.sum(action[0])/(len(action[0]))
                  steps += 1

                  if reward > 0:
                      correct_count += 1
                  
                  print(f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}', end='\r')

                  if done:
                      print()
                      precision = int(correct_count/steps * 100)
                      print(f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss/steps}, mean_action : {mean_action/steps}')
                      if self.use_wandb:
                          wandb.log({'precision':precision, 'total_reward':episode_reward, 'epsilone': self.epsilon, 'q_loss' : q_loss/steps, 'mean_action' : mean_action/steps})
                      episodic_precision_history.append(precision)
                      episodic_loss_history.append(q_loss/steps)
              
              #if (episode+1)%50 == 0:
                  #plt.plot(episodic_precision_history)
                  #plt.savefig(os.path.join(self.save_model_weight_dir, f'images/training_precision_%_top_5.png'))

              if (episode+1)%1000 == 0 or episode == max_episode_num-1:
                  self.save_model(os.path.join(self.save_model_weight_dir, f'actor_fixed_final.h5'),
                                  os.path.join(self.save_model_weight_dir, f'critic_fixed_final.h5'))
                  plt.plot(episodic_loss_history)
                  plt.legend(['Q-loss'])
                  plt.show()
              save_loss.append(np.sum(episodic_loss_history))
              loss_df=pd.DataFrame(episodic_loss_history,columns=['loss'])
              loss_df.to_csv('./save_model/model_weight_dict2/loss_final.csv',index=False)  
              #return episodic_precision_history,episodic_loss_history 
              #return np.mean(episodic_loss_history)

                

    def save_model(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

