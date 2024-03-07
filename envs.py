import numpy as np
import random
import torch
import tensorflow as tf
class OfflineEnv(object):
    
    def __init__(self, users_dict, users_history_lens, items_id_to_name, state_size, ipc_to_embedding, fix_user_id=None):

        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        self.items_id_to_name = items_id_to_name
        
        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id
        self.ipc_to_embedding = ipc_to_embedding
        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 3000
        
    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users
    
    def reset(self):        
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        
        return self.user, self.items, self.done
    

    def reset_all(self):
        self.user=self.user
        self.user_items = self.user_items
        chit_items = [data[0] for data in self.users_dict[self.user]]
        self.items=list(self.items_id_to_name.keys())
        
        plus_items=random.sample(self.items,len(chit_items))
        self.items=chit_items+plus_items
        self.done = self.done
        return self.user, self.items, self.done
        
    def step(self, action,top_k=False):

        reward = -0.5

        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append(self.user_items[act])
                else:
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards

        else:
            #print(score)
            #print('업데이트 전 : ',self.items)
            #if str(action) in self.user_items.keys() and str(action) not in self.recommended_items:
            if action in self.user_items.keys() and action not in self.recommended_items:
                #print('score : ', score)  
                #print('action : ',action)
                #print(self.user_items)
                #items_ids=list(self.user_items.keys())
                #print('action : ',action)
            #print(items_ids)
            #items_ids=list(items_ids)
                #item_list=[]
                #for id in items_ids:
                  #item_list.append(self.ipc_to_embedding[id])
                #items_tensor = torch.cat(item_list, dim=0)
                #items_tensor_cpu = items_tensor.cpu()
                #action_eb=self.ipc_to_embedding[action]
            #이제 items_tensor_cpu를 NumPy 배열로 변환할 수 있습니다.
                #items_numpy = items_tensor_cpu.numpy()
                #items_ebs=tf.convert_to_tensor(items_numpy)
                #items_ebs=tf.cast(items_ebs,tf.float32)
                #print(items_ebs.shape)
                #items_ebs=tf.transpose(items_ebs,perm=(1,0))
                #print(items_ebs.shape)
                #action_eb_cpu=action_eb.cpu()
                #action_numpy = action_eb_cpu.numpy()
                #action_ebs=tf.convert_to_tensor(action_numpy)
                #action_ebs=tf.transpose(action_ebs, perm=(1,0))
                #action_ebs=tf.cast(action_ebs,tf.float32)
                #action = tf.cast(action_ebs, tf.float32)

                #item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action_ebs))
                #item=items_ids[item_idx]


                reward = self.user_items[action] # reward
            if reward > 0:
                self.items = self.items[1:] + [action]
                #print('업데이트 후 : ',self.items)
            self.recommended_items.add(action)
        #print('실제 길이',len(self.user_items))
        #print('테스트 길이',len(self.recommended_items) )
        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= len(self.user_items):
            self.done = True
            
        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names

