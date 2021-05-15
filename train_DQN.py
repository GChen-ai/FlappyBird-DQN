import torch
from DQN import DQN
import pybullet_envs
import gym
import cv2
import sys
import random
sys.path.append("game/")
import wrapped_flappy_bird as game
import os
import random
import numpy as np
from collections import deque
seed=6
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
save_path='models/'
max_training_timesteps = int(2e10)
print_steps=1000

max_steps = 30000
save_steps=1000
gamma = 0.99
capacity=70000
lr = 0.00001
learning_starts=30000
schedule_timesteps=40000
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0')
batch_size=64
mode='test'
env=game.GameState(mode)
action_dim=2

agent=DQN(action_dim,gamma,lr,capacity,schedule_timesteps=schedule_timesteps,learning_starts=learning_starts)
#agent.load(save_path)
step_reward=0
steps=0
print_epochs=0
print_reward=0
#env.init(mode)
state,reward,done=env.frame_step(0)
for step in range(1,max_training_timesteps):
    env.init(mode)
    step_reward=0
    state,reward,done=env.frame_step(0)
    state=np.repeat(state,4, axis=-1)
    #print(state.shape)
    #state=agent.data_transform(state).to(device)
    #state=agent.embedding(state)
    f=0
    for t in range(1,max_steps):
        if f%4==0:
            if steps<learning_starts:
                action=np.random.choice(range(action_dim), 1).item()
            else:
                action=agent.select_action(state)
        else:
            action=0
        
        next_state,reward,done=env.frame_step(action)
        #reward = np.clip(reward, -1.0, 1.0)
        next_state=np.array(next_state)
            #next_state=agent.data_transform(next_state).to(device)
            #next_state=agent.embedding(next_state)
        next_state=np.append(next_state,state[:,:,:9],axis=2)
        agent.store_transition(state,action,reward,next_state,done)
        state=next_state
        step_reward+=reward
        steps+=1
        f+=1
        agent.update(batch_size)
        if steps>learning_starts and steps%save_steps==0:
            agent.save(save_path)
        if steps%print_steps==0:
            avg_reward=float(print_reward)/print_epochs
            
            print('*********reward*********')
            print('steps:  %d, avg reward:  %.2f'%(steps,avg_reward))
            print('*********reward*********')
            print('\n')
            print_reward=0
            print_epochs=0
        if done or t>=max_steps-1:
            
            break
    print_reward+=step_reward
    print_epochs+=1
    
    