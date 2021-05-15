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
seed=666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
save_path='models/'
max_training_timesteps = int(200000)
print_steps=50

max_steps = 50000

eps_clip = 0.2
gamma = 0.99


batch_size=12
mode='test'
env=game.GameState(mode)
action_dim=2
print_reward=0
learning_starts=30000
schedule_timesteps=40000
agent=DQN(action_dim)
agent.load(save_path)
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
        step_reward+=reward
        if done:
            break
    print_reward+=step_reward
    avg_reward=float(print_reward)
    print('\n')
    print('*********reward*********')
    print('steps:  %d, avg reward:  %.2f'%(step,avg_reward))
    print('*********reward*********')
    print('\n')
    print_reward=0
