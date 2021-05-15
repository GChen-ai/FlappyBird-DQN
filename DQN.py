import torch
import torch.nn as nn
from torchvision import models
import torch.distributions
import numpy as np
import random
from torch.utils.data import DataLoader
from BirdDataset import BirdDatasetDQN
import os
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
device = torch.device('cpu')
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0')
class Buffer:
    def __init__(self,capacity):
        self.actions=[None]*capacity
        self.states=[None]*capacity
        self.rewards=[None]*capacity
        self.next_states=[None]*capacity
        self.is_terminate=[None]*capacity


class Net(nn.Module):
    def __init__(self,action_dim):
        super(Net,self).__init__()
        
        #vgg = models.vgg16(pretrained=True)
        #features = list(vgg.features.children())[:10]
        #self.conv=nn.Sequential(*features)
        self.conv=nn.Sequential(
            nn.Conv2d(12,64,8,4,4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,4,2,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU()
        )
        #self.pooling=nn.AdaptiveAvgPool2d(1)
        self.linear=nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(256),
            nn.Linear(256,action_dim)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
                nn.init.orthogonal_(m.weight)
                #m.weight=m.weight*0.1
                #print(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
                nn.init.orthogonal_(m.weight)
                #m.weight=m.weight*0.1
                #print(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self,x):
        x=self.conv(x)
        #x=self.pooling(x)
        x=x.reshape(x.shape[0],-1)
        output=self.linear(x)
        return output


    
class DQN:
    def __init__(self,action_dim,gamma=0.99,lr=0.003,capacity=50000,initial_p=1.0,final_p=0.1,schedule_timesteps=40000,learning_starts=30000):
        self.gamma=gamma
        self.action_dim=action_dim
        self.QNet=Net(action_dim).to(device)
        self.targetQNet=Net(action_dim).to(device)
        self.targetQNet.load_state_dict(self.QNet.state_dict())
        self.capacity=capacity
        self.buffer=Buffer(self.capacity)
        self.mse=nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr)
        self.memory_count=0
        self.update_count=0
        self.print_count=0
        self.t=0
        self.initial_p=initial_p
        self.final_p=final_p
        self.schedule_timesteps=schedule_timesteps
        self.learning_starts=learning_starts
    def select_action(self,state):
        self.t+=4
        fraction  = min(float(self.t) / self.schedule_timesteps, 1.0)
        eps=self.initial_p + fraction * (self.final_p - self.initial_p)
        state=self.data_transform(state).unsqueeze(0).to(device)
        value = self.QNet(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) <= eps: # epslion greedy
            action = np.random.choice(range(self.action_dim), 1).item()
        return action
    def store_transition(self,states,actions,rewards,next_states,done):
        index = self.memory_count % self.capacity
        self.buffer.states[index]=states#self.data_transform(states)
        self.buffer.actions[index] = actions
        self.buffer.rewards[index]=rewards
        self.buffer.next_states[index]=next_states#self.data_transform(next_states)
        self.buffer.is_terminate[index]=done
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self,batchsize):
        if self.memory_count >= self.learning_starts:
        #rewards=torch.FloatTensor(rewards).to(device)
        #rewards=(rewards)/(np.std(rewards)+1e-8)
        #rewards=np.clip(rewards, -20, 20)
            #rewards=np.array([r for r in self.buffer.rewards])
            #rewards = (rewards) / (np.std(rewards) + 1e-7)
            
            bird_dataset=BirdDatasetDQN(self.buffer,self.memory_count,self.capacity)
            bird_dataloader = DataLoader(bird_dataset, batch_size=batchsize, shuffle=True, num_workers=4,drop_last=True)

            for i, batch_samples in enumerate(bird_dataloader):
                train_states, train_actions, train_next_states, train_rewards, is_terminate= batch_samples["state"], batch_samples["action"], batch_samples["next_state"],batch_samples["reward"],batch_samples['is_terminate']
                
                '''print(torch.mean(train_states[:,0,:,:]))
                print(torch.mean(train_states[:,1,:,:]))
                print(torch.mean(train_states[:,2,:,:]))
                
                print(torch.std(train_states[:,0,:,:]))
                print(torch.std(train_states[:,1,:,:]))
                print(torch.std(train_states[:,2,:,:]))
                print('**************************')'''
                train_states=train_states.to(device)
                train_actions=train_actions.unsqueeze(1).to(device)
                train_next_states=train_next_states.to(device)
                train_rewards=train_rewards.to(device)
                target_v=torch.zeros(train_rewards.shape).to(device)
                with torch.no_grad():
                    for j in range(batchsize):
                        if is_terminate[j]:
                            target_v[j] = train_rewards[j]
                        else:
                            target_v[j] = train_rewards[j]+self.gamma * self.targetQNet(train_next_states[j].unsqueeze(0)).max(1)[0]
                #print(target_v)
                #print(target_v)
                #print(train_actions.shape)
                #print(self.QNet(train_states).shape)
                v=self.QNet(train_states).gather(1, train_actions)
                #print(v.shape)
                loss = self.mse(target_v.unsqueeze(1), v)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #
                #print('update')
                self.update_count+=1
                self.print_count+=1
            #self.opti_scheduler.step()
                if self.print_count%100==0:
                    print("DQN loss: %.5f"%(loss.item()))
                if self.update_count%1000==0:
                    self.targetQNet.load_state_dict(self.QNet.state_dict())
                break

    def save(self,path):
        if not os.path.exists(path):
          os.makedirs(path)
        torch.save(self.targetQNet.state_dict(),path+'policy.pth')
    def load(self,path):
        self.QNet.load_state_dict(torch.load(path+'policy.pth'))
        self.targetQNet.load_state_dict(torch.load(path+'policy.pth'))
    def data_transform(self,state):
        #state=np.array(state)
        mean_std=([0.5203,0.7677,0.6946],[0.1790,0.0876,0.1828])
        #state=Image.fromarray(state)
        img_transform = transforms.Compose([
            #transforms.Resize((112,112)),
            transforms.ToTensor(),
            #transforms.Normalize(*mean_std),
    ])
        
        state=img_transform(state)
        return state
    
        
            
            
    