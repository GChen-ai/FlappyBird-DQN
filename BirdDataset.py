from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0')
class BirdDataset(Dataset):
    def __init__(self,data,label,logprob,reward):

        self.data = data
        self.label=label
        self.logprob=logprob
        self.reward=reward
    def __getitem__(self, index):
        state=self.data[index]
        action=self.label[index]
        logprob=self.logprob[index]
        reward=self.reward[index]
        sample={'state':state,'action':action,'logprob':logprob,'reward':reward}
        return sample
    
    def __len__(self):
        return len(self.data)

class BirdDatasetDQN(Dataset):
    def __init__(self,buffer,memory_count,capacity):
        self.state = buffer.states
        self.action=buffer.actions
        self.next_state=buffer.next_states
        self.reward=np.array(buffer.rewards)
        self.reward=(self.reward[:memory_count]) / (np.std(self.reward[:memory_count]) + 1e-7)
        self.is_terminate=buffer.is_terminate
        self.img_transform = transforms.Compose([
            #transforms.Resize((112,112)),
            transforms.ToTensor(),
            #transforms.Normalize(*mean_std),
        ])
        self.memory_count=min(memory_count,capacity)
    def __getitem__(self, index):
        state=self.img_transform(self.state[index])
        action=self.action[index]
        next_state=self.img_transform(self.next_state[index])
        reward=self.reward[index]
        is_terminate=self.is_terminate[index]
        sample={'state':state,'action':action,'next_state':next_state,'reward':reward,'is_terminate':is_terminate}
        return sample
    
    def __len__(self):
        return self.memory_count