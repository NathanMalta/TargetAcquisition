#Modified from https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char01%20DQN/DQN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import cv2

import numpy as np
import matplotlib.pyplot as plt
import copy

from Simulator.Constants import IMG_WIDTH, IMG_HEIGHT, NUM_FRAMES_STACKED

# hyper-parameters
BATCH_SIZE = 128
LR = 0.0003
GAMMA = 0.95
EPS_RANGE = [0.2, 1]
SIM_DT_RANGE = [0.05, 0.05]
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 200
NUM_EPISODES = 400

NUM_ACTIONS = 36 #36 actions - between [0, 1] in increments of 0.2 for both right and left voltages
NUM_STATES = IMG_WIDTH * IMG_HEIGHT * NUM_FRAMES_STACKED
ENV_A_SHAPE = 0

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(NUM_STATES, 500)
        # self.fc2 = nn.Linear(500,250)
        # self.fc3 = nn.Linear(250,70)
        # self.out = nn.Linear(70,NUM_ACTIONS)

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()

        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(NUM_STACKED_IMGS, 10, 5), 
        #     nn.MaxPool2d(3, 3), #kernel size, stride
        #     nn.ReLU(),
        #     nn.Conv2d(10,20,5),
        #     nn.MaxPool2d(3, 3), #kernel size, stride
        #     nn.ReLU()
        # )

        # self.fc_layers = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(200, 50), #in features, out features
        #     nn.Linear(50, NUM_ACTIONS)
        # )

        self.conv_layers = nn.Sequential(nn.Conv2d(NUM_FRAMES_STACKED, 32, kernel_size=8, stride=4), #input channels, output channels, kernel size
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.fc_layers = nn.Sequential(nn.Flatten(),
                                        nn.Linear(896, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, NUM_ACTIONS)
                                        )
        
        self.m = nn.Softmax(dim=1)

    def forward(self,x):
        conv_output = self.conv_layers.forward(x)
        model_output = self.fc_layers.forward(conv_output)
        model_output = self.m(model_output)
        return model_output

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR, weight_decay=1e-5)
        self.loss_func = nn.MSELoss()
        self.episilo = EPS_RANGE[0] #start at the beginning end of the epsilon
        self.simDt = SIM_DT_RANGE[0]

    def choose_action(self, state):
        state = self.format_state(state)
        if np.random.randn() <= self.episilo: # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action
    
    def updateHyperParams(self, episodeNumber):
        epsSlope = (EPS_RANGE[1] - EPS_RANGE[0]) / NUM_EPISODES
        self.episilo = EPS_RANGE[0] + epsSlope * episodeNumber

        dtSlope = (SIM_DT_RANGE[1] - SIM_DT_RANGE[0]) / NUM_EPISODES
        self.simDt = SIM_DT_RANGE[0] + dtSlope * episodeNumber
        print(f"updated epsilon: {self.episilo}; updated simDt: {self.simDt}")

    
    def format_state(self, state):
        '''Converts a binary image to a properly formatted torch image vector
        '''
        # state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        state = state.reshape(1, NUM_FRAMES_STACKED, IMG_HEIGHT, IMG_WIDTH) #dim [Dim: (N,C,H,W)]
        state = torch.from_numpy(state)

        return state


    def store_transition(self, state, action, reward, next_state):
        #state and next_state images must be flat to fit into replay memory - we'll reshape them when they come out
        state = state.flatten()
        next_state = next_state.flatten()

        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            print("UPDATING EVAL NET!!")
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]

        #state images are flat in replay memory - unflatten them
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_state = batch_state.reshape(BATCH_SIZE, NUM_FRAMES_STACKED, IMG_HEIGHT, IMG_WIDTH)
        
        #next state images are flat in replay memory - unflatten them
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])
        batch_next_state = batch_next_state.reshape(BATCH_SIZE, NUM_FRAMES_STACKED, IMG_HEIGHT, IMG_WIDTH)

        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])



        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
