#TAKEN AND MODIFIED FROM: https://github.com/MoritzTaylor/ddpg-pytorch
#LICENSE: MIT LICENSE

from xml.parsers.expat import model
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Simulator.Constants import IMG_WIDTH, IMG_HEIGHT, NUM_FRAMES_STACKED

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = len(action_space)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(NUM_FRAMES_STACKED, 2, kernel_size=3), 
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
                                nn.Linear(7636, 1028),
                                nn.LayerNorm(1028),
                                nn.ReLU(True),
                                nn.Linear(1028, 512),
                                nn.LayerNorm(512),
                                nn.ReLU(True),
                                nn.Linear(512, 256),
                                nn.LayerNorm(256),
                                nn.ReLU(True),
                                nn.Linear(256, num_outputs)
                                )
        for layer in self.fc_layers:
            if type(layer) is nn.Linear:
                fan_in_uniform_init(layer.weight)
                fan_in_uniform_init(layer.bias)
        
        nn.init.uniform_(self.fc_layers[-1].weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.fc_layers[-1].bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

            

    def forward(self, inputs):
        x = inputs

        conv_output = self.conv_layers.forward(x)
        model_output = self.fc_layers.forward(conv_output)

        # Output
        mu = torch.tanh(model_output)
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = len(action_space)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(NUM_FRAMES_STACKED, 2, kernel_size=3), 
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
                                nn.Linear(7636 + num_outputs, 1028),
                                nn.LayerNorm(1028),
                                nn.ReLU(True),
                                nn.Linear(1028, 512),
                                nn.LayerNorm(512),
                                nn.ReLU(True),
                                nn.Linear(512, 256),
                                nn.LayerNorm(256),
                                nn.ReLU(True),
                                nn.Linear(256, 1)
                                )
        
        for layer in self.fc_layers:
            if type(layer) is nn.Linear:
                fan_in_uniform_init(layer.weight)
                fan_in_uniform_init(layer.bias)
        
        nn.init.uniform_(self.fc_layers[-1].weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.fc_layers[-1].bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)



    def forward(self, inputs, actions):
        x = inputs

        conv_output = self.conv_layers.forward(x)
        conv_output = torch.cat((conv_output, actions), 1)
        model_output = self.fc_layers.forward(conv_output)

        return model_output
