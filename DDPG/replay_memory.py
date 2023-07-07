import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        state, action, done, next_state, reward = args
        #convert to cpu tensors to save vram
        state = state.cpu()
        action = action.cpu()
        done = done.cpu()
        next_state = next_state.cpu()
        reward = reward.cpu()

        self.memory[self.position] = Transition(state, action, done, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
