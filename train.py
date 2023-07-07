#TAKEN AND MODIFIED FROM: https://github.com/MoritzTaylor/ddpg-pytorch
#LICENSE: MIT LICENSE

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import wandb

from DDPG.ddpg import DDPG
from DDPG.noise import OrnsteinUhlenbeckActionNoise, NormalNoise
from DDPG.replay_memory import ReplayMemory, Transition

from Simulator.Constants import WIN_THRESHOLD, IMG_HEIGHT, IMG_WIDTH, NUM_FRAMES_STACKED
from DDPG.DriveEnv import DriveEnv

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse given arguments
# gamma, tau, hidden_size, replay_size, batch_size, hidden_size are taken from the original paper
parser = argparse.ArgumentParser()
parser.add_argument("--render_train", default=False, type=bool,
                    help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=False, type=bool,
                    help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default="./saved_models/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--timesteps", default=1e6, type=int,
                    help="Num. of total timesteps of training (default: 1e6)")
parser.add_argument("--batch_size", default=2048, type=int,
                    help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e6, type=int,
                    help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.0005,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.01, type=int,
                    help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=2, default=[400, 300], type=tuple,
                    help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=10, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
args = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

def update_wandb(reward, policy_loss, value_loss, iteration):
    # Update wandb
    wandb.log({'reward': reward,
               'policy_loss': policy_loss,
               'value_loss': value_loss,
               'episode_number': iteration,
               })

if __name__ == "__main__":
    wandb.init(project="target-aquisition", 
               config = {
                   
               })

    # Create the env
    env = DriveEnv()

    # Define the reward threshold when the task is solved (if existing) for model saving
    reward_threshold = WIN_THRESHOLD

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)
    agent = DDPG(args.gamma,
                 args.tau,
                 hidden_size,
                 IMG_HEIGHT * IMG_WIDTH * NUM_FRAMES_STACKED,
                 env.action_space,
                 )

    # Initialize replay memory
    memory = ReplayMemory(int(args.replay_size))

    # Initialize OU-Noise
    nb_actions = len(env.action_space)
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(args.noise_stddev) * np.ones(nb_actions))
    
    normal_noise = NormalNoise()

    # Define counters and other variables
    start_step = 0
    # timestep = start_step
    if args.load_model:
        # Load agent if necessary
        start_step, memory = agent.load_checkpoint()
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    # Start training
    logger.info('Doing {} timesteps'.format(args.timesteps))
    logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    episodeNum = 0
    voltFig, voltAx = plt.subplots()
    while episodeNum <= 4000:
        epoch_return = 0
        normal_noise.step()
    
        episodeMem = env.runEpisode(agent, normal_noise)
        episodeNum += 1
        episodeReward = 0
        for transition in episodeMem:
            state, action, reward, next_state, done = transition
            epoch_return += reward
            episodeReward += reward

            #convert to tensor
            state = torch.Tensor(np.array([state.numpy()])).to(device)
            action = torch.Tensor(np.array([action.cpu().numpy()])).to(device)
            reward = torch.Tensor(np.array([reward])).to(device)
            next_state = torch.Tensor(np.array([next_state.numpy()])).to(device)
            mask = torch.Tensor(np.array([done])).to(device)

            memory.push(state, action, mask, next_state, reward)


        print(f"episode {episodeNum} reward: {episodeReward}")

        epoch_value_loss = 0
        epoch_policy_loss = 0
        if len(memory) > args.batch_size:
            transitions = memory.sample(args.batch_size)
            # Transpose the batch
            # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
            batch = Transition(*zip(*transitions))

            # Update actor and critic according to the batch
            value_loss, policy_loss = agent.update_params(batch)

            epoch_value_loss += value_loss
            epoch_policy_loss += policy_loss

            print(f'Value loss: {value_loss} Policy loss: {policy_loss}')
            update_wandb(epoch_return, policy_loss, value_loss, episodeNum)        


        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)

        # Test every 10th episode run a test episode
        if episodeNum % 10 == 0:
            t += 1
            episodeMem = env.runEpisode(agent, None)
            episodeReward = 0
            for transition in episodeMem:
                state, action, reward, next_state, done = transition
                epoch_return += reward
                episodeReward += reward
                # cv2.imshow(f"frame", state.numpy().reshape(NUM_FRAMES_STACKED, IMG_HEIGHT, IMG_WIDTH)[NUM_FRAMES_STACKED - 1])
                # cv2.waitKey(1)

            print(f'Test episode reward: {episodeReward}')

        #save every 100 episodes
        if episodeNum % 100 == 0:
            agent.save_checkpoint(episodeNum, memory)
            logger.info('Saved model at episode {}'.format(episodeNum))