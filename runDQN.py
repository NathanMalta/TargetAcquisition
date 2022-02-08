import cv2
import copy
import matplotlib.pyplot as plt

from Simulator import Constants
from DQN.DQN import DQN, MEMORY_CAPACITY
from DQN.DriveEnv import DriveEnv

def main():
    dqn = DQN()
    episodes = 400
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    env = DriveEnv()

    #fill replay memory
    while dqn.memory_counter < MEMORY_CAPACITY:
        episodeMem = env.runEpisode(dqn)
        print(f"filling memory... {dqn.memory_counter} / {MEMORY_CAPACITY}")
        for transition in episodeMem:
            dqn.store_transition(*transition)
            reward = transition[2]

    voltFig, voltAx = plt.subplots()
    for i in range(episodes):
        print(f"episode {i} learn step: {dqn.learn_step_counter}")
        ep_reward = 0
        dqn.updateHyperParams(i) #update epsilon based on how many runs we've done
        env = DriveEnv(dt=dqn.simDt)
        episodeMem = env.runEpisode(dqn)
        for transition in episodeMem:
            dqn.store_transition(*transition)
            reward = transition[2]
            ep_reward += reward
            
            frameState = transition[0].reshape([Constants.NUM_FRAMES_STACKED, Constants.IMG_HEIGHT, Constants.IMG_WIDTH])
            for i in range(frameState.shape[0]):
                cv2.imshow(f"frame{i}", frameState[i])
            cv2.waitKey(1)

            # action = transition[1]
            # leftVoltage = ((action // 6) / 5) * 12 # - 4.8
            # rightVoltage = ((action % 6) / 5) * 12 # - 4.8
            # print(f"left voltage: {leftVoltage} right voltage: {rightVoltage}")
            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()

        voltAx.cla()
        voltAx.plot(env.leftVoltageRecords, 'g', label='left_voltages')
        voltAx.plot(env.rightVoltageRecords, 'r', label='right_voltages')
        
        print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))

        r = copy.copy(ep_reward)
        reward_list.append(r)
        ax.set_xlim(0, episodes)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
    plt.pause(1E8)
    

if __name__ == '__main__':
    main()