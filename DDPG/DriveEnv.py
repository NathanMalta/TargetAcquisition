import math
from collections import deque
import random

import numpy as np
from scipy import integrate
from scipy.integrate._ivp import ivp
import cv2
import torch

from Simulator.Constants import IMG_HEIGHT, IMG_WIDTH, massTotal, massDriveTrain, halfWheelbase, wheelAxisToCg, wheelDiameter, moiTotal, moiWheelAxial, gearRatio #drivetrain constants
from Simulator.Constants import Kt, D, J, Kv, R, L #import motor constants
from Simulator.Constants import MAX_SIM_TIME, NUM_FRAMES_STACKED #reinforcement learning constants
from Simulator.CameraSim import CameraSim

#using the differential drive kinematic model from https://www.semanticscholar.org/paper/Dynamic-Modelling-of-Differential-Drive-Mobile-and-Dhaouadi-Hatab/e919330857f80116050078a311953da07ec8876b?p2df

class DriveEnv:
    def __init__(self, dt=0.05, startingPose=[0,0,0]): #startingPose: [xPos, yPos, heading]
        self.startingState = [0,0, startingPose[2], startingPose[0], startingPose[1], 0, 0]
        self.action_space = [[-1,1], [-1,1]]

        self.dt = dt #Note this is the update of the controller TODO: better name

        self.targetPose = [10,0,0] #xPos, yPos, heading of the target TODO: make this configurable
        
        self.cameraSim = CameraSim()
        self.reset()



    def reset(self):
        startingPos = self.getRandomizedStartingPosition([-3, 3], [-3, 3], [-math.radians(-5), math.radians(5)])
        self.startingState[3] = startingPos[0]
        self.startingState[4] = startingPos[1]
        self.startingState[2] = startingPos[2]
        self.lastLeftVoltage = None
        self.lastRightVoltage = None

        #update info
        self.controllerUpdateNum = 0 #update number
        self.controllerRecords = []
        self.lastFrameStack = deque([], maxlen = NUM_FRAMES_STACKED)

        self.leftVoltageRecords = []
        self.rightVoltageRecords = []

        for _ in range(NUM_FRAMES_STACKED):
            self.lastFrameStack.append(np.zeros((IMG_HEIGHT, IMG_WIDTH)))
        
    def getRandomizedStartingPosition(self, xRange, yRange, headingRange):
        x = random.uniform(*xRange)
        y = random.uniform(*yRange)
        heading = math.tan((-y - self.targetPose[1]) / (x - self.targetPose[0])) + random.uniform(*headingRange)

        return [x, y, heading]



        
    def getDerivatives(self, t, y, agent, ou_noise):
        linearVel = y[0]
        angularVel = y[1]
        theta = y[2]
        xPos = y[3]
        yPos = y[4]
        leftI = y[5]
        rightI = y[6]

        if self.controllerUpdateNum * self.dt <= t:
            #controller update needed!

            #get image from camera at this position
            currentFrame = self.getFrame(xPos, yPos, theta)
            self.lastFrameStack.append(currentFrame)
            currentFrameState = np.vstack((self.lastFrameStack))
            currentFrameState = torch.tensor(currentFrameState.flatten(), dtype=torch.float)
            #convert this image into an action from the DQN
            action = agent.calc_action(torch.tensor(currentFrameState.flatten(), dtype=torch.float), ou_noise)
            #store state, frame, and action in controllerRecords
            self.controllerRecords.append([y, currentFrameState, action])

            #translate action number from DQN into voltages for motors
            leftVoltage = action[0] * 12
            rightVoltage = action[1] * 12

            if leftVoltage < -12 or rightVoltage < -12 or leftVoltage > 12 or rightVoltage > 12:
                print(f"left: {leftVoltage} right: {rightVoltage}")

            #update last voltages, update number
            self.lastLeftVoltage = leftVoltage
            self.lastRightVoltage = rightVoltage
            self.controllerUpdateNum += 1

            #add voltages to records for graphing
            self.leftVoltageRecords.append(leftVoltage)
            self.rightVoltageRecords.append(rightVoltage)
        else:
            #no controller update needed - just use values from last controller
            leftVoltage = self.lastLeftVoltage
            rightVoltage = self.lastRightVoltage

        #Motor Calculations
        
        #calculate angular velocity at the motor shafts
        wheelRadius = wheelDiameter / 2
        leftAngVel = (linearVel - halfWheelbase * angularVel) * (1/wheelRadius) / gearRatio
        rightAngVel = (linearVel + halfWheelbase * angularVel) * (1/wheelRadius) / gearRatio

        #calculate currents
        rightIDot = - (Kv / L) * rightAngVel - (R/L) * rightI + (1/L) * leftVoltage
        leftIDot = - (Kv / L) * leftAngVel - (R/L) * leftI + (1/L) * rightVoltage

        #calculate torques produced by left and right motors
        #Note: assumes viscous drag (D) and counter torque due to MoI (J) are very small.  TODO: remove assumptions
        rightTorque = (Kt * rightI) / gearRatio 
        leftTorque = (Kt * leftI) / gearRatio

        #Drivetrain Calculations

        #calculate derivative of linear velocity according to equation 47 of AUS source
        vDot = (1 / (massTotal + (2 * moiWheelAxial) / (wheelRadius ** 2))) * (((1 / wheelRadius) * (rightTorque + leftTorque)) + (massTotal * wheelAxisToCg * angularVel ** 2))

        #calculate derivative of angular velocity according to equation 47 of AUS source
        wDot = (1 / (moiTotal + (2 * halfWheelbase ** 2) / (wheelRadius**2) * moiWheelAxial)) * ((halfWheelbase / wheelRadius * (rightTorque - leftTorque)) - (massDriveTrain * wheelAxisToCg * angularVel * linearVel))

        #calculate derivative of theta (same as angular vel)
        thetaDot = angularVel

        #calculate changes in x and y position
        xDot = linearVel * math.cos(theta)
        yDot = linearVel * math.sin(theta)

        return np.array([vDot, wDot, thetaDot, xDot, yDot, leftIDot, rightIDot])

    def runEpisode(self, agent, ou_noise):
        '''Runs the simulator for one episode using the given dqn
        '''
        self.reset() #reset last voltages, last update time
        timeSpan = [0, MAX_SIM_TIME]
        evalTimes = np.arange(0, MAX_SIM_TIME, self.dt)
        ivpOutput = integrate.solve_ivp(self.getDerivatives, timeSpan, self.startingState, method='RK45', args=(agent, ou_noise,), max_step=self.dt / 10, atol=1e-8, t_eval=evalTimes)

        memory = [] 

        for i in range(1, len(self.controllerRecords) - 1):
            #each array element contains [state, frame, action]            
            prev = self.controllerRecords[i - 1]
            curr = self.controllerRecords[i]
            next = self.controllerRecords[i + 1]
            reward = self.getReward(curr[0], prev[0])

            currFrameState = curr[1]
            nextFrameState = next[1]
            action = curr[2]

            isDone = False
            if 255 not in currFrameState:
                #handle robot turns away from target - really bad
                currentDist = math.hypot(curr[0][3] - self.targetPose[0], curr[0][4] - self.targetPose[1])
                isDone = True
                if currentDist > 2:
                    reward = -1

                memory.append([currFrameState, action, reward, nextFrameState, isDone])
                break
            else:
                memory.append([currFrameState, action, reward, nextFrameState, isDone]) 

        return memory

        
    
    def getFrame(self, xPos, yPos, theta):
        '''Gets the view of the target at the robot's current position
        '''
        self.cameraSim.setCamPos(yPos, 0, xPos, theta)
        return self.cameraSim.getFrame()

    def getReward(self, currentState, prevState):
        currentPos = (currentState[3], currentState[4])
        prevPos = (prevState[3], prevState[4])

        #add portion of reward from vehicle getting closer to target
        currentDist = math.hypot(currentPos[0] - self.targetPose[0], currentPos[1] - self.targetPose[1])
        prevDist = math.hypot(prevPos[0] - self.targetPose[0], prevPos[1] - self.targetPose[1])
        dist = prevDist - currentDist
        reward = dist #math.copysign(dist**2, dist)

        if abs(reward) > 100:
            print("LARGE MAG REWEARD DETECTED!!")
            print(f"reward: {reward}")
            print(f"curentPos: {currentPos}, prevPos: {prevPos}")
            print(f"currentDist: {currentDist}, prevDist: {prevDist}")
            print(f"currentState: {currentState}, prevState: {prevState}")
            import time
            while(True):
                time.sleep(1)

        return reward
        


if __name__ == '__main__':
    #Let's see if this works...

    import matplotlib.pyplot as plt

    env = DriveEnv()
    results = env.runEpisode(None)

   

    plt.plot(results.t, results.y[0, :], 'y', label='vel(t)')
    plt.plot(results.t, results.y[1, :], 'c', label='w(t)')
    plt.plot(results.t, results.y[2, :], 'b', label='theta(t)')
    plt.plot(results.t, results.y[5, :], 'r', label='left_i(t)')
    plt.plot(results.t, results.y[6, :], 'g', label='right_i(t)')

    # leftVel = (results[:, 0] - halfWheelbase * results[:, 1]) * (1/(wheelDiameter/2))
    # rightVel = (results[:, 0] + halfWheelbase * results[:, 1]) * (1/(wheelDiameter/2))

    # plt.plot(times, leftVel, 'r', label='leftVel')
    # plt.plot(times, rightVel, 'g', label='rightVel')
    

    plt.legend(loc='best')
    plt.xlabel('time (sec)')
    # plt.ylabel('current (A)\nvelocity (rad/s)')
    plt.grid()
    plt.show()
