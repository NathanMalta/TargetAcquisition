import math
import numpy as np
from scipy import integrate
from Constants import massTotal, massDriveTrain, halfWheelbase, wheelAxisToCg, wheelDiameter, moiTotal, moiWheelAxial

#using the differential drive kinematic model from https://www.semanticscholar.org/paper/Dynamic-Modelling-of-Differential-Drive-Mobile-and-Dhaouadi-Hatab/e919330857f80116050078a311953da07ec8876b?p2df

class DriveSim:
    def __init__(self, initialConditions, leftTorqueFunction, rightTorqueFunction):
        self.initialConditions = initialConditions
        self.leftTorqueFunction = leftTorqueFunction
        self.rightTorqueFunction = rightTorqueFunction

    def getDerivatives(self, y, t):
        leftTorque = self.leftTorqueFunction(t)
        rightTorque = self.rightTorqueFunction(t)
        linearVel = y[0]
        angularVel = y[1]
        theta = y[2]
        xPos = y[3]
        yPos = y[4]

        wheelRadius = wheelDiameter / 2

        #calculate derivative of linear velocity according to equation 47 of AUS source
        vDot = (1 / (massTotal + (2 * moiWheelAxial) / (wheelRadius ** 2))) * (((1 / wheelRadius) * (rightTorque + leftTorque)) + (massTotal * wheelAxisToCg * angularVel ** 2))

        #calculate derivative of angular velocity according to equation 47 of AUS source
        wDot = (1 / (moiTotal + (2 * halfWheelbase ** 2) / (wheelRadius**2) * moiWheelAxial)) * ((halfWheelbase / wheelRadius * (rightTorque - leftTorque)) - (massDriveTrain * wheelAxisToCg * angularVel * linearVel))

        #calculate derivative of theta (same as angular vel)
        thetaDot = angularVel

        #calculate changes in x and y position
        xDot = linearVel * math.cos(theta)
        yDot = linearVel * math.sin(theta)

        return np.array([vDot, wDot, thetaDot, xDot, yDot])

    def getOutputs(self, times):
        return integrate.odeint(self.getDerivatives, self.initialConditions, times, rtol=0, atol=1e-6)


if __name__ == '__main__':
    #Let's see if this works...

    import matplotlib.pyplot as plt

    def leftTorqueFunc(t):
        return 1 #constant input

    def rightTorqueFunc(t):
        return 2 #constant input

    initialConds = np.array([0,0,0,0,0]) #starting with 0A input and 0 velocity
    motor = DriveSim(initialConds, leftTorqueFunc, rightTorqueFunc)
    times = np.linspace(0, 100, 10001)
    results = motor.getOutputs(times)

    xCenter = (np.max(results[:,3]) + np.min(results[:,3])) * 0.5
    yCenter = (np.max(results[:,4]) + np.min(results[:,4])) * 0.5
    rad = (np.max(results[:,3]) - np.min(results[:,3])) * 0.5
    errs = []
    for i in range(results.shape[0]):
        x = results[i,3]
        y = results[i,4]
        errs.append(abs(rad - math.dist([x,y], [xCenter, yCenter])))
    
    plt.plot(results[:, 3], results[:, 4], 'ro', label='path')

    # plt.plot(times, errs, 'y', label='vel(t)')
    # plt.plot(times, results[:, 0], 'y', label='vel(t)')
    # plt.plot(times, results[:, 1], 'c', label='w(t)')
    # plt.plot(times, results[:, 2], 'b', label='theta(t)')
    

    plt.legend(loc='best')
    plt.xlabel('time (sec)')
    # plt.ylabel('current (A)\nvelocity (rad/s)')
    plt.grid()
    plt.show()
