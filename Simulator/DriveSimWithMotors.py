import math
import numpy as np
from scipy import integrate
from Constants import massTotal, massDriveTrain, halfWheelbase, wheelAxisToCg, wheelDiameter, moiTotal, moiWheelAxial, gearRatio #drivetrain constants
from Constants import Kt, D, J, Kv, R, L #import motor constants

#using the differential drive kinematic model from https://www.semanticscholar.org/paper/Dynamic-Modelling-of-Differential-Drive-Mobile-and-Dhaouadi-Hatab/e919330857f80116050078a311953da07ec8876b?p2df

class DriveSimWithMotors:
    def __init__(self, initialConditions, leftVoltageFunc, rightVoltageFunc):
        self.initialConditions = initialConditions
        self.leftVoltageFunc = leftVoltageFunc
        self.rightVoltageFunc = rightVoltageFunc

    def getDerivatives(self, t, y):
        linearVel = y[0]
        angularVel = y[1]
        theta = y[2]
        xPos = y[3]
        yPos = y[4]
        leftI = y[5]
        rightI = y[6]

        #Motor Calculations
        #get motor input voltages
        leftVoltage = self.leftVoltageFunc(y, t)
        rightVoltage = self.rightVoltageFunc(y, t)
        
        #calculate angular velocity at the motor shafts
        wheelRadius = wheelDiameter / 2
        leftAngVel = (linearVel - halfWheelbase * angularVel) * (1/wheelRadius) / gearRatio
        rightAngVel = (linearVel + halfWheelbase * angularVel) * (1/wheelRadius) / gearRatio

        #calculate currents
        rightIDot = - (Kv / L) * rightAngVel - (R/L) * rightI + (1/L) * rightVoltage
        leftIDot = - (Kv / L) * leftAngVel - (R/L) * leftI + (1/L) * leftVoltage

        rightTorque = (Kt * rightI - D * rightAngVel) / gearRatio
        leftTorque = (Kt * leftI - D * leftAngVel) / gearRatio

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

    def getOutputs(self, times):
        return integrate.odeint(self.getDerivatives, self.initialConditions, times, tfirst=True, rtol=0, atol=1e-5)

    def getOutputsSolveIVP(self, timeSpan):
        return integrate.solve_ivp(self.getDerivatives, timeSpan, self.initialConditions, method='RK45', rtol=1e-5, atol=1e-5)



if __name__ == '__main__':
    #Let's see if this works...

    import matplotlib.pyplot as plt

    def leftVelFunc(y, t):
        return 12 #constant input

    def rightVelFunc(y, t):
        return 6 #constant input

    initialConds = np.array([0,0, 0,0,0, 0,0]) #starting with 0A input and 0 velocity
    motor = DriveSimWithMotors(initialConds, leftVelFunc, rightVelFunc)
    times = np.linspace(0, 5, 1001)
    results = motor.getOutputsSolveIVP([0, 5])
   
    # plt.plot(results.t, results.y[0, :], 'y', label='vel(t)')
    # plt.plot(results.t, results.y[1, :], 'c', label='w(t)')
    # plt.plot(results.t, results.y[2, :], 'b', label='theta(t)')
    # plt.plot(results.t, results.y[5, :], 'r', label='left_i(t)')
    # plt.plot(results.t, results.y[6, :], 'g', label='right_i(t)')

    # plt.plot(results.y[4, :], results.y[3, :], 'g', label='right_i(t)')

    # leftVel = (results[:, 0] - halfWheelbase * results[:, 1]) * (1/(wheelDiameter/2))
    # rightVel = (results[:, 0] + halfWheelbase * results[:, 1]) * (1/(wheelDiameter/2))

    # plt.plot(times, leftVel, 'r', label='leftVel')
    # plt.plot(times, rightVel, 'g', label='rightVel')
    

    plt.legend(loc='best')
    plt.xlabel('time (sec)')
    # plt.ylabel('current (A)\nvelocity (rad/s)')
    plt.grid()
    plt.show()
