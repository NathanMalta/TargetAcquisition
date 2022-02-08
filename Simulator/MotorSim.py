import math
import numpy as np
from scipy import integrate

from Constants import Kt, D, J, Kv, R, L #import motor constants

class MotorSim:

    def __init__(self, initialConditions, voltageFunc, inputTorqueFunc):
        self.initialConditions = initialConditions
        self.voltageFunc = voltageFunc #gives voltage as a function of time
        self.inputTorqueFunc = inputTorqueFunc #gives input torque as a function of time
    
    def getDerivatives(self, y, t):
        withoutForcing = np.array([[-D/J,  Kt/J], \
                                   [-Kv/L, -R/L]])
        voltageForcing = np.array([0, 1/L]) * self.voltageFunc(t)
        torqueForcing = np.array([1/J, 0]) * self.inputTorqueFunc(t)

        fullDeriv = np.matmul(withoutForcing, y) + voltageForcing + torqueForcing
        # print(f"before forcing: {np.matmul(withoutForcing, y)}")
        # print(f"after forcing: {fullDeriv} {y}")

        return fullDeriv

    def getOutputs(self, times):
        return integrate.odeint(self.getDerivatives, self.initialConditions, times, rtol=0, atol=1e-10)

if __name__ == '__main__':
    #Let's see if this works...

    import matplotlib.pyplot as plt

    def vFunc(t):
        return 12 #constant input

    def tFunc(t):
        return 0 #no input torque - motor is free spinning

    initialConds = np.array([0,0]) #starting with 0A input and 0 velocity
    motor = MotorSim(initialConds, vFunc, tFunc)
    times = np.linspace(0, 2, 1001)
    results = motor.getOutputs(times)

    plt.plot(times, (results[:, 0]), 'b', label='velocity(t)')
    plt.plot(times, results[:, 1], 'g', label='currentDraw(t)')
    plt.legend(loc='best')
    plt.xlabel('time (sec)')
    plt.ylabel('current (A)\nvelocity (rad/s)')
    plt.grid()
    plt.show()
