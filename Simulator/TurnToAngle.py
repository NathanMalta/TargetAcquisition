import numpy as np
import math

from DriveSimWithMotors import DriveSimWithMotors
import matplotlib.pyplot as plt

class TurnToAngle():
    def __init__(self):
        self.driveSim = DriveSimWithMotors([0,0, 0,0,0, 0,0], self.leftMotorVoltage, self.rightMotorVoltage)
        self.setpoint = 0

        self.lastRecalcError = None
        self.lastRecalcTime = None
        self.lastRecalcOutput = None
        self.integralWindup = 0

        self.kP = 2.5
        self.kI = 0
        self.kD = 0.18

        self.voltageOutputs = []
        self.timeOutputs = []

        self.updateRate = 20
        self.updateDt = 1 / self.updateRate
    
    def leftMotorVoltage(self, y, t):
        return -self.getDesiredOutput(y, t)
    
    def rightMotorVoltage(self, y, t):
        out = self.getDesiredOutput(y, t)
        self.voltageOutputs.append(out)
        self.timeOutputs.append(t)
        return out
    
    def getDesiredOutput(self, y, t):
        theta = y[2]

        if self.lastRecalcOutput == None:
            #this is the first pass, only use proportional control
            error = self.setpoint - theta
            out = error * self.kP
            out = max(min(out, 1), -1)

            #store current values
            self.lastRecalcTime = t
            self.lastRecalcError = error
            self.lastRecalcOutput = out
            
            return out * 12
        
        if t - self.lastRecalcTime > self.updateDt:
            # it's time to update the entire controller again
            error = self.setpoint - theta
            deriv_error = (self.lastRecalcError - error) / self.updateDt

            self.integralWindup += error * self.updateDt * self.kI
            self.integralWindup = max(min(self.integralWindup, 1), -1)

            out = error * self.kP + self.integralWindup - deriv_error * self.kD
            out = max(min(out, 1), -1) * 12

            #store current values
            self.lastRecalcTime = t
            self.lastRecalcError = error
            self.lastRecalcOutput = out

            return out
        else:
            #reuse the last output
            return self.lastRecalcOutput

    def setSetpoint(self, setpoint):
        # self.pid.setSetpoint(setpoint)
        self.setpoint = setpoint
    
    def run(self):
        results = self.driveSim.getOutputsSolveIVP([0, 10])
        print(results)
    
        plt.plot(results.t, results.y[0, :], 'y', label='vel(t)')
        plt.plot(results.t, results.y[1, :], 'c', label='w(t)')
        plt.plot(results.t, results.y[2, :], 'b', label='theta(t)')
        plt.plot(self.timeOutputs, self.voltageOutputs, 'm', label='voltage(t)')
        # plt.plot(results.t, results.y[5, :], 'r', label='left_i(t)')
        # plt.plot(results.t, results.y[6, :], 'g', label='right_i(t)')

        plt.legend(loc='best')
        plt.xlabel('time (sec)')
        plt.grid()
        plt.show()

if __name__ == '__main__':
    turn = TurnToAngle()
    turn.setSetpoint(10)
    turn.run()
