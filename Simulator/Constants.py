import math

#camera constants (intrinsic parameters)
HFOV = math.radians(64.4)
VFOV = math.radians(36.2)
IMG_WIDTH = int(1280 / 15) # 85
IMG_HEIGHT = int(720 / 15) # 48

#Target parameters
TARGET_POINTS = [[1, 1, 10], [1, -1, 10], [-1, -1, 10], [-1, 1, 10]]

#Motor Constants (for an FRC CIM motor)
Kt = 0.018396947 #motor torque coefficient
D = 8.89926E-05 #Viscous damping coefficient
J = 0.0000775 #Motor rotor moment of intertia
Kv = 0.021056238 #motor voltage coefficient
R = 0.091603053 #motor winding resistance
L = 0.000059 #Motor winding inductance

#Vehicle Constants

#vehicle measurements
halfWheelbase = 0.5 #meters, half the horizontal distance between the wheels (aka L)
wheelAxisToCg = 0 #meters, the perpendicular distance between the line the wheels lie on and the center of mass of the drivetrain (aka d)
wheelDiameter = 0.3 #meters, diameter of a wheel
drivetrainWidth = 1 #meters, width of the drivetrain (length parallel to wheel axis)
drivetrainHeight = 1 #meters, height of the drivetrain (length perpendicular to wheel axis)

#vehicle intrinsic parameters
gearRatio = 1/26.667

#masses
massDriveTrain = 60 #kg, mass of the drivetrain (aka mc)
massWheel = 0.1 #kg, mass of one wheel (aka mw)
massTotal = massDriveTrain + 2 * massWheel # mass of the total system (aka m)

#Moment of Inertias (rotational)
moiDriveTrain = (1/12) * massDriveTrain * (drivetrainWidth ** 2 + drivetrainHeight ** 2) #moment of intertia of the drivetrain around the vertical axis at the center of mass - assumes the drivetrain is a rectangular plate (aka Ic)
moiWheelAxial = (1/2) * massWheel * (wheelDiameter/2)**2 #moment of inertia of a wheel around its driven axis - assumes wheels are disks (aka Iw)
moiWheelParallel = (1/4) * massWheel * (wheelDiameter/2)**2 #moment of inertia of a wheel around its diameter (perpendicular to driven axis) (aka Im)
moiTotal = moiDriveTrain + 2 * moiWheelParallel

#reinforcement learning constants
MAX_SIM_TIME = 5
NUM_FRAMES_STACKED = 4

#constants for DDPG exclusively
WIN_THRESHOLD = 10 #reward threshold where the simulation is considered to be beaten


