# Target Acquisition

## About the Project
This is an application of the [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) reinforcement learning algorithm to learn voltage-level control of a two-wheeled Differential Drive mobile robot.  Normally, voltage-level control is handled by dedicated controllers running at kilohertz speeds.  Then, higher level path planners can command the vehicle using velocity or position setpoints.  In this project, I show a simple MLP with 2 hidden layers, running at 20 Hz, can learn control of a highly nonlinear vehicle.  This model operates on a series of 4 85x48 binary images (state space dimensionality: 16320) and outputs a continous value for each motor.  For this project, the model is tasked with facing a square target in the provided binary image.  The reward provided is the increase or decrease in robot heading, compared to the heading exactly focused on the target.

## Project Structure
This repo 2 main sections:
1) DDPG Folder: An existing pytorch implementation of DDPG with slight modifications from [this repo](https://github.com/schneimo/ddpg-pytorch).
2) Simulator Folder: A custom Differential Drive, DC Brushed Motor, and Pinhole Camera simulator.

## Using this repository
First, install dependencies from the requirements.txt.  This can be done easily with pip:
`pip install -r requirements.txt`

For training the model from scratch, simply run the train.py file.  Results can optionally be logged to weights and biases with the --wandb flag.
`python train.py`

To run a pretrained model, first download the checkpoints from [here](https://drive.google.com/file/d/1LBug3RBT_DMytyRYjuC9JXjfZyQxGatr/view?usp=sharing) and extract into the project directory.  Then run all cells in the evaluate.ipynb notebook.  At the bottom, a video showing the model controlling a robot should appear.

## Results from Model Training

DDPG is a rather unstable algorithm.  This can result in the policy converging and collapsing repeatedly as shown by the reward graph below.  Interestingly, because the task of facing a target is rather open-ended, the algorithm converges on a few different policies over the training period.

![reward graph](https://github.com/NathanMalta/TargetAcquisition/blob/master/media/imgs/reward.png)

Initially, the algorithm performs very poorly

https://github.com/NathanMalta/TargetAcquisition/assets/5125254/c3e84b67-807e-4d81-8aaa-696cbbcfdc32

Around epoch 1800, the first useful policy of moving forward and facing the target emerges.  Note that this corresponds to a spike in the reward graph as well:

https://github.com/NathanMalta/TargetAcquisition/assets/5125254/0c3b9aa8-402d-4222-b909-74ae18cbd6a8

At epoch 2000, this policy changes slightly, resulting in underdamped control of the system:

https://github.com/NathanMalta/TargetAcquisition/assets/5125254/3f9bb71a-c888-450d-a72d-2c779a977aee

Near epoch 2400, a new policy emerges, with the robot facing the target and driving backwards:

https://github.com/NathanMalta/TargetAcquisition/assets/5125254/d26d7cda-a5cf-40fe-8e93-8d0eb354c885

At epoch 3000, underdamped behavior of this backwards policy also emerges

https://github.com/NathanMalta/TargetAcquisition/assets/5125254/34aecc7b-63f7-4737-84de-6f1d27d3174d

