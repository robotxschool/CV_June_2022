import urx, time
import numpy as np
import math3d as m3d
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from config import ROBOT_PROTECT_RANGES, ROBOT_IP

class ManipulatorRobot:
    def __init__(self, home_pos_apply=True):
        self.robot = urx.Robot(ROBOT_IP)
        self.gripper = Robotiq_Two_Finger_Gripper(self.robot)
        if home_pos_apply:
            self.home_state()
    
    def home_state(self):
        self.gripper.gripper_action(0)
        self.go_pos_file("positions/home_pos", vel=0.3, acc=0.2)


    def go_pos_file(self, path, *args, **kwargs):
        p = np.loadtxt(path)
        self.robot.movel(p, *args, **kwargs, wait=True)

    def robot_go(self, coords, velocity=0.3, acceleration=0.2):
        x_min = ROBOT_PROTECT_RANGES[0]
        x_max = ROBOT_PROTECT_RANGES[1]
        y_min = ROBOT_PROTECT_RANGES[2]
        y_max = ROBOT_PROTECT_RANGES[3]
        x = coords[0] / 1000
        y = coords[1] / 1000
        coords[0] = x
        coords[1] = y
        if coords[2] >= 0.02:
            print(coords)
            self.robot.movel(coords, vel=velocity, acc=acceleration, wait=True)
        else:
            print("Error, bad coordinates", x, y)
        

