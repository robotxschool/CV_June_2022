import urx, time
import numpy as np
import math3d as m3d

from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
robot = urx.Robot('192.168.137.60')

def save_pos(fname):
    p = robot.getl()
    np.savetxt(fname, np.array(p))
save_pos("orange_box")
