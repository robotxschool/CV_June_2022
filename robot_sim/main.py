import time
from locatelib import *
import threading, _thread

import sys
sys.path.insert(0, '/home/konstantin/RoboDK/Python')
from robodk.robolink import *
from robodk.robomath import *

RDK = Robolink()

robot_sim = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
if not robot_sim.Valid():
    raise Exception('No robot selected or available')

# import urx
# from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
# robot = urx.Robot('192.168.137.60')
# robot_gripper = Robotiq_Two_Finger_Gripper(robot)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

class App:
    def __init__(self):
        self.cubes_collected = 0
        self.first_run = True
        self.main_thread = threading.Thread(target=self.main)
        self.main_thread.start()
        
    def main(self):
        self.t1 = threading.Thread(target=self.camera)
        self.t1.start()

        self.t2 = threading.Thread(target=self.userinput)
        self.t2.start()
    
    def camera(self):
        while True:
            image = get_image(ip='192.168.137.114')
            
            # if self.first_run:
            #     pts_src, pts_dst = get_points(image, aruco_dict)
            #     # np.savetxt('pts_src.txt', pts_src)
            #     # np.savetxt('pts_dst.txt', pts_dst)
            #     # pts_src = np.loadtxt('pts_src.txt')
            #     # pts_dst = np.loadtxt('pts_dst.txt')
            #     self.first_run = False
            
            # straight_image = warp_image(image, pts_src, pts_dst)
            
            straight_image = get_field(image, aruco_dict)
            
            if straight_image is None:
                cv2.imshow('image', image)
            else:
                # bin_mask = find_bin_mask_difference(straight_image, back_image)
                red_mask = find_bin_mask_hsv_colors(straight_image, color='red')
                yellow_mask = find_bin_mask_hsv_colors(straight_image, color='yellow')
                green_mask = find_bin_mask_hsv_colors(straight_image, color='green')
                self.coords, rect = get_coords_meta(red_mask, yellow_mask, green_mask, 287, 200)
                self.coords = get_robot_coords(self.coords)
                
                for i, c in enumerate(rect):
                    x, y, w, h = c[0]
                    color = ((0, 0, 255), (0, 255, 255), (50, 205, 50))[self.coords[i][2]]
                    cv2.rectangle(straight_image, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)
                
                for i in range(len(self.coords)):
                    approx = rect[i][2]
                    for p in approx:
                        cv2.circle(straight_image, p[0], radius=2, color=(0, 255, 255), thickness=2)
                    cX = rect[i][1][0]
                    cY = rect[i][1][1]
                    cv2.circle(straight_image, (cX, cY), radius=2, color=(0, 0, 255), thickness=2)
                    shape = ('ball', 'cube', 'rot. cube')[self.coords[i][3]]
                    cv2.putText(straight_image, f'{i}-{shape}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(straight_image, f'{len(approx)}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
                    
                cv2.imshow('image', straight_image)

            if cv2.waitKey(500) == 27:
                break
        
        self.exit()
            
    def userinput(self):
        while True:
            try:
                num = int(input('Enter object num: '))
                # num = 0
                coords = self.coords[num]
                print(f'Moving robot to {coords}')
                    
                self.move_robot_real(coords[0], coords[1], 150)
                
                if coords[3] == 2:
                    self.move_robot_real(coords[0], coords[1], 150, rotate=45)
                    self.move_robot_real(coords[0], coords[1], 30, rotate=45)
                else:
                    self.move_robot_real(coords[0], coords[1], 30)
                
                if coords[3] == 0:
                    self.close_gripper(strength='medium')
                else:
                    self.close_gripper()
                
                self.move_robot_real(coords[0], coords[1], 150)
            
                if coords[3] == 0:
                    if coords[2] == 0:
                        self.move_red_box()
                    elif coords[2] == 1:
                        self.move_yellow_box()
                    elif coords[2] == 2:
                        self.move_green_box()
                    self.open_gripper()
                else:
                    self.cubes_collected += 1
                    self.put_square(self.cubes_collected)
                    self.move_start_pos()
                    
            except ValueError:
                print('ValueError')
            except IndexError:
                print('IndexError')
                self.move_start_pos()
                time.sleep(5)
                self.exit()
            except AttributeError:
                self.move_start_pos()
                self.exit()
    
    def movel(self, coord, *args, **kwargs):
        x, y, z, u, v, w = coord
        target = UR_2_Pose([x * 1000, y * 1000, z * 1000, u, v, w])
        try:
            robot_sim.MoveL(target)
        except TargetReachError:
            print('Cannot reach target')
            pass
        
    # def movel(self, *args, **kwargs):
    #     robot.movel(*args, **kwargs)
        
    def move_robot_real(self, x, y, z, rotate=0):
        if not rotate:
            self.movel([x/1000, y/1000, z/1000, 0, 3.142, 0], 0.3, 0.2, wait=True)
        elif rotate == 45:
            self.movel([x/1000, y/1000, z/1000, 1.035, -2.968, 0], 0.3, 0.2, wait=True)
        
    def open_gripper(self):
        pass
        # robot_gripper.gripper_action(0)
        
    def close_gripper(self, strength='hard'):
        pass
        # if strength == 'hard':
        #     robot_gripper.gripper_action(255)
        # elif strength == 'medium':
        #     robot_gripper.gripper_action(150)
        
    def move_start_pos(self):
        self.movel([-0.10871767195377022, 0.27994419455442154, 0.21588923565573276, 0, 3.14, 0], 0.3, 0.2, wait=True)
        
    def move_red_box(self):
        self.movel([-3.936378009951659873e-01, 2.759808607357815968e-01, 0.15, 2.221, 2.221, 0], 0.3, 0.2, wait=True)
        
    def move_yellow_box(self):
        self.movel([-3.837950140950165123e-02, 3.080104613778973932e-01, 0.15, 2.221, 2.221, 0], 0.3, 0.2, wait=True)
        
    def move_green_box(self):
        self.movel([-2.348482416464362510e-01, 2.926413067363354270e-01, 0.15, 2.221, 2.221, 0], 0.3, 0.2, wait=True)
    
    def put_square(self, num):
        self.movel([-0.472, 0, 0.15, 0, 3.14, 0], 0.3, 0.2, wait=True)
        self.movel([-0.472, 0, 0.025 + 0.06 * (num-1), 0, 3.14, 0], 0.3, 0.2, wait=True)
        self.open_gripper()
        self.movel([-0.472, 0, 0.15, 0, 3.14, 0], 0.3, 0.2, wait=True)
    
    def exit(self): # stop main thread
        cv2.destroyAllWindows()
        _thread.interrupt_main()
            
app = App()
