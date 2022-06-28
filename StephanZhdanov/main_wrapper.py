from get_img import get_img
from get_field import get_field
from get_bin import get_bin
from get_coords import get_coords
from get_robot_coords import get_robot_coords
from make_picture import make_picture, draw_boxes
from config import *
from get_bin_neuralnet import inference_frame
from robot_control import ManipulatorRobot
import cv2
import numpy as np
import time
from model_classification import CustomConvNet
import torch

rectangles = []
cubes_puted = 0
one_cube_height = 0.07

def inPolygon(x, y, xp, yp):
    c=0
    for i in range(len(xp)):
        if (((yp[i] <= y and y<yp[i-1]) or (yp[i-1]<=y and y<yp[i])) and 
            (x > (xp[i-1] - xp[i]) * (y - yp[i]) / (yp[i-1] - yp[i]) + xp[i])): c = 1 - c    
    return c
 
 

def click_wrapper(event, x, y, flags, param):
    global rectangles
    if event == cv2.EVENT_LBUTTONDOWN:
        for rect in rectangles:
            x_click = x - 500
            y_click = y - 500
            x = rect[0]
            y = rect[1]
            xp = rect[0] + rect[2]
            yp = rect[1] + rect[3]
            print(x, y, xp, yp)
            print(x_click, y_click)

def recognition(window, net=None):
    global rectangles, cubes_puted
    model_classificator = torch.load("model_final.p", map_location=torch.device("cpu"))
    model_classificator.eval()
    #robot = ManipulatorRobot(home_pos_apply=False)
    robot = None #debug
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_wrapper)

    while True:
        img = get_img(CAMERA_IP)
        #img = cv2.imread("tests/image.png")
        field = get_field(img)
        #field = cv2.imread("dataset/1655371596.4197195.jpg")
        #img = field.copy()
        if field is not None:
            if DETECTION_TYPE == 0:
                mask = get_bin(field)
            else:
                mask = inference_frame(field, net, INFERENCE_DEVICE)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            data_objects = get_coords(field, mask, 286, 200, model_classificator)
            #coords, rectangles
            field = draw_boxes(field, data_objects)
            
            robot_coords = get_robot_coords(data_objects)
            window.label_objects_cnt["text"] = f"Objects count: {len(robot_coords)}"
            window.label_objects_coords["text"] = f"Objects coords: {robot_coords}"
            
            outimage = make_picture(data_objects, img, field)
            cv2.imshow('image', outimage)
            key = cv2.waitKey(10)
            if key == 27:
                exit()
            elif key == ord("g"):
                print("Move")
                print(robot_coords)
                
                state = robot.robot.getl()
                state[0] = robot_coords[0][0]
                state[1] = robot_coords[0][1]
                state[2] = 0.15
                robot.robot_go(state)
                state_2 = robot.robot.getl()
                state_2[2] = 0.02

                robot.robot.movel(state_2, acc=0.2, vel=0.3)
                print(state_2)

                robot.gripper.gripper_action(200)
                state_2[2] = 0.2
                robot.robot.movel(state_2, acc=0.2, vel=0.3)
                robot.go_pos_file("positions/green_box", acc=0.2, vel=0.3)
                robot.home_state()
            elif key == ord("h"):
                robot.home_state()
            elif key == ord("s"):
                name = str(time.time())
                cv2.imwrite(f"dataset/{name}.jpg", dataset_field)
            elif key == ord("a"):
                for object_ in robot_coords:
                    object_class = object_[2]
                    if object_class in [1, 2, 3]:
                        state = robot.robot.getl()
                        state[0] = object_[0]
                        state[1] = object_[1]
                        state[2] = 0.15
                        robot.robot_go(state)
                        state_2 = robot.robot.getl()
                        state_2[2] = 0.02

                        robot.robot.movel(state_2, acc=0.4, vel=0.3)

                        robot.gripper.gripper_action(155)
                        state_2[2] = 0.2
                        robot.robot.movel(state_2, acc=0.4, vel=0.3)
                        
                        if object_class == 3:
                            robot.go_pos_file("positions/green_box", acc=0.3, vel=0.3)
                        elif object_class == 1:
                            robot.go_pos_file("positions/orange_box", acc=0.3, vel=0.3)
                        elif object_class == 2:
                            robot.go_pos_file("positions/yeelow_box", acc=0.3, vel=0.3)

                        robot.home_state()
                    elif object_class in [0, 4]:
                        print("Cube")
                        if object_class == 0:
                            state = robot.robot.getl()
                            #state[0] = object_[0]
                            #state[1] = object_[1]
                            #state[2] = 0.15
                            state[3] = 0
                            state[4] = 3.14
                            state[5] = 0
                            robot.robot.movel(state, vel=0.3, acc=0.3)


                            state = robot.robot.getl()
                            state[0] = object_[0]
                            state[1] = object_[1] - 35
                            state[2] = 0.15
                            robot.robot_go(state)

                            

                            
                            state_2 = robot.robot.getl()
                            state_2[0] = object_[0]
                            state_2[1] = object_[1]
                            state_2[2] = 0.03

                            robot.robot_go(state_2)

                            

                            robot.gripper.gripper_action(255)


                            state_2 = robot.robot.getl()
                            state_2[0] = object_[0]
                            state_2[1] = object_[1]
                            state_2[2] = 0.15

                            robot.robot_go(state_2)

                           
                            robot.robot.movel([-0.472, 0, 0.15, 0, 3.14, 0], vel=0.3, acc=0.2, wait=True)
                            robot.robot.movel([-0.472, 0, 0.031 + cubes_puted * 0.067, 0, 3.14, 0], vel=0.3, acc=0.2, wait=True)
                            robot.gripper.gripper_action(0)
                            robot.robot.movel([-0.472, 0, 0.15, 0, 3.14, 0], vel=0.3, acc=0.2, wait=True)
                            robot.home_state()
                            cubes_puted += 1

                            


                           

                        elif object_class == 4:
                            print("45 deg")
                            state = robot.robot.getl()
                            #state[0] = object_[0]
                            #state[1] = object_[1]
                            #state[2] = 0.15
                            state[3] = 1.035
                            state[4] = -2.968
                            state[5] = 0
                            robot.robot.movel(state, vel=0.3, acc=0.3)

                            state = robot.robot.getl()
                            state[0] = object_[0]
                            state[1] = object_[1]
                            state[2] = 0.15
                            robot.robot_go(state)




                            state_2 = robot.robot.getl()
                            state_2[0] = object_[0]
                            state_2[1] = object_[1]
                            state_2[2] = 0.03

                            robot.robot_go(state_2)

                            robot.gripper.gripper_action(255)


                            state_2 = robot.robot.getl()
                            state_2[0] = object_[0]
                            state_2[1] = object_[1]
                            state_2[2] = 0.15

                            robot.robot_go(state_2)


                            state = robot.robot.getl()
                            #state[0] = object_[0]
                            #state[1] = object_[1]
                            #state[2] = 0.15
                            state[3] = 0
                            state[4] = 3.14
                            state[5] = 0
                            robot.robot.movel(state, vel=0.3, acc=0.3, wait=True)

                            robot.robot.movel([-0.472, 0, 0.15, 0, 3.14, 0], vel=0.3, acc=0.2, wait=True)
                            robot.robot.movel([-0.472, 0, 0.031 + cubes_puted * 0.067, 0, 3.14, 0], vel=0.3, acc=0.2, wait=True)
                            robot.gripper.gripper_action(0)
                            robot.robot.movel([-0.472, 0, 0.15, 0, 3.14, 0], vel=0.3, acc=0.2, wait=True)
                            robot.home_state()

                            cubes_puted += 1


                        






                
