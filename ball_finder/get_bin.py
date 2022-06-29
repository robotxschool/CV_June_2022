import cv2
import numpy as np


def get_bin(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), 'uint8')
    all_min = (0,100,80)
    all_max = (180,255,255)
    img_bin_all = cv2.inRange(hsv,all_min,all_max)
    img_bin_all = cv2.erode(img_bin_all, kernel, iterations=10)
    img_bin_all = cv2.dilate(img_bin_all, kernel, iterations=10)
    
    img_bin_green = cv2.inRange(hsv, (30, 59, 56), (70, 255, 255))
    img_bin_green = cv2.erode(img_bin_green, kernel, iterations=10)
    img_bin_green = cv2.dilate(img_bin_green, kernel, iterations=10)
    
    img_bin_yellow = cv2.inRange(hsv, (13, 59, 56), (33, 255, 255))
    img_bin_yellow = cv2.erode(img_bin_yellow, kernel, iterations=10)
    img_bin_yellow = cv2.dilate(img_bin_yellow, kernel, iterations=10)
    
    img_bin_orange = cv2.inRange(hsv, (0, 59, 56), (17, 255, 255))
    img_bin_orange = cv2.erode(img_bin_orange, kernel, iterations=10)
    img_bin_orange = cv2.dilate(img_bin_orange, kernel, iterations=10)
    
    return img_bin_all, img_bin_orange, img_bin_green, img_bin_yellow
    
