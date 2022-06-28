import cv2
import numpy as np
import get_coords

def get_bin(image):
    '''
    yellow_h1 = 0
    yellow_s1 = 45
    yellow_v1 = 71
    yellow_h2 = 255
    yellow_s2 = 255
    yellow_v2 = 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV )

    h_min = np.array((yellow_h1, yellow_s1, yellow_v1), np.uint8)
    h_max = np.array((yellow_h2, yellow_s2, yellow_v2), np.uint8)
    img_bin = cv2.inRange(hsv, h_min, h_max)
    kernel = np.ones((5, 5), 'uint8')
    img_bin = cv2.erode(img_bin, kernel, iterations=8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=8)
    cv2.imshow("A", img_bin)
    yellow_cords, frames = get_coords.get_coords(img_bin, x_size_mm=286, y_size_mm=200)
    print(yellow_cords)
    return yellow_cords, frames
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_min = np.array((0, 89, 38), np.uint8)
    h_max = np.array((80, 255, 255), np.uint8)
    img_bin = cv2.inRange(hsv, h_min, h_max)
    cv2.imshow("A", img_bin)
    kernel = np.ones((5, 5), 'uint8')
    img_bin = cv2.erode(img_bin, kernel, iterations=10)
    img_bin = cv2.dilate(img_bin, kernel, iterations=10)
    return img_bin
    
    
