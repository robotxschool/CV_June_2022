import cv2, get_img, get_field, get_bin, get_coords
import time
import numpy as np

path = "/home/stephan/Progs/ManipulatorBall/cube_dataset/45_new/"
while True:
    img = get_img.get_img("192.168.137.114")
    img = get_field.get_field(img)
    img_orig = img.copy()
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

    h_min = np.array((0, 89, 38), np.uint8)
    h_max = np.array((80, 255, 255), np.uint8)
    img_bin = cv2.inRange(hsv, h_min, h_max)

    kernel = np.ones((5, 5), 'uint8')
    img_bin = cv2.erode(img_bin, kernel, iterations=3)
    img_bin = cv2.dilate(img_bin, kernel, iterations=3)

    contour, hierarchy = cv2.findContours(img_bin,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    x, y, w, h = cv2.boundingRect(contour[0])
    
    img_orig = img_orig[y - 20: y + h + 20, x - 20: x + w + 20]


    cv2.imshow("Dataset", img_orig)
    name = f"{path}{time.time()}.jpg"
    #cv2.imwrite(name, img_orig)
    
    key = cv2.waitKey(25)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        name = f"{path}{time.time()}.jpg"
        cv2.imwrite(name, img_orig)
        
    