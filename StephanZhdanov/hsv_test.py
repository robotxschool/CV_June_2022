import cv2
import get_img, get_field
import numpy as np
import math

#15 81 62
#45 255 255
#Yellow
h1 = 0
s1 = 67
v1 = 46
h2 = 81
s2 = 255
v2 = 255
while True:
    try:
        img = get_img.get_img("77.37.184.204")
        #img = cv2.imread("cube.jpg")
        img = get_field.get_field(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        gray = cv2.inRange(hsv, h_min, h_max)

        kernel = np.ones((7, 7), 'uint8')
        img_bin = cv2.erode(gray, kernel, iterations=10)
        img_bin = cv2.dilate(gray, kernel, iterations=10)

        cv2.imshow("Gray", gray)


        edges = cv2.Canny(gray,90,450,apertureSize = 3)
        cv2.imshow("Edges", edges)
        lines = cv2.HoughLines(edges,1,np.pi / 25, 50)
        rho = lines[0][0][0]
        theta = lines[0][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        print(pt1, pt2)
        y2 = pt2[1]
        y1 = pt1[1]
        x2 = pt2[0]
        x1 = pt1[0]
        #angle = math.atan(y2 - y1 / x2 - x1)
        angle = degrees = ((math.atan2(y2 - y1,x2 - x1) + 2 * np.pi) * 180 / np.pi) % 360
        print(angle)
        if angle > 200:
            print("Normal position")
            status = "90 degrees position"
        else:
            print("45 degreed")
            status = "45 degrees rotated"
        img = cv2.putText(img, status, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 1, cv2.LINE_AA)


        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)



        cv2.imshow('result', img) 

        key = cv2.waitKey(300)
        if key == ord("e"):
            break
    except:
        pass
