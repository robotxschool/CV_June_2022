
import cv2
import numpy as np
from get_img import get_img
from get_field import get_field
def nothing(*arg):
    pass

cv2.namedWindow( "result" ) 
cv2.namedWindow( "settings" )
cv2.createTrackbar('h1', 'settings', 0, 180, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 180, 180, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)

img = get_img('77.37.184.204')
img = get_field(img)

while True:
    
    h,w,_=img.shape
    img=cv2.resize(img,(w,h))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
 
    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')
    img_bin = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
    cv2.imshow('result', img_bin)
    cv2.imshow('original', img)
    ch = cv2.waitKey(5)
    if ch == 27:
        break
cv2.destroyAllWindows()
