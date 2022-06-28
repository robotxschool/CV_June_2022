import cv2
import numpy as np


CLASS_COLORS = {0: (50, 168, 62), 1: (19, 134, 235), 2: (19, 235, 206) , 3: (0, 255, 0), 4: (29, 99, 36)}

def make_picture(objects_data, camera_img, warped_img):
    black = np.zeros((500, 1000, 3), dtype='uint8')
    cv2.circle(black, (500, 500), 100, (125, 125, 0), thickness=-1)
    cv2.circle(black, (500, 500), 500, (20, 75, 0), thickness=1)
    cv2.rectangle(black, (500 - 143, 100), (500 + 143, 300), (0, 0, 255))
    for obj in objects_data:
        x, y = obj["centroid_coord"]
        x = 500 + x - 143
        y = 500 - (y + 200)
        cv2.circle(black, (x, y), 20, CLASS_COLORS[obj["object_class"]], thickness=-1)
    #cv2.rectangle(img, pt1, pt2, color)
    cam = cv2.hconcat([camera_img, warped_img])
    cam = cv2.resize(cam, (1000, 375))
    black = cv2.vconcat([black, cam])

    return black
def draw_boxes(field, coords_boxes):
    
    for obj in coords_boxes:
        box = obj["frame_box"]
        cv2.rectangle(field, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), CLASS_COLORS[obj["object_class"]], thickness=3)
    return field