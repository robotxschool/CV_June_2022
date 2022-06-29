import cv2
import numpy as np

def make_picture(coords_orange,coords_green,coords_yellow,robot_coords_orange,robot_coords_green,robot_coords_yellow, img, field, frames_orange,frames_green,frames_yellow, robot_start):
    black = np.zeros((500, 1000, 3), dtype='uint8')
    cv2.circle(black, (500, 500), 100, (125, 125, 0), thickness=-1)
    cv2.circle(black, (500, 500), 500, (20, 75, 0), thickness=1)
    cv2.rectangle(black, (500-143, 100), (500+143, 300), (255,0,0), 2)
    for i,c in enumerate(coords_orange):
        x = 500+c[0]-143
        y = 500-(c[1]+200)
        cv2.circle(black, (x, y), 20, (39,127,255), thickness=-1)
        cv2.putText(black, f"x: {robot_coords_orange[i][0]}, y:{robot_coords_orange[i][1]}",
                        (x+20,y+5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (39,127,255), 2, cv2.LINE_AA)
    for x,y,w,h in frames_orange:
        cv2.rectangle(field, (x, y), (x+w, y+h), (39,127,255), 2)


    for i,c in enumerate(coords_green):
        x = 500+c[0]-143
        y = 500-(c[1]+200)
        cv2.circle(black, (x, y), 20,(0,127,0), thickness=-1)
        cv2.putText(black, f"x: {robot_coords_green[i][0]}, y:{robot_coords_green[i][1]}",
                        (x+20,y+5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,127,0), 2, cv2.LINE_AA)
    for x,y,w,h in frames_green:
        cv2.rectangle(field, (x, y), (x+w, y+h), (0,127,0), 2)

    for i,c in enumerate(coords_yellow):
        x = 500+c[0]-143
        y = 500-(c[1]+200)
        cv2.circle(black, (x, y), 20, (0,200,200), thickness=-1)
        cv2.putText(black, f"x: {robot_coords_yellow[i][0]}, y:{robot_coords_yellow[i][1]}",
                        (x+20,y+5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,200,200), 2, cv2.LINE_AA)
    for x,y,w,h in frames_yellow:
        cv2.rectangle(field, (x, y), (x+w, y+h), (0,200,200), 2)

    photos = cv2.hconcat([img,field])
    h = photos.shape[0]/(photos.shape[1]/1000)
    photos = cv2.resize(photos,(1000,int(h)))
    all_images = cv2.vconcat([photos,black])
    if robot_start:
        cv2.putText(all_images, f"GO!",
                            (10,420),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2, cv2.LINE_AA)
    else:
        cv2.putText(all_images, f"Robot is waiting...",
                            (10,420),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2, cv2.LINE_AA)
        
    return all_images
