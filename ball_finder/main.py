import cv2
import numpy as np
from get_img import get_img
from get_field import get_field
from get_bin import get_bin
from get_coords import get_coords
from get_robot_coords import get_robot_coords
from make_picture import make_picture
from robot_control import robot_go

list_pos = [0] * 40
while True:
    img = get_img()
    field = get_field(img)
    if field is not None:
        mask_all, mask_orange, mask_green, mask_yellow = get_bin(field)
        coords_orange, frames_orange, coords_green, frames_green, coords_yellow, frames_yellow = get_coords(mask_orange,
                                                                                                            mask_green,
                                                                                                            mask_yellow,
                                                                                                            x_size_mm=286,
                                                                                                            y_size_mm=200)
        robot_coords_orange, robot_coords_green, robot_coords_yellow = get_robot_coords(coords_orange, coords_green,
                                                                                        coords_yellow)

        pos = len(robot_coords_orange + robot_coords_green + robot_coords_yellow) * np.sum(
            robot_coords_orange + robot_coords_green + robot_coords_yellow)
        list_pos.append(pos)
        if len(list_pos) > 40: list_pos.pop(0)
        if abs(pos - sum(list_pos) / len(list_pos)) < 2 and len(
                robot_coords_orange + robot_coords_green + robot_coords_yellow) >= 1:
            robot_start = True
        else:
            robot_start = False
        robot_img = make_picture(coords_orange,coords_green,coords_yellow,robot_coords_orange,robot_coords_green,robot_coords_yellow, img, field, frames_orange,frames_green,frames_yellow, robot_start)
        cv2.imshow('image', robot_img)
        cv2.waitKey(50)
        if robot_start:
            if len(robot_coords_orange)>0: robot_go(robot_coords_orange[0],'orange')
            elif len(robot_coords_green) > 0: robot_go(robot_coords_green[0], 'green')
            elif len(robot_coords_yellow) > 0: robot_go(robot_coords_yellow[0], 'yellow')
            list_pos = [0] * 40

    if cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()
