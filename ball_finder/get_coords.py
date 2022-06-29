import cv2
import numpy as np


def get_coords(image_orange,image_green, image_yellow,x_size_mm,y_size_mm):
    contours_orange, hierarchy = cv2.findContours(image_orange,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    centers_orange = []
    frames_orange = []
    height, width = image_orange.shape
    for contour in contours_orange:
        x, y, w, h = cv2.boundingRect(contour)
        frames_orange.append((x, y, w, h))
        x_center = x + w // 2
        y_center = y + h // 2
        x_mm = x_size_mm*x_center//width
        y_mm = y_size_mm - (y_size_mm*y_center//height)
        centers_orange.append((x_mm,y_mm))

    contours_green, hierarchy = cv2.findContours(image_green,
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    centers_green = []
    frames_green = []
    height, width = image_green.shape
    for contour in contours_green:
        x, y, w, h = cv2.boundingRect(contour)
        frames_green.append((x, y, w, h))
        x_center = x + w // 2
        y_center = y + h // 2
        x_mm = x_size_mm * x_center // width
        y_mm = y_size_mm - (y_size_mm * y_center // height)
        centers_green.append((x_mm, y_mm))

    contours_yellow, hierarchy = cv2.findContours(image_yellow,
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)
    centers_yellow = []
    frames_yellow = []
    height, width = image_yellow.shape
    for contour in contours_yellow:
        x, y, w, h = cv2.boundingRect(contour)
        frames_yellow.append((x, y, w, h))
        x_center = x + w // 2
        y_center = y + h // 2
        x_mm = x_size_mm * x_center // width
        y_mm = y_size_mm - (y_size_mm * y_center // height)
        centers_yellow.append((x_mm, y_mm))
    return centers_orange, frames_orange, centers_green,frames_green, centers_yellow,frames_yellow
