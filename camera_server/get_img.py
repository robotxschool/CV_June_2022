import cv2
import numpy as np
import requests


def get_img(ip='127.0.0.1'):
    url = f'http://{ip}/image.jpg'
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, -1)
    
